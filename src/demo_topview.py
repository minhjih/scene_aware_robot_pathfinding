#!/usr/bin/env python3
"""
demo_topview.py — Top-view demo video of CA-MAPF.

Loads the trained MAPPO policy, runs one episode, and saves a top-down
animation to results/demo_topview.mp4.  Each agent is rendered with its
PLY mesh footprint (top-down convex-hull projection) rotated by its heading.

Usage:
    python src/demo_topview.py
    python src/demo_topview.py --max_steps 500 --output results/my_demo.mp4
"""
import os, sys, pickle, struct, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
import torch
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    N_ROBOTS, N_HUMANS, MAX_STEPS,
    AP_POSITIONS, CHECKPOINT_DIR, CSI_MAP_PATH,
    HUMAN_MESH, ROBOT_MESH, RESULTS_DIR,
    MCS_SINR_THRESHOLD,
)
from src.env import CAMAPFEnv
from src.models import DualBranchNet


# ── PLY loader + top-down convex hull ────────────────────────────────────────

def _read_ply_vertices(path: str) -> np.ndarray:
    """Parse PLY file (binary little-endian or ASCII) and return (N,3) float32."""
    with open(path, 'rb') as f:
        header, props, n_verts = [], [], 0
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header.append(line)
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
            if line.startswith('property float') or line.startswith('property double'):
                props.append(line.split()[-1])
            if line == 'end_header':
                break

        is_bin = any('binary_little_endian' in l for l in header)
        n_props = len(props)

        if is_bin:
            is_double = any('property double' in l for l in header)
            fmt = '<' + ('d' if is_double else 'f') * n_props
            sz  = struct.calcsize(fmt)
            raw = np.frombuffer(f.read(n_verts * sz),
                                dtype=np.float64 if is_double else np.float32)
            verts = raw.reshape(n_verts, n_props)[:, :3].astype(np.float32)
        else:
            rows = [list(map(float, f.readline().decode().split()))
                    for _ in range(n_verts)]
            verts = np.array(rows, dtype=np.float32)[:, :3]
    return verts   # (N, 3)  x, y, z in PLY convention


def _topdown_hull(path: str, target_radius: float,
                  subsample: int = 8000) -> np.ndarray:
    """
    Load PLY, project vertices onto the (x, y) floor plane (top-down view,
    z = up), compute convex hull, normalize so the max radial extent equals
    `target_radius` (metres in nav-plane units).

    Returns (K+1, 2) hull vertices centred at origin (polygon closed).
    """
    verts = _read_ply_vertices(path)
    if len(verts) > subsample:
        rng   = np.random.default_rng(42)
        idx   = rng.choice(len(verts), subsample, replace=False)
        verts = verts[idx]

    # Top-down projection: PLY x = left/right, PLY y = forward (z = up)
    xy = verts[:, :2].copy()
    xy -= xy.mean(axis=0)

    hull   = ConvexHull(xy)
    hverts = xy[hull.vertices]                     # (K, 2)
    max_r  = np.sqrt((hverts ** 2).sum(axis=1)).max()
    if max_r > 1e-6:
        hverts *= (target_radius / max_r)

    return np.vstack([hverts, hverts[0]])          # closed polygon (K+1, 2)


def _rotate_hull(hull: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """Rotate hull so that the PLY +x axis aligns with `heading` (2-D vector)."""
    angle = np.arctan2(float(heading[1]), float(heading[0]))
    c, s  = np.cos(angle), np.sin(angle)
    R     = np.array([[c, -s], [s, c]])
    return hull @ R.T


# ── Factory floor background ──────────────────────────────────────────────────

def draw_factory_background(ax, nav_xy: np.ndarray):
    """
    Draw the factory grid.
    • Valid (navigable) cells  → light gray fill
    • Blocked (obstacle) cells → dark gray fill
    Cell size = 2 m × 2 m aligned on odd world-coordinate centres.
    """
    valid_set = {(round(float(p[0])), round(float(p[1])))
                 for p in nav_xy}

    valid_r, block_r = [], []
    for xi in range(40):
        for yi in range(40):
            wx = float(xi * 2 + 1)
            wy = float(-1.5 + yi * (-2))
            r  = mpatches.Rectangle((wx - 1.0, wy - 1.0), 2.0, 2.0)
            if (round(wx), round(wy)) in valid_set:
                valid_r.append(r)
            else:
                block_r.append(r)

    if block_r:
        ax.add_collection(PatchCollection(
            block_r, facecolor='#3A3A3A', edgecolor='none', zorder=0))
    if valid_r:
        ax.add_collection(PatchCollection(
            valid_r, facecolor='#D8D8D8', edgecolor='#BBBBBB',
            linewidth=0.2, zorder=0))

    # Outer factory border
    ax.add_patch(mpatches.Rectangle((-1, -81), 82, 82,
                                    fill=False, edgecolor='#888888',
                                    linewidth=2, zorder=1))


# ── Episode rollout ───────────────────────────────────────────────────────────

def _build_sinr_grid(grid_positions: np.ndarray,
                     sinr_map: dict) -> np.ndarray:
    """
    Build a 40×40 mean-SINR (dB) grid for the radiomap overlay.

    grid_positions columns: [world_x, world_z, world_y]
      xi = round((world_x - 1) / 2)   ∈ [0, 39]
      yi = round((-world_y - 1.5) / 2) ∈ [0, 39]

    Returns float32 array (40, 40), NaN for blocked cells.
    """
    sinr_grid = np.full((40, 40), np.nan, dtype=np.float32)
    for idx, sinr_arr in sinr_map.items():
        wp = grid_positions[idx]
        wx = float(wp[0])    # world_x
        wy = float(wp[2])    # world_y  (nav plane column 2)
        xi = round((wx - 1) / 2)
        yi = round((-wy - 1.5) / 2)
        if 0 <= xi < 40 and 0 <= yi < 40:
            sinr_db = float(10 * np.log10(np.mean(sinr_arr) + 1e-12))
            sinr_grid[yi, xi] = sinr_db
    return sinr_grid


def collect_frames(env: CAMAPFEnv, model: DualBranchNet,
                   max_steps: int) -> list:
    """
    Run one episode with the given policy, collecting a per-step snapshot dict.
    Snapshot keys:
        step        : int
        positions   : list[int]   robot grid indices
        headings    : list[ndarray(2,)]
        goals       : list[int]
        human_pos   : list[ndarray(2,)]   nav-plane (world_x, world_y)
        human_hdg   : list[ndarray(2,)]
        active_id   : int   which robot just moved
        robots_done : set[int]
    """
    nav_xy = env._nav_xy
    frames = []

    obs, _ = env.reset()

    for step_i in range(max_steps):
        humans_pos, humans_hdg = [], []
        if env.human_manager:
            for a in env.human_manager.agents:
                humans_pos.append(nav_xy[a.pos_idx].copy())
                humans_hdg.append(a.heading.copy())

        frames.append({
            'step':       step_i,
            'positions':  list(env.positions),
            'headings':   [h.copy() for h in env.headings],
            'goals':      list(env.goals),
            'human_pos':  humans_pos,
            'human_hdg':  humans_hdg,
            'active_id':  env._current_robot_id,
            'robots_done': {i for i in range(env.n_robots)
                            if env.positions[i] == env.goals[i]},
        })

        # Policy action
        from src.config import MAX_AGENTS_PER_AP
        C      = torch.FloatTensor(obs['csi']).unsqueeze(0)
        s      = torch.FloatTensor(obs['state']).unsqueeze(0)
        h_obs  = torch.FloatTensor(
            obs.get('human_obs', np.zeros(N_HUMANS * 2))
        ).unsqueeze(0)
        p_obs  = torch.FloatTensor(
            obs.get('p_t', np.zeros(4))
        ).unsqueeze(0)
        ap_obs = torch.FloatTensor(
            obs.get('ap_obs', np.zeros(MAX_AGENTS_PER_AP * 2))
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = model(C, s, h_obs if N_HUMANS > 0 else None,
                                 p_obs, ap_obs)
            action = Categorical(logits=logits).sample()

        obs, _, term, trunc, _ = env.step(action.item())

        if term or trunc:
            # Final snapshot
            h_pos2, h_hdg2 = [], []
            if env.human_manager:
                for a in env.human_manager.agents:
                    h_pos2.append(nav_xy[a.pos_idx].copy())
                    h_hdg2.append(a.heading.copy())
            frames.append({
                'step':       step_i + 1,
                'positions':  list(env.positions),
                'headings':   [h.copy() for h in env.headings],
                'goals':      list(env.goals),
                'human_pos':  h_pos2,
                'human_hdg':  h_hdg2,
                'active_id':  -1,
                'robots_done': {i for i in range(env.n_robots)
                                if env.positions[i] == env.goals[i]},
            })
            break

    n_done = len(frames[-1]['robots_done'])
    print(f"[demo] Episode done: {len(frames)} steps, "
          f"{n_done}/{env.n_robots} goals reached")
    return frames


# ── Animation ─────────────────────────────────────────────────────────────────

# Colour palette
_ROBOT_CMAP  = plt.get_cmap('tab20')
_HUMAN_COLOR = '#FF6B35'
_GOAL_COLOR  = '#2ECC71'
_AP_COLOR    = '#9B59B6'
_DONE_COLOR  = '#27AE60'

TRAIL_LEN = 35   # steps of trail history per robot


def build_animation(frames: list,
                    nav_xy: np.ndarray,
                    robot_hull: np.ndarray,
                    human_hull: np.ndarray,
                    sinr_grid: np.ndarray = None,
                    fps: int = 15) -> animation.FuncAnimation:
    """Build a FuncAnimation from collected frames."""

    n_robots = len(frames[0]['positions'])

    # ── Figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    draw_factory_background(ax, nav_xy)

    # ── SINR radiomap overlay (transparent) ──────────────────────────────────
    # Drawn over the factory floor but under robots/humans.
    # extent = [xmin, xmax, ymin, ymax] in data coords;
    # Sionna x: right=0, left=80  → xlim(81,-1) flips axis automatically.
    if sinr_grid is not None:
        # sinr_map stores SINR (linear) = bf_gain × P_TX / NOISE_VAR.
        # sinr_grid already converted to dB via 10*log10(mean SINR).
        # Display range: 802.11ax MCS0 threshold (≈-1 dB) to MCS11 threshold (25 dB)
        # with headroom → [-10, 35] dB covers the full useful SNR operating range.
        SINR_VMIN = float(MCS_SINR_THRESHOLD[0]) - 9.0    # ≈ -10 dB
        SINR_VMAX = float(MCS_SINR_THRESHOLD[11]) + 10.0  # ≈  35 dB

        sinr_masked = np.ma.masked_invalid(sinr_grid)
        cmap = plt.get_cmap('RdYlGn').copy()
        cmap.set_bad(alpha=0.0)          # transparent for NaN / blocked cells
        im = ax.imshow(
            sinr_masked,
            origin='upper',
            extent=[0, 80, -80, 0],      # (xmin, xmax, ymin, ymax) world coords
            cmap=cmap,
            vmin=SINR_VMIN, vmax=SINR_VMAX,
            alpha=0.28,
            zorder=1,
            interpolation='nearest',
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.set_label('SNR (dB)', color='#CCCCCC', fontsize=7)
        # Mark key MCS thresholds on the colorbar
        for mcs_idx in [0, 4, 7, 11]:
            t = MCS_SINR_THRESHOLD[mcs_idx]
            if SINR_VMIN <= t <= SINR_VMAX:
                cbar.ax.axhline(y=(t - SINR_VMIN) / (SINR_VMAX - SINR_VMIN),
                                color='white', linewidth=0.6, alpha=0.6)
                cbar.ax.text(1.15, (t - SINR_VMIN) / (SINR_VMAX - SINR_VMIN),
                             f'MCS{mcs_idx}', color='#CCCCCC',
                             fontsize=5, va='center', transform=cbar.ax.transAxes)
        cbar.ax.yaxis.set_tick_params(colors='#888888', labelsize=6)

    # AP markers (static)
    for i, ap in enumerate(AP_POSITIONS):
        ax.plot(ap[0], ap[2], '^', color=_AP_COLOR,
                markersize=14, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.text(ap[0] - 0.8, ap[2] + 0.8, f'AP{i+1}',
                color=_AP_COLOR, fontsize=7, zorder=11)

    # Sionna x convention: right=0, left=80  (inverted x-axis)
    ax.set_xlim(81, -1)
    ax.set_ylim(-81, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('World X  (m)', color='#CCCCCC', fontsize=10)
    ax.set_ylabel('World Y  (m)', color='#CCCCCC', fontsize=10)
    ax.tick_params(colors='#888888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor=_AP_COLOR, markersize=10,
               label='Access Point', linestyle='None'),
        mpatches.Patch(facecolor='#4FC3F7', edgecolor='white',
                       linewidth=0.5, label='AGV (active)'),
        mpatches.Patch(facecolor='#29B6F6', edgecolor='#AAAAAA',
                       linewidth=0.5, label='AGV (idle)'),
        mpatches.Patch(facecolor=_HUMAN_COLOR, edgecolor='#FF8C00',
                       linewidth=0.5, label='Pedestrian'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=_GOAL_COLOR, markersize=12,
               label='Goal', linestyle='None'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              framealpha=0.4, labelcolor='white',
              facecolor='#2C2C4A', edgecolor='#555577',
              fontsize=8)

    ax.set_title('CA-MAPF — MAPPO Top-View Demo',
                 color='white', fontsize=13, pad=10)

    status_txt = ax.text(
        0.01, 0.985, '', transform=ax.transAxes,
        color='white', fontsize=9, va='top',
        bbox=dict(facecolor='#2C2C4A', alpha=0.7, edgecolor='#555577', pad=4),
    )

    # Per-robot trail history  (deque-like list)
    trails = {i: [] for i in range(n_robots)}

    # Dynamic artist lists (cleared each frame)
    dyn_artists: list = []

    # ── Per-frame update function ─────────────────────────────────────────
    def _update(fi: int):
        # Reset trails at frame 0 (handles multiple renders of the animation)
        if fi == 0:
            for ri in range(n_robots):
                trails[ri].clear()

        # Remove all dynamic artists from last frame
        for a in dyn_artists:
            a.remove()
        dyn_artists.clear()

        snap     = frames[fi]
        pos      = snap['positions']
        heads    = snap['headings']
        goals    = snap['goals']
        h_pos    = snap['human_pos']
        h_hdg    = snap['human_hdg']
        active   = snap['active_id']
        done_set = snap['robots_done']

        # ── Robot trails ─────────────────────────────────────────────────
        for ri in range(n_robots):
            pxy = nav_xy[pos[ri]]
            trails[ri].append(pxy.copy())
            if len(trails[ri]) > TRAIL_LEN:
                trails[ri].pop(0)
            if len(trails[ri]) > 1:
                trail_np = np.array(trails[ri])
                n_seg = len(trails[ri]) - 1
                alphas = np.linspace(0.04, 0.3, n_seg)
                for seg in range(n_seg):
                    ln, = ax.plot(
                        trail_np[seg:seg + 2, 0],
                        trail_np[seg:seg + 2, 1],
                        color=_ROBOT_CMAP(ri / n_robots),
                        alpha=float(alphas[seg]), linewidth=1.2, zorder=2,
                    )
                    dyn_artists.append(ln)

        # ── Goal markers ──────────────────────────────────────────────────
        for ri in range(n_robots):
            gxy   = nav_xy[goals[ri]]
            color = _DONE_COLOR if ri in done_set else _ROBOT_CMAP(ri / n_robots)
            star, = ax.plot(
                gxy[0], gxy[1], '*',
                color=color, markersize=13,
                markeredgecolor='white', markeredgewidth=0.5,
                zorder=5, alpha=0.9,
            )
            dyn_artists.append(star)

        # ── Human footprints ──────────────────────────────────────────────
        for hxy, hhdg in zip(h_pos, h_hdg):
            verts = _rotate_hull(human_hull, hhdg) + hxy
            patch = MplPolygon(
                verts, closed=True,
                facecolor=_HUMAN_COLOR, edgecolor='#FF8C00',
                linewidth=0.8, alpha=0.88, zorder=7,
            )
            ax.add_patch(patch)
            dyn_artists.append(patch)

        # ── Robot footprints ──────────────────────────────────────────────
        for ri in range(n_robots):
            rxy   = nav_xy[pos[ri]]
            color = _ROBOT_CMAP(ri / n_robots)

            is_active = (ri == active)
            is_done   = (ri in done_set)

            alpha     = 0.97 if is_active else (0.55 if is_done else 0.80)
            edge_col  = 'white' if is_active else ('#00FF88' if is_done else '#AAAAAA')
            edge_w    = 1.8    if is_active else (1.0 if is_done else 0.4)

            verts = _rotate_hull(robot_hull, heads[ri]) + rxy
            patch = MplPolygon(
                verts, closed=True,
                facecolor=color, edgecolor=edge_col,
                linewidth=edge_w, alpha=alpha, zorder=8,
            )
            ax.add_patch(patch)
            dyn_artists.append(patch)

            # Robot ID label
            lbl = ax.text(
                rxy[0], rxy[1], str(ri),
                ha='center', va='center',
                color='white', fontsize=6, fontweight='bold', zorder=9,
            )
            dyn_artists.append(lbl)

            # "Done" tick mark above robot
            if is_done:
                ck = ax.text(
                    rxy[0] + 0.7, rxy[1] + 0.7, '✓',
                    ha='center', va='center',
                    color='#00FF88', fontsize=8, fontweight='bold', zorder=9,
                )
                dyn_artists.append(ck)

        # ── Status bar ────────────────────────────────────────────────────
        # status_txt is a PERSISTENT axes artist (not in dyn_artists).
        # Just update its text content each frame.
        n_done = len(done_set)
        pct    = n_done / n_robots * 100
        active_str = f'#{active}' if active >= 0 else '—'
        status_txt.set_text(
            f"Step {snap['step']:4d} / {len(frames) - 1}  │  "
            f"Goals {n_done:2d}/{n_robots} ({pct:.0f}%)  │  "
            f"Active robot: {active_str}"
        )

        return dyn_artists

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(frames),
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )
    return ani, fig


# ── Entry point ───────────────────────────────────────────────────────────────

def run_demo(args):
    os.makedirs(args.results_dir, exist_ok=True)
    output_path = os.path.join(args.results_dir, 'demo_topview.mp4')

    # ── Load env ─────────────────────────────────────────────────────────────
    print(f"[demo] Loading CSI map: {args.csi_map}")
    with open(args.csi_map, 'rb') as f:
        data = pickle.load(f)
    grid_positions = data['grid_positions']
    csi_map        = data['csi_map']
    sinr_map       = data['sinr_map']

    env = CAMAPFEnv(
        grid_positions, csi_map, sinr_map,
        n_robots=N_ROBOTS, max_steps=MAX_STEPS,
        n_humans=N_HUMANS, dynamic_obs=True,
    )
    nav_xy = env._nav_xy   # (N, 2) — (world_x, world_y)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device('cpu')
    model  = DualBranchNet(n_humans=N_HUMANS).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        print(f"[demo] Checkpoint loaded (episode {ckpt.get('episode', '?')})")
    else:
        print("[demo] WARNING: checkpoint not found — using random policy")
    model.eval()

    # ── Load PLY mesh silhouettes ─────────────────────────────────────────────
    print("[demo] Building mesh footprints from PLY …")
    robot_hull = _topdown_hull(ROBOT_MESH, target_radius=0.75, subsample=12000)
    human_hull = _topdown_hull(HUMAN_MESH, target_radius=0.45)
    print(f"        robot hull: {len(robot_hull)-1} pts,  "
          f"human hull: {len(human_hull)-1} pts")

    # ── Build SINR radiomap grid ──────────────────────────────────────────────
    print("[demo] Building SINR radiomap …")
    sinr_grid = _build_sinr_grid(grid_positions, sinr_map)
    n_valid = int(np.sum(~np.isnan(sinr_grid)))
    print(f"        SINR grid 40×40, {n_valid} valid cells, "
          f"range [{np.nanmin(sinr_grid):.1f}, {np.nanmax(sinr_grid):.1f}] dB")

    # ── Run episode ───────────────────────────────────────────────────────────
    frames = collect_frames(env, model, max_steps=args.max_steps)

    # ── Build & save animation ────────────────────────────────────────────────
    print("[demo] Building animation …")
    ani, fig = build_animation(
        frames, nav_xy, robot_hull, human_hull,
        sinr_grid=sinr_grid, fps=args.fps,
    )

    print(f"[demo] Encoding MP4 → {output_path}")
    writer = animation.FFMpegWriter(
        fps=args.fps, bitrate=3000,
        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
        metadata={'title': 'CA-MAPF MAPPO Demo'},
    )
    ani.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"[demo] Done!  Video: {output_path}")


def main():
    p = argparse.ArgumentParser(description='CA-MAPF top-view demo video')
    p.add_argument('--checkpoint',   default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--csi_map',      default=CSI_MAP_PATH)
    p.add_argument('--results_dir',  default=RESULTS_DIR)
    p.add_argument('--max_steps',    type=int, default=600)
    p.add_argument('--fps',          type=int, default=15)
    args = p.parse_args()
    run_demo(args)


if __name__ == '__main__':
    main()
