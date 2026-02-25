#!/usr/bin/env python3
"""
demo_scenarios.py — Per-scenario top-view comparison demo for CA-MAPF.

Saves results/demo_S1.mp4 … demo_S4.mp4, one per scenario.
Each video shows 32 AGVs navigating under controlled communication load.
Status bar displays live: n_cell, RU-type, R(Mbps), T_comm, penalty.

n_cell_override per scenario (representative midpoint):
  S1  n_cell= 2  242-tone  T_comm≈ 28ms   no penalty     (baseline)
  S2  n_cell= 6   52-tone  T_comm≈227ms  +127ms/step    (medium stress)
  S3  n_cell=10   26-tone  T_comm≈567ms  +467ms/step    (high + humans)
  S4  n_cell=14   26-tone  T_comm≈567ms  +467ms/step    (extreme + humans)

Usage:
    python src/demo_scenarios.py                      # all 4
    python src/demo_scenarios.py --scenarios S2 S4    # specific
    python src/demo_scenarios.py --fps 20 --max_steps 600
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
    N_ROBOTS, MAX_STEPS, AP_POSITIONS,
    CHECKPOINT_DIR, CSI_MAP_PATH, RESULTS_DIR,
    MCS_SINR_THRESHOLD, SCENARIOS, T_POLL,
    HUMAN_MESH, ROBOT_MESH,
)
from src.env import CAMAPFEnv
from src.models import DualBranchNet
from src.wifi_layer import get_ru_type

# ── Scenario → controlled n_cell (representative midpoint of range) ──────────
SCENARIO_NCELL = {
    "S1":  2,   # 242-tone, T_comm≈28ms,  no penalty
    "S2":  6,   # 52-tone,  T_comm≈227ms, +127ms/step
    "S3": 10,   # 26-tone,  T_comm≈567ms, +467ms/step  (+humans)
    "S4": 14,   # 26-tone,  T_comm≈567ms, +467ms/step  (+max humans)
}

RU_NAMES = {0: "242-tone", 1: "106-tone", 2: "52-tone", 3: "26-tone", 4: "Overflow"}

# ── Colour palette ────────────────────────────────────────────────────────────
_ROBOT_CMAP  = plt.get_cmap('tab20')
_HUMAN_COLOR = '#FF6B35'
_GOAL_COLOR  = '#2ECC71'
_AP_COLOR    = '#9B59B6'
_DONE_COLOR  = '#27AE60'
TRAIL_LEN    = 25


# ── PLY utilities (copied from demo_topview) ──────────────────────────────────
def _read_ply_vertices(path: str) -> np.ndarray:
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
        is_bin    = any('binary_little_endian' in l for l in header)
        n_props   = len(props)
        is_double = any('property double' in l for l in header)
        if is_bin:
            fmt = '<' + ('d' if is_double else 'f') * n_props
            sz  = struct.calcsize(fmt)
            raw = np.frombuffer(f.read(n_verts * sz),
                                dtype=np.float64 if is_double else np.float32)
            return raw.reshape(n_verts, n_props)[:, :3].astype(np.float32)
        rows = [list(map(float, f.readline().decode().split()))
                for _ in range(n_verts)]
        return np.array(rows, dtype=np.float32)[:, :3]


def _topdown_hull(path: str, target_radius: float, subsample: int = 8000):
    verts = _read_ply_vertices(path)
    if len(verts) > subsample:
        idx   = np.random.default_rng(42).choice(len(verts), subsample, replace=False)
        verts = verts[idx]
    xy = verts[:, :2].copy()
    xy -= xy.mean(axis=0)
    hull   = ConvexHull(xy)
    hverts = xy[hull.vertices]
    max_r  = np.sqrt((hverts ** 2).sum(axis=1)).max()
    if max_r > 1e-6:
        hverts *= (target_radius / max_r)
    return np.vstack([hverts, hverts[0]])


def _rotate_hull(hull, heading):
    angle = np.arctan2(float(heading[1]), float(heading[0]))
    c, s  = np.cos(angle), np.sin(angle)
    return hull @ np.array([[c, -s], [s, c]]).T


def draw_factory_background(ax, nav_xy):
    valid_set = {(round(float(p[0])), round(float(p[1]))) for p in nav_xy}
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
    ax.add_patch(mpatches.Rectangle((-1, -81), 82, 82,
                                    fill=False, edgecolor='#888888',
                                    linewidth=2, zorder=1))


def _build_sinr_grid(grid_positions, sinr_map):
    sinr_grid = np.full((40, 40), np.nan, dtype=np.float32)
    for idx, sinr_arr in sinr_map.items():
        wp = grid_positions[idx]
        xi = round((float(wp[0]) - 1) / 2)
        yi = round((-float(wp[2]) - 1.5) / 2)
        if 0 <= xi < 40 and 0 <= yi < 40:
            sinr_grid[yi, xi] = float(10 * np.log10(np.mean(sinr_arr) + 1e-12))
    return sinr_grid


# ── Episode rollout (captures comm info per step) ─────────────────────────────
def collect_frames(env: CAMAPFEnv, model: DualBranchNet,
                   max_steps: int, n_humans: int) -> list:
    """
    Run one episode and collect per-step snapshots.
    Snapshot includes comm_info: n_cell, R_actual, T_comm, comm_penalty.
    """
    nav_xy = env._nav_xy
    frames = []
    obs, _ = env.reset()

    last_comm = {"n_cell": 0, "R_actual": 0.0, "T_comm": 0.0, "comm_penalty": 0.0}

    for step_i in range(max_steps):
        humans_pos, humans_hdg = [], []
        if env.human_manager:
            for a in env.human_manager.agents:
                humans_pos.append(nav_xy[a.pos_idx].copy())
                humans_hdg.append(a.heading.copy())

        frames.append({
            'step':         step_i,
            'positions':    list(env.positions),
            'headings':     [h.copy() for h in env.headings],
            'goals':        list(env.goals),
            'human_pos':    humans_pos,
            'human_hdg':    humans_hdg,
            'active_id':    env._current_robot_id,
            'robots_done':  {i for i in range(env.n_robots)
                             if env.positions[i] == env.goals[i]},
            'comm_info':    dict(last_comm),
        })

        from src.config import MAX_AGENTS_PER_AP
        C      = torch.FloatTensor(obs['csi']).unsqueeze(0)
        s      = torch.FloatTensor(obs['state']).unsqueeze(0)
        h_obs  = torch.FloatTensor(
            obs.get('human_obs', np.zeros(n_humans * 2))
        ).unsqueeze(0)
        p_obs  = torch.FloatTensor(obs.get('p_t', np.zeros(4))).unsqueeze(0)
        ap_obs = torch.FloatTensor(
            obs.get('ap_obs', np.zeros(MAX_AGENTS_PER_AP * 2))
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _, _ = model(C, s, h_obs if n_humans > 0 else None,
                                 p_obs, ap_obs)
            action = Categorical(logits=logits).sample()

        obs, _, term, trunc, info = env.step(action.item())
        last_comm = {
            "n_cell":       info.get("n_cell", 0),
            "R_actual":     info.get("R_actual", 0.0),
            "T_comm":       info.get("T_comm", 0.0),
            "comm_penalty": info.get("comm_penalty", 0.0),
        }

        if term or trunc:
            h2, hh2 = [], []
            if env.human_manager:
                for a in env.human_manager.agents:
                    h2.append(nav_xy[a.pos_idx].copy())
                    hh2.append(a.heading.copy())
            frames.append({
                'step':        step_i + 1,
                'positions':   list(env.positions),
                'headings':    [h.copy() for h in env.headings],
                'goals':       list(env.goals),
                'human_pos':   h2,
                'human_hdg':   hh2,
                'active_id':   -1,
                'robots_done': {i for i in range(env.n_robots)
                                if env.positions[i] == env.goals[i]},
                'comm_info':   dict(last_comm),
            })
            break

    n_done = len(frames[-1]['robots_done'])
    print(f"  [collect] {len(frames)} steps, {n_done}/{env.n_robots} goals reached")
    return frames


# ── Animation builder ─────────────────────────────────────────────────────────
def build_animation(frames, nav_xy, robot_hull, human_hull,
                    sinr_grid, scenario_name, scenario_desc,
                    n_cell_repr, fps=15):

    n_robots = len(frames[0]['positions'])
    ru_name  = RU_NAMES.get(get_ru_type(n_cell_repr), "?")

    fig, ax = plt.subplots(figsize=(11, 11), dpi=110)
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#1A1A2E')

    draw_factory_background(ax, nav_xy)

    # ── SINR radiomap overlay ─────────────────────────────────────────────────
    SINR_VMIN = float(MCS_SINR_THRESHOLD[0]) - 9.0
    SINR_VMAX = float(MCS_SINR_THRESHOLD[11]) + 10.0
    sinr_masked = np.ma.masked_invalid(sinr_grid)
    cmap_sinr   = plt.get_cmap('RdYlGn').copy()
    cmap_sinr.set_bad(alpha=0.0)
    im = ax.imshow(
        sinr_masked, origin='upper', extent=[0, 80, -80, 0],
        cmap=cmap_sinr, vmin=SINR_VMIN, vmax=SINR_VMAX,
        alpha=0.25, zorder=1, interpolation='nearest',
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label('SNR (dB)', color='#CCCCCC', fontsize=7)
    for mcs_idx in [0, 4, 7, 11]:
        t = MCS_SINR_THRESHOLD[mcs_idx]
        if SINR_VMIN <= t <= SINR_VMAX:
            y = (t - SINR_VMIN) / (SINR_VMAX - SINR_VMIN)
            cbar.ax.axhline(y=y, color='white', linewidth=0.6, alpha=0.6)
            cbar.ax.text(1.15, y, f'MCS{mcs_idx}', color='#CCCCCC',
                         fontsize=5, va='center', transform=cbar.ax.transAxes)
    cbar.ax.yaxis.set_tick_params(colors='#888888', labelsize=6)

    # AP markers
    for i, ap in enumerate(AP_POSITIONS):
        ax.plot(ap[0], ap[2], '^', color=_AP_COLOR, markersize=14, zorder=10,
                markeredgecolor='white', markeredgewidth=0.8)
        ax.text(ap[0] - 0.8, ap[2] + 0.8, f'AP{i+1}',
                color=_AP_COLOR, fontsize=7, zorder=11)

    ax.set_xlim(81, -1)
    ax.set_ylim(-81, 1)
    ax.set_aspect('equal')
    ax.set_xlabel('World X  (m)', color='#CCCCCC', fontsize=9)
    ax.set_ylabel('World Y  (m)', color='#CCCCCC', fontsize=9)
    ax.tick_params(colors='#888888', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    # Title: scenario name + description
    ax.set_title(
        f'CA-MAPF — {scenario_name}: {scenario_desc}\n'
        f'n_cell={n_cell_repr}  RU={ru_name}  T_POLL={T_POLL*1000:.0f}ms  '
        f'payload=500KB (H.264 1080p)',
        color='white', fontsize=10, pad=8,
    )

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor=_AP_COLOR, markersize=10,
               label='Access Point', linestyle='None'),
        mpatches.Patch(facecolor='#4FC3F7', label='AGV (active)'),
        mpatches.Patch(facecolor='#29B6F6', edgecolor='#AAAAAA',
                       linewidth=0.5, label='AGV (idle)'),
        mpatches.Patch(facecolor=_HUMAN_COLOR, label='Pedestrian'),
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor=_GOAL_COLOR, markersize=12,
               label='Goal', linestyle='None'),
    ]
    ax.legend(handles=legend_handles, loc='upper right',
              framealpha=0.4, labelcolor='white',
              facecolor='#2C2C4A', edgecolor='#555577', fontsize=7)

    # Status bar (bottom-left): step / goals
    status_txt = ax.text(
        0.01, 0.985, '', transform=ax.transAxes,
        color='white', fontsize=8, va='top',
        bbox=dict(facecolor='#2C2C4A', alpha=0.75, edgecolor='#555577', pad=4),
    )
    # Comm info bar (bottom of plot): n_cell, R, T_comm, penalty
    comm_txt = ax.text(
        0.01, 0.03, '', transform=ax.transAxes,
        color='white', fontsize=8, va='bottom',
        bbox=dict(facecolor='#1A1A2E', alpha=0.80, edgecolor='#555577', pad=4),
    )

    trails = {i: [] for i in range(n_robots)}
    dyn_artists: list = []

    def _update(fi):
        if fi == 0:
            for ri in range(n_robots):
                trails[ri].clear()

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
        ci       = snap['comm_info']

        # Robot trails
        for ri in range(n_robots):
            pxy = nav_xy[pos[ri]]
            trails[ri].append(pxy.copy())
            if len(trails[ri]) > TRAIL_LEN:
                trails[ri].pop(0)
            if len(trails[ri]) > 1:
                trail_np = np.array(trails[ri])
                n_seg = len(trails[ri]) - 1
                alphas = np.linspace(0.03, 0.25, n_seg)
                for seg in range(n_seg):
                    ln, = ax.plot(
                        trail_np[seg:seg+2, 0], trail_np[seg:seg+2, 1],
                        color=_ROBOT_CMAP(ri / n_robots),
                        alpha=float(alphas[seg]), linewidth=0.9, zorder=2,
                    )
                    dyn_artists.append(ln)

        # Goal markers
        for ri in range(n_robots):
            gxy   = nav_xy[goals[ri]]
            color = _DONE_COLOR if ri in done_set else _ROBOT_CMAP(ri / n_robots)
            star, = ax.plot(gxy[0], gxy[1], '*', color=color, markersize=11,
                            markeredgecolor='white', markeredgewidth=0.4,
                            zorder=5, alpha=0.85)
            dyn_artists.append(star)

        # Human footprints
        for hxy, hhdg in zip(h_pos, h_hdg):
            verts = _rotate_hull(human_hull, hhdg) + hxy
            patch = MplPolygon(verts, closed=True,
                               facecolor=_HUMAN_COLOR, edgecolor='#FF8C00',
                               linewidth=0.7, alpha=0.88, zorder=7)
            ax.add_patch(patch)
            dyn_artists.append(patch)

        # Robot footprints (no ID labels for 32 robots → too crowded)
        for ri in range(n_robots):
            rxy      = nav_xy[pos[ri]]
            color    = _ROBOT_CMAP(ri / n_robots)
            is_done  = ri in done_set
            is_act   = ri == active
            alpha    = 0.97 if is_act else (0.50 if is_done else 0.78)
            edge_col = 'white' if is_act else ('#00FF88' if is_done else '#888888')
            edge_w   = 1.6    if is_act else (0.8 if is_done else 0.3)

            verts = _rotate_hull(robot_hull, heads[ri]) + rxy
            patch = MplPolygon(verts, closed=True,
                               facecolor=color, edgecolor=edge_col,
                               linewidth=edge_w, alpha=alpha, zorder=8)
            ax.add_patch(patch)
            dyn_artists.append(patch)

            if is_done:
                ck = ax.text(rxy[0] + 0.6, rxy[1] + 0.6, '✓',
                             color='#00FF88', fontsize=7,
                             fontweight='bold', zorder=9)
                dyn_artists.append(ck)

        # Status bar
        n_done = len(done_set)
        status_txt.set_text(
            f"Step {snap['step']:4d}/{len(frames)-1}  │  "
            f"Goals {n_done:2d}/{n_robots} ({n_done/n_robots*100:.0f}%)  │  "
            f"Active: #{active if active >= 0 else '—'}"
        )

        # Comm info
        R   = ci['R_actual']
        Tc  = ci['T_comm'] * 1000   # ms
        pen = ci['comm_penalty'] * 1000  # ms
        nc  = ci['n_cell']
        ru  = RU_NAMES.get(get_ru_type(nc), '?')
        if pen > 0:
            comm_color = '#FF4444'
            flag = f'⚠ +{pen:.0f}ms PENALTY'
        else:
            comm_color = '#44FF88'
            flag = '✓ OK'
        comm_txt.set_text(
            f"n_cell={nc:2d}  {ru:10s}  R={R:6.1f}Mbps  "
            f"T_comm={Tc:6.1f}ms  T_POLL={T_POLL*1000:.0f}ms  {flag}"
        )
        comm_txt.set_color(comm_color)

        return dyn_artists

    ani = animation.FuncAnimation(
        fig, _update,
        frames=len(frames),
        interval=int(1000 / fps),
        blit=False, repeat=False,
    )
    return ani, fig


# ── Entry point ───────────────────────────────────────────────────────────────
def run_demos(args):
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"[demo] Loading CSI map: {args.csi_map}")
    with open(args.csi_map, 'rb') as f:
        data = pickle.load(f)
    grid_positions = data['grid_positions']
    csi_map        = data['csi_map']
    sinr_map       = data['sinr_map']

    device = torch.device('cpu')

    print("[demo] Building mesh footprints …")
    robot_hull = _topdown_hull(ROBOT_MESH, target_radius=0.70, subsample=12000)
    human_hull = _topdown_hull(HUMAN_MESH, target_radius=0.40)

    print("[demo] Building SINR radiomap …")
    sinr_grid = _build_sinr_grid(grid_positions, sinr_map)

    for sc_name in args.scenarios:
        if sc_name not in SCENARIOS:
            print(f"[demo] Unknown scenario {sc_name}, skipping.")
            continue

        sc_cfg      = SCENARIOS[sc_name]
        n_cell_repr = SCENARIO_NCELL[sc_name]
        n_humans    = sc_cfg.get('n_humans', 0)
        n_robots    = sc_cfg.get('n_robots', N_ROBOTS)
        desc        = sc_cfg['desc']

        print(f"\n[demo] ── {sc_name}: {desc}")
        print(f"         n_cell_override={n_cell_repr}  n_humans={n_humans}")

        # Load model (reload per scenario to allow diff checkpoints later)
        model = DualBranchNet(n_humans=n_humans).to(device)
        if os.path.exists(args.checkpoint):
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt['model'], strict=False)
            print(f"         Checkpoint ep {ckpt.get('episode','?')} loaded")
        else:
            print("         WARNING: checkpoint not found — random policy")
        model.eval()

        env = CAMAPFEnv(
            grid_positions, csi_map, sinr_map,
            n_robots=n_robots,
            max_steps=args.max_steps,
            n_humans=n_humans,
            dynamic_obs=sc_cfg.get('dynamic_obs', False),
            n_cell_override=n_cell_repr,
        )
        nav_xy = env._nav_xy

        frames = collect_frames(env, model, args.max_steps, n_humans)

        print(f"  [build] Building animation …")
        ani, fig = build_animation(
            frames, nav_xy, robot_hull, human_hull,
            sinr_grid, sc_name, desc, n_cell_repr, fps=args.fps,
        )

        out_path = os.path.join(args.results_dir, f'demo_{sc_name}.mp4')
        print(f"  [encode] → {out_path}")
        writer = animation.FFMpegWriter(
            fps=args.fps, bitrate=3500,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'],
            metadata={'title': f'CA-MAPF {sc_name} Demo'},
        )
        ani.save(out_path, writer=writer, dpi=110)
        plt.close(fig)
        print(f"  [done]  {out_path}")

    print("\n[demo] All scenarios complete.")


def main():
    p = argparse.ArgumentParser(description='CA-MAPF scenario demo videos')
    p.add_argument('--checkpoint',  default=os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
    p.add_argument('--csi_map',     default=CSI_MAP_PATH)
    p.add_argument('--results_dir', default=RESULTS_DIR)
    p.add_argument('--scenarios',   nargs='+', default=list(SCENARIOS.keys()))
    p.add_argument('--max_steps',   type=int, default=800)
    p.add_argument('--fps',         type=int, default=15)
    args = p.parse_args()
    run_demos(args)


if __name__ == '__main__':
    main()
