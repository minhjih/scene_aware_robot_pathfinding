"""
CSI Map Precomputation using Sionna 1.2.1 Ray Tracing.

Loads valid grid positions from the scene JSON (scenes/*.json),
then computes per-position MIMO channel tensors + SINR via ray tracing.
Results saved to data/csi_map.pkl for reuse in RL training.

Coordinate system: Sionna Y-forward, Z-up.
Grid index [xi, yi] maps to world position [xi*2+1, height, -1.5+yi*(-2)].

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/precompute_csi.py [--scene car_factory_2line]

Sionna 1.2.1 key API changes from 0.18.0:
  - scene.compute_paths() → PathSolver()(scene, ...)
  - from sionna.phy.channel import cir_to_ofdm_channel  (was sionna.channel)
  - paths.cir(out_type="tf") returns tensors WITHOUT batch dim → must add
  - PathSolver uses samples_per_src (not num_samples), diffuse_reflection (not scattering)
"""
import os, sys, pickle, time, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # GPU 0 has hardware error; use GPU 1 (RTX 3090)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ── Sionna 1.2.1 imports ──────────────────────────────────────────────────────
# sionna.rt uses Mitsuba 3 / Dr.Jit backend (auto-initializes CUDA variant)
from sionna.rt import (
    load_scene    as rt_load_scene,
    PathSolver,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
)
# sionna.phy.channel still uses TensorFlow for CIR→OFDM conversion
import tensorflow as tf
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

from src.config import (
    CARRIER_FREQ, N_TX, N_RX, C_IN, T_WIN, MAX_DEPTH, FFT_SIZE, NUM_DATA_SC,
    AP_POSITIONS, P_TX, NOISE_VAR, RT_NUM_SAMPLES, RT_BATCH_SIZE,
    DATA_DIR, MERGED_SCENE_XML, BANDWIDTH, SCENE_DIR, GRID_COORDS_JSON,
    ROBOT_HEIGHT,
)
from src.scene_utils import build_merged_scene_xml, load_grid_positions
from src.wifi_layer import select_mcs, get_ru_type, compute_throughput

# ── Module-level PathSolver (created once, reused) ────────────────────────────
_PATH_SOLVER = PathSolver()

# ── Sub-carrier frequencies for 802.11ax 20 MHz ──────────────────────────────
def _build_sc_frequencies() -> tf.Tensor:
    """
    Sub-carrier centre frequencies for the NUM_DATA_SC data subcarriers.
    802.11ax 20 MHz: data SCs ±117 around carrier, skip DC and guard tones.
    """
    sc_idxs = np.arange(-FFT_SIZE // 2, FFT_SIZE // 2)
    valid = np.zeros(FFT_SIZE, dtype=bool)
    valid[FFT_SIZE // 2 - 117: FFT_SIZE // 2]       = True
    valid[FFT_SIZE // 2 + 1  : FFT_SIZE // 2 + 118] = True
    idxs  = sc_idxs[valid][:NUM_DATA_SC]
    freqs = CARRIER_FREQ + idxs * (BANDWIDTH / FFT_SIZE)
    return tf.constant(freqs, dtype=tf.float32)


SC_FREQUENCIES = _build_sc_frequencies()


# ── Channel extraction ────────────────────────────────────────────────────────

def _extract_h(a_batch: tf.Tensor,
               tau_batch: tf.Tensor,
               rx_slot: int) -> tf.Tensor:
    """
    Extract the frequency-domain channel for one receiver slot.

    In Sionna 1.2.1 with single-antenna arrays (num_cols=1):
      a_batch   shape: [1, num_rx, 1, N_APs, 1, num_paths, 1]   (complex64)
      tau_batch shape: [1, num_rx, N_APs, num_paths]             (float32, synthetic_array=True)

    Returns H_complex: (N_RX, N_TX, K) = (1, N_APs, K) complex64 tensor.
    N_TX = len(AP_POSITIONS) = 4, one channel per AP.
    """
    # Select the j-th receiver
    a_j   = a_batch[:, rx_slot:rx_slot+1, ...]   # [1, 1, 1, N_APs, 1, paths, 1]
    tau_j = tau_batch[:, rx_slot:rx_slot+1, ...]  # [1, 1, N_APs, paths]

    try:
        # normalize=False: preserve actual path-loss amplitude so that
        # near-AP positions have large |H| and far/blocked ones have small |H|.
        # normalize=True (previous) discarded all distance-dependent attenuation,
        # causing bf_gain ≈ 1–4 everywhere → SINR > 100 dB at all positions.
        h = cir_to_ofdm_channel(SC_FREQUENCIES, a_j, tau_j, normalize=False)
        # h shape: [1, 1, 1, N_APs, 1, 1, K]
        h = tf.squeeze(h)   # → [N_APs, K] or [K] (if N_APs=1)

        # Force target shape (N_RX=1, N_TX=N_APs, K)
        target = N_RX * N_TX * NUM_DATA_SC
        h_flat = tf.reshape(h, [-1])
        # Pad with zeros if fewer elements than needed
        if h_flat.shape[0] < target:
            h_flat = tf.pad(h_flat, [[0, target - h_flat.shape[0]]])
        h_flat = h_flat[:target]
        return tf.reshape(h_flat, [N_RX, N_TX, NUM_DATA_SC])
    except Exception as e:
        raise RuntimeError(f"_extract_h failed for slot {rx_slot}: {e}") from e


def _postbf_sinr(H_complex: tf.Tensor,
                 p_tx: float = P_TX,
                 noise_var: float = NOISE_VAR) -> tf.Tensor:
    """
    SVD beamforming SINR per subcarrier.

    H_complex: (N_rx, N_tx, K) — complex64, where N_tx = N_APs
    Returns:   (K,)            — float32 SINR (linear)

    With N_rx=1, N_tx=N_APs, Hk = [1, N_APs].  SVD gives singular value =
    ||h_k||_2 (MRC gain combining all APs), which is the optimal receive gain.
    """
    sinr_list = []
    for k in range(NUM_DATA_SC):
        Hk      = H_complex[:, :, k]           # (N_rx=1, N_tx=N_APs)
        s       = tf.linalg.svd(Hk, compute_uv=False)
        bf_gain = tf.square(tf.abs(s[0]))       # dominant singular value squared
        sinr_list.append((bf_gain * p_tx) / noise_var)
    return tf.stack(sinr_list)                  # (K,)


# ── Fallback: free-space path loss ────────────────────────────────────────────

def _fallback_channel(position: np.ndarray):
    """Distance-based FSPL channel when ray tracing fails."""
    import math
    H_mag = np.zeros((N_RX, N_TX, NUM_DATA_SC), dtype=np.float32)
    sinr  = np.zeros(NUM_DATA_SC, dtype=np.float32)
    lam   = 3e8 / CARRIER_FREQ

    for i, ap in enumerate(AP_POSITIONS):
        # Use navigation-plane distance (columns 0 and 2 = world_x, world_y)
        d    = max(np.linalg.norm(
                   np.array([position[0], position[2]]) - np.array([ap[0], ap[2]])
               ), 1.0)
        fspl = (lam / (4 * math.pi * d)) ** 2
        gain = math.sqrt(fspl)

        for k in range(NUM_DATA_SC):
            H_mag[0, i % N_TX, k] += gain / N_TX

        sinr_k = (gain ** 2 * P_TX) / NOISE_VAR
        sinr  += sinr_k / len(AP_POSITIONS)

    return H_mag, sinr


# ── CSI tensor formatter ──────────────────────────────────────────────────────

def to_cnn_input(H_mag: np.ndarray, t_win: int = T_WIN) -> np.ndarray:
    """
    H_mag: (N_rx=1, N_tx=N_APs, K) → CNN input (C_in=N_rx*N_tx, K, T_win).
    Each time slice gets a small multiplicative noise to simulate
    quasi-static fading variation across the T_win packet window.
    """
    base = H_mag.reshape(C_IN, NUM_DATA_SC)          # (C_in, K)
    slices = [
        base * (1.0 + 0.02 * np.random.randn(*base.shape))
        for _ in range(t_win)
    ]
    return np.stack(slices, axis=-1).astype(np.float32)  # (C_in, K, T)


# ── Batch ray-trace ───────────────────────────────────────────────────────────

def _compute_batch(scene, positions_batch: np.ndarray):
    """
    Place all receivers in `positions_batch`, run Sionna 1.2.1 PathSolver
    once, return list of (H_mag, sinr) per position.

    Sionna 1.2.1 API:
        solver = PathSolver()  (no-arg constructor)
        paths  = solver(scene, max_depth=..., samples_per_src=..., ...)
        a, tau = paths.cir(num_time_steps=1, out_type="tf")
        # a:   complex TF tensor [num_rx, 1, N_APs, 1, paths, 1]  (NO batch dim)
        # tau: float TF tensor   [num_rx, N_APs, paths]            (synthetic_array=True)
        # → add batch dim before passing to cir_to_ofdm_channel
    """
    rx_names = []
    for j, pos in enumerate(positions_batch):
        name = f"rx_{j}"
        scene.add(Receiver(name=name, position=pos.tolist()))
        rx_names.append(name)

    results = []
    try:
        # Sionna 1.2.1: PathSolver.__call__ replaces scene.compute_paths()
        paths = _PATH_SOLVER(
            scene,
            max_depth         = MAX_DEPTH,
            specular_reflection = True,
            diffuse_reflection  = False,      # was scattering=False
            refraction          = True,
            diffraction         = False,       # wedge diffraction
            edge_diffraction    = False,
            samples_per_src   = int(RT_NUM_SAMPLES),   # was num_samples
            synthetic_array   = True,          # default; reduces complexity
        )

        # paths.cir() in 1.2.1: returns (a, tau) WITHOUT batch dimension
        a_tf, tau_tf = paths.cir(
            num_time_steps   = 1,
            normalize_delays = True,
            out_type         = "tf",
        )
        # a_tf   shape: [num_rx, 1, N_APs, 1, num_paths, 1]   complex64
        # tau_tf shape: [num_rx, N_APs, num_paths]             float32

        # Add batch=1 dimension required by cir_to_ofdm_channel
        a_batch   = tf.expand_dims(a_tf,   axis=0)   # [1, num_rx, 1, N_APs, 1, paths, 1]
        tau_batch = tf.expand_dims(tau_tf, axis=0)   # [1, num_rx, N_APs, paths]

        n_rx_total = int(a_tf.shape[0])

        for j in range(len(positions_batch)):
            slot = min(j, n_rx_total - 1)
            try:
                h = _extract_h(a_batch, tau_batch, slot)
                sinr  = _postbf_sinr(h)
                H_mag = tf.abs(h).numpy()
                results.append((H_mag, sinr.numpy()))
            except Exception as e:
                print(f"    [warn] slot {j}: {e}")
                results.append(_fallback_channel(positions_batch[j]))

    except Exception as e:
        print(f"  [warn] Batch ray-trace failed: {e} — using FSPL fallback")
        for pos in positions_batch:
            results.append(_fallback_channel(pos))

    # Remove all receiver objects added for this batch
    for name in rx_names:
        try:
            scene.remove(name)
        except Exception:
            pass

    return results


# ── Scene loader ──────────────────────────────────────────────────────────────

def load_scene(scene_xml: str = MERGED_SCENE_XML):
    """
    Build merged XML if needed, then load and configure the Sionna 1.2.1 scene.

    Sionna 1.2.1 API:
        - load_scene(xml_path) returns a Scene object (same as before)
        - scene.frequency = freq  (same)
        - scene.tx_array / rx_array use PlanarArray (same)
        - Transmitter constructor: position (list OK), look_at (kwarg, still works)
    """
    if not os.path.exists(scene_xml):
        build_merged_scene_xml()

    scene = rt_load_scene(scene_xml)
    scene.frequency = CARRIER_FREQ

    # Single-antenna per AP/robot for clean per-AP channel model:
    #   N_TX=4 APs × 1 antenna = 4 independent SISO AP channels
    #   N_RX=1 robot antenna
    #   C_IN = N_RX * N_TX = 4 CNN input channels
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,              # single antenna per AP
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,              # single antenna per robot
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )

    # Add APs as transmitters (position format: [world_x, world_z, world_y])
    for i, pos in enumerate(AP_POSITIONS):
        # look_at points AP antenna toward the ground plane (z=0)
        look_at_pos = [pos[0], 0.0, pos[2]]
        scene.add(Transmitter(
            name    = f"ap_{i}",
            position = pos,                 # [x, z, y] list accepted by Mitsuba
            look_at  = look_at_pos,
        ))

    n_aps = len(AP_POSITIONS)
    print(f"[precompute] Scene loaded | {n_aps} APs | freq={CARRIER_FREQ/1e9:.2f} GHz")
    return scene


# ── Top-view render ───────────────────────────────────────────────────────────

def _render_topview(scene, grid_positions: np.ndarray, scene_name: str) -> None:
    """Render a top-down verification image and save to scenes/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Scene centre in (world_x, world_y) — grid_positions columns: [wx, wz, wy]
    cx = float((grid_positions[:, 0].min() + grid_positions[:, 0].max()) / 2)
    cy = float((grid_positions[:, 2].min() + grid_positions[:, 2].max()) / 2)
    # Slight Y-offset avoids gimbal lock when looking straight down (Z-up convention)
    z_cam = 100.0

    cam = Camera(
        position=[cx, z_cam, cy - 5.0],   # [world_x, height, world_y-5m offset]
        look_at=[cx, 0.0, cy],            # look at scene centre at ground level
    )

    out_path = os.path.join(SCENE_DIR, f"{scene_name}_topview_check.png")
    try:
        fig = scene.render(
            camera=cam,
            fov=70,                # covers ~80×80 m scene from 100 m altitude
            resolution=(1920, 1920),
            num_samples=256,
            show_devices=True,    # draw AP transmitter positions on image
        )
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[precompute] Top-view check image saved → {out_path}")
    except Exception as e:
        print(f"[precompute] Warning: top-view render failed ({e})")


# ── Main precompute routine ───────────────────────────────────────────────────

def precompute_csi(scene_name: str = "car_factory_2line"):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load valid grid positions from JSON
    grid_json = os.path.join(SCENE_DIR, f"{scene_name}_trim_grid_gr_coords.json")
    grid_positions = load_grid_positions(grid_json, z=ROBOT_HEIGHT)
    N_total = len(grid_positions)
    print(f"[precompute] Scene: {scene_name} | Valid grid cells: {N_total}")
    print(f"             world_x: {grid_positions[:,0].min():.0f}–{grid_positions[:,0].max():.0f} m")
    print(f"             world_y: {grid_positions[:,2].min():.0f}–{grid_positions[:,2].max():.0f} m")
    print(f"             world_z: {grid_positions[:,1].mean():.2f} m (robot height)")

    # Checkpoint path for resume support
    ckpt_path = os.path.join(DATA_DIR, f"csi_map_{scene_name}_ckpt.pkl")

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        csi_map   = ckpt["csi_map"]
        sinr_map  = ckpt["sinr_map"]
        start_idx = ckpt["next_idx"]
        print(f"[precompute] Resuming from index {start_idx}/{N_total}")
    else:
        csi_map, sinr_map, start_idx = {}, {}, 0

    # Build merged XML and load scene
    build_merged_scene_xml()
    scene = load_scene()
    _render_topview(scene, grid_positions, scene_name)
    t0    = time.time()

    for batch_start in range(start_idx, N_total, RT_BATCH_SIZE):
        batch_end = min(batch_start + RT_BATCH_SIZE, N_total)
        batch_pos = grid_positions[batch_start:batch_end]

        batch_results = _compute_batch(scene, batch_pos)

        for j, (H_mag, sinr) in enumerate(batch_results):
            idx = batch_start + j
            csi_map[idx]  = to_cnn_input(H_mag)   # (C_in, K, T)
            sinr_map[idx] = sinr                    # (K,)

        # Checkpoint every 10 batches
        if (batch_start // RT_BATCH_SIZE) % 10 == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump({"csi_map": csi_map, "sinr_map": sinr_map,
                             "next_idx": batch_end}, f)

        elapsed = time.time() - t0
        rate    = batch_end / max(elapsed, 1.0)
        eta     = (N_total - batch_end) / max(rate, 1e-3)
        print(f"  [{batch_end:4d}/{N_total}] {batch_end/N_total*100:5.1f}%  "
              f"elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s")

    # Save final result
    out_path = os.path.join(DATA_DIR, f"csi_map_{scene_name}.pkl")
    final = {"csi_map": csi_map, "sinr_map": sinr_map,
             "grid_positions": grid_positions, "scene_name": scene_name}
    with open(out_path, "wb") as f:
        pickle.dump(final, f)

    # Also save to default path for downstream scripts
    with open(os.path.join(DATA_DIR, "csi_map.pkl"), "wb") as f:
        pickle.dump(final, f)

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    print(f"[precompute] Done → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="car_factory_2line",
                        choices=["car_factory_2line", "car_factory_2line_narrow",
                                 "car_factory_2line_wide", "car_factory_3line",
                                 "car_factory_3line_narrow", "car_factory_3line_wide"])
    args = parser.parse_args()
    precompute_csi(args.scene)
