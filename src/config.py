"""
CA-MAPF Configuration
All constants and hyperparameters used across the project.
"""
import os

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_DIR      = os.path.join(PROJECT_ROOT, "scenes")
DATA_DIR       = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

# Scene assets
SCENE_LAYOUT_DIR   = os.path.join(PROJECT_ROOT, "car_factory_2line")
MATERIAL_XML_PATH  = os.path.join(PROJECT_ROOT, "car_factory_integrate_xml",
                                  "car_factory_standard.xml")
MERGED_SCENE_XML   = os.path.join(SCENE_DIR, "factory_standard.xml")

# Extra meshes (human / mobile robot) kept in integrate_xml/meshes
EXTRA_MESHES_DIR   = os.path.join(PROJECT_ROOT, "car_factory_integrate_xml", "meshes")
HUMAN_MESH         = os.path.join(EXTRA_MESHES_DIR, "human_clean.ply")
ROBOT_MESH         = os.path.join(EXTRA_MESHES_DIR, "robot_clean.ply")

# Scene-specific valid grid JSON
SCENE_NAME         = "car_factory_2line"
GRID_COORDS_JSON   = os.path.join(SCENE_DIR,
                                  f"{SCENE_NAME}_trim_grid_gr_coords.json")

CSI_MAP_PATH       = os.path.join(DATA_DIR, "csi_map.pkl")

# ── GPU ───────────────────────────────────────────────────────────────────────
CUDA_DEVICE = "1"   # Use GPU 1 only

# ── Sionna scene coordinate system ───────────────────────────────────────────
# Sionna: Y-forward, Z-up.  Position tuple order = [x, z, y].
#
# Grid index → world coordinate rule:
#   world_x = xi * 2 + 1
#   world_z = height  (constant per agent type)
#   world_y = -1.5 + yi * (-2)     ← negative; grows more negative as yi ↑
#
# Sionna position = [world_x, world_z, world_y]
#
GRID_RESOLUTION = 2.0       # effective metres per grid step (2 m in world)
ROBOT_HEIGHT    = 0.75      # m  (mobile robot, world_z)
HUMAN_HEIGHT    = 1.0       # m  (pedestrian mesh, world_z)
AP_HEIGHT       = 5.0       # m  (ceiling AP, world_z)

# Grid index spans 0..39 in both xi and yi
GRID_X_MAX = 39
GRID_Y_MAX = 39


def gi_to_world(xi: int, yi: int, z: float) -> list:
    """Convert grid indices (xi, yi) and height z to Sionna [x, z, y] position."""
    return [float(xi * 2 + 1), float(z), float(-1.5 + yi * (-2))]


def world_to_gi(world_x: float, world_y: float):
    """Inverse: world (x, y) → nearest grid indices (xi, yi)."""
    xi = round((world_x - 1) / 2)
    yi = round((world_y + 1.5) / (-2))
    return int(xi), int(yi)


# AP grid indices (xi, yi) and their world positions [x, z, y]
AP_GRID_INDICES = [(10, 10), (10, 30), (30, 10), (30, 30)]
AP_POSITIONS = [gi_to_world(xi, yi, AP_HEIGHT) for xi, yi in AP_GRID_INDICES]

# ── 802.11ax PHY parameters ───────────────────────────────────────────────────
CARRIER_FREQ    = 5.18e9     # 5 GHz band
BANDWIDTH       = 20e6       # 20 MHz
FFT_SIZE        = 256
NUM_DATA_SC     = 234        # K: valid data subcarriers
SC_SPACING      = 78.125e3   # Hz
T_SYM           = 1 / SC_SPACING + 0.8e-6   # OFDM symbol period with GI
N_TX            = 4          # AP TX antennas (1×4 ULA)
N_RX            = 1          # Robot RX antennas
C_IN            = N_RX * N_TX  # CNN input channels = 4
T_WIN           = 16         # Observation window (packets)
MAX_DEPTH       = 5          # Max ray-tracing reflections

# ── MCS table ─────────────────────────────────────────────────────────────────
# {mcs_idx: (modulation_order M, coding_rate r, bits_per_subcarrier)}
MCS_TABLE = {
    0:  (2,    0.5,   0.5),
    1:  (4,    0.5,   1.0),
    2:  (4,    0.75,  1.5),
    3:  (16,   0.5,   2.0),
    4:  (16,   0.75,  3.0),
    5:  (64,   2/3,   4.0),
    6:  (64,   0.75,  4.5),
    7:  (64,   5/6,   5.0),
    8:  (256,  0.75,  6.0),
    9:  (256,  5/6,   6.667),
    10: (1024, 0.75,  7.5),
    11: (1024, 5/6,   8.333),
}

# RU type index → number of data subcarriers
RU_DATA_SUBCARRIERS = {
    0: 234,   # 242-tone (full 20 MHz)
    1: 102,   # 106-tone
    2: 48,    # 52-tone
    3: 24,    # 26-tone
    4: 0,     # Overflow: unschedulable
}

# Minimum SINR (dB) required per MCS
MCS_SINR_THRESHOLD = {
    0: -1.0, 1: 2.0,  2: 4.5,  3: 7.0,
    4: 9.5,  5: 11.0, 6: 13.0, 7: 15.0,
    8: 18.0, 9: 20.0, 10: 23.0, 11: 25.0,
}

# ── RL / PPO hyperparameters ──────────────────────────────────────────────────
LR            = 1e-4    # reduced from 3e-4: loss spikes up to 20 indicated too-large steps
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_EPS      = 0.1     # tightened from 0.2: prevents large policy updates that caused divergence
VALUE_COEF    = 0.25   # lowered from 0.5: large initial MSE (returns≈-30~-100) caused divergence
ENTROPY_COEF  = 0.05    # raised: entropy collapsed to ~0 at ep 60/220 causing loss spikes
N_EPOCHS      = 10
BATCH_SIZE    = 256    # larger mini-batch for 32-robot diversity
ROLLOUT_LEN   = 6144   # covers ~2.7 full 32-robot episodes (6144 / 2240 ≈ 2.7)
MAX_EPISODES  = 4000
N_ROBOTS      = 32     # 32 AGVs moving concurrently (2× density → meaningful congestion)
# Each episode allows N_ROBOTS × ~70 individual moves.
# World step = 2 m; longest diagonal in 80×80 m scene ≈ 113 m → ~57 steps.
# 32 robots × 70 moves each = 2240 total steps.
MAX_STEPS     = 2240
LAMBDA_COMM   = 1.0
R_GOAL        = 10.0

# ── Human agent parameters ────────────────────────────────────────────────────
N_HUMANS                = 16     # pedestrians (scaled with robot density, 32 AGVs)
HUMAN_SPEED             = 0.8    # m/s  (pedestrian walking speed)
HUMAN_DETECTION_RANGE   = 10.0   # m  (max sensor range for robot)
ROBOT_FOV_DEG           = 120.0  # degrees (field of view half-angle = 60°)
HUMAN_SHADOW_DB         = 8.0    # dB shadowing loss when human blocks LOS
HUMAN_SHADOW_RADIUS     = 1.5    # m  (radius within which human blocks LOS)

# ── Communication parameters ──────────────────────────────────────────────────
L_DATA      = 500e3    # 500 KB — H.264 1080p video I-frame (heavy task)
# Sweet spot with T_POLL=0.1s:
#   n_cell ≤ 4 (106-tone): T_comm ≤ 80 ms → no penalty (ours_ppo safe zone)
#   n_cell  5–8  (52-tone): T_comm ≈ 227 ms → +127 ms penalty per step
#   n_cell 9–16 (26-tone): T_comm ≈ 567 ms → +467 ms penalty per step
# With 32 AGVs / 4 APs (avg 8/AP), most positions see n_cell=5-8 →
# ours_ppo gains large advantage by routing through low-congestion areas.
T_PENALTY   = 16e-3   # 802.11ax trigger-frame retransmit penalty (s)
# IEEE 802.11ax (Wi-Fi 6) at 5.18 GHz, UNII-1 indoor factory:
#   FCC Part 15.407 UNII-1: conducted TX power ≤ 17 dBm (50 mW).
#   AGV/STA uplink (embedded device, 5 GHz): 13 dBm (20 mW) = 0.020 W.
P_TX        = 0.020   # W  (13 dBm — robot STA UL, 5 GHz UNII-1)

# Thermal noise at AP receiver:  N = k·T·B·F
#   k = 1.38e-23 J/K,  T = 290 K (IEEE std temp),  B = 20 MHz
#   NF = 10 dB (robot/embedded Rx noise figure)  → F = 10.0
#   N = 1.38e-23 × 290 × 20e6 × 10 = 8.0e-13 W ≈ 1.0e-12 W
NOISE_VAR   = 1.0e-12  # W  (-90 dBm — realistic AP Rx noise floor)

# Polling interval: time between consecutive TF (Trigger Frame) slots for one robot.
# = 1 / SLAM_keyframe_upload_rate.  If T_comm > T_POLL the robot misses its next
# scheduled uplink slot (polling-deadline violation → comm_penalty).
# 10 Hz SLAM upload → T_POLL = 0.1 s.
T_POLL = 0.1           # s  (polling interval / uplink deadline)

# Maximum achievable 802.11ax throughput (MCS11, 242-tone RU) for tp_loss normalisation.
# Empirical ceiling ~150 Mbps; use 200 as safe upper bound.
MAX_THROUGHPUT_MBPS = 200.0

# Same-AP peer observation: how many nearest same-AP robots to track per robot.
# With 32 AGVs / 4 APs, average 8/AP.  Padded with zeros when fewer are present.
# Gives model visibility into near-future n_cell (who is about to contend for the AP).
MAX_AGENTS_PER_AP = 8

# 802.11ax HE PPDU preamble overhead (μs → s)
# HE-STF(8) + HE-LTF(8+4*N_LTF) + HE-SIG-A(8) + HE-SIG-B + L-preamble(20)
# Typical MU-OFDMA trigger-based PPDU preamble ≈ 60 μs
T_PREAMBLE  = 60e-6   # s  (HE PPDU preamble for trigger-based UL MU-OFDMA)

# ── Ray-tracing precompute settings ──────────────────────────────────────────
RT_NUM_SAMPLES = int(5e5)   # samples per ray-trace batch
RT_BATCH_SIZE  = 20         # receivers per batch

# ── Evaluation scenarios ──────────────────────────────────────────────────────
SCENARIOS = {
    # n_cell_range: concurrent robots per AP at evaluation time.
    # With 32 AGVs / 4 APs, mean load = 8 per AP → avg 52-tone RU.
    # L_DATA = 500 KB: penalty triggers at n_cell ≥ 5 (T_comm > T_POLL=0.1s).
    #
    # S1: low load → no penalty; all methods behave similarly (baseline)
    # S2: medium congestion → significant penalty; ours_ppo routes around
    # S3: high congestion + humans → large penalty; comm-awareness critical
    # S4: extreme congestion + max humans → 26-tone, +467ms/step penalty
    "S1": {"n_cell_range": (1,  4), "dynamic_obs": False, "n_humans":  0,
           "n_robots": 32, "desc": "Low-density static (32 AGVs, n_cell 1–4, no penalty)"},
    "S2": {"n_cell_range": (5,  8), "dynamic_obs": False, "n_humans":  0,
           "n_robots": 32, "desc": "Medium congestion (32 AGVs, 52-tone, +127ms/step)"},
    "S3": {"n_cell_range": (5, 16), "dynamic_obs": True,  "n_humans": 16,
           "n_robots": 32, "desc": "High congestion + dynamic (32 AGVs + 16 humans)"},
    "S4": {"n_cell_range": (9, 16), "dynamic_obs": True,  "n_humans": 32,
           "n_robots": 32, "desc": "Extreme congestion (32 AGVs + 32 humans, 26-tone)"},
}

BASELINES = ["shortest_path", "snr_threshold", "random_walk",
             "paper1_mlp", "ours_supervised", "ours_ppo"]

METRICS = ["mission_latency", "throughput_actual", "overflow_rate",
           "path_length", "success_rate", "comm_penalty_ratio"]
