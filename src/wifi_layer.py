"""
IEEE 802.11ax protocol layer:
  - MCS table and RU allocation
  - BLER curves
  - Throughput and communication delay computation
"""
import math
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MCS_TABLE, RU_DATA_SUBCARRIERS, MCS_SINR_THRESHOLD,
    T_SYM, L_DATA, T_PENALTY, T_PREAMBLE,
)

# Max users that fit in one TF round per RU type (802.11ax 20 MHz)
#   ru_type 0 → 242-tone : 1  slot  (full BW, 1 user per round)
#   ru_type 1 → 106-tone : 2  slots
#   ru_type 2 →  52-tone : 4  slots
#   ru_type 3 →  26-tone : 9  slots
MAX_SIMULTANEOUS = {0: 1, 1: 2, 2: 4, 3: 9}

# Number of RU blocks that partition the 20 MHz band per RU type.
# 9 × 26-tone  = 234 SC  (exact partition)
# 4 × 52-tone  ≈ 4 × 58-59 SC  (234 / 4, rounded)
# 2 × 106-tone = 2 × 117 SC   (exact)
# 1 × 242-tone = 1 × 234 SC   (full band)
RU_BLOCK_COUNT = {0: 1, 1: 2, 2: 4, 3: 9}


def get_ru_block_boundaries(ru_type: int, n_sc: int = 234) -> list:
    """
    Return (start, end) subcarrier index pairs for each RU block.

    The NUM_DATA_SC=234 data subcarriers are partitioned into
    RU_BLOCK_COUNT[ru_type] contiguous, non-overlapping blocks.
    First (n_sc % n_blocks) blocks get one extra subcarrier to cover remainder.

        ru_type=3: 9 × 26 SC  (9×26=234, exact)
        ru_type=2: 4 × [59,59,58,58] SC
        ru_type=1: 2 × 117 SC
        ru_type=0: 1 × 234 SC
    """
    n_blocks = RU_BLOCK_COUNT[ru_type]
    base = n_sc // n_blocks
    rem  = n_sc % n_blocks          # first `rem` blocks get one extra SC
    boundaries, s = [], 0
    for b in range(n_blocks):
        e = s + base + (1 if b < rem else 0)
        boundaries.append((s, e))
        s = e
    return boundaries


def select_best_ru_block(sinr_arr: np.ndarray, ru_type: int):
    """
    Frequency-selective RU assignment: find the contiguous RU block with the
    highest mean linear SINR and return its dB SINR + block index.

    In 802.11ax UL MU-OFDMA the AP assigns each STA to an RU block based on
    CSI feedback.  We model this as the STA getting its best available block.

    Args:
        sinr_arr : (NUM_DATA_SC,) linear SINR per subcarrier from sinr_map
        ru_type  : RU type index (0–3)

    Returns:
        (best_block_idx, sinr_ru_db)
          best_block_idx : int — which block (0-indexed) was selected
          sinr_ru_db     : float — mean dB SINR of that block
    """
    boundaries = get_ru_block_boundaries(ru_type)
    best_block  = 0
    best_sinr_lin = -np.inf
    for b, (s, e) in enumerate(boundaries):
        block_sinr_lin = float(np.mean(sinr_arr[s:e]))
        if block_sinr_lin > best_sinr_lin:
            best_sinr_lin = block_sinr_lin
            best_block    = b
    sinr_ru_db = float(10.0 * np.log10(max(best_sinr_lin, 1e-12)))
    return best_block, sinr_ru_db


def get_bler(mcs: int, sinr_db: float) -> float:
    """Approximate BLER for given MCS and SINR (dB)."""
    threshold = MCS_SINR_THRESHOLD[mcs]
    margin = sinr_db - threshold
    if margin >= 3.0:
        return 1e-4
    elif margin >= 0.0:
        return 0.01
    elif margin >= -2.0:
        return 0.1
    else:
        return 0.5


def select_mcs(sinr_db: float) -> int:
    """802.11ax link adaptation: select best MCS from SINR."""
    best_mcs = 0
    for m in range(11, -1, -1):
        if sinr_db >= MCS_SINR_THRESHOLD[m]:
            best_mcs = m
            break
    return best_mcs


def get_ru_type(n_cell: int) -> int:
    """
    Map concurrent-robot count at one AP to 802.11ax RU type.

    Based on 802.11ax 20 MHz UL MU-OFDMA RU allocation.
    With 32 AGVs / 4 APs, mean per-AP load = 8 → 52-tone territory.
    L_DATA=500KB penalty threshold: n_cell ≥ 5 (T_comm > T_POLL=0.1s).

        1–2   robots → 242-tone (full BW)   T_comm≈28ms,  no penalty
        3–4   robots → 106-tone (half BW)   T_comm≈80ms,  no penalty
        5–8   robots → 52-tone  (¼ BW)      T_comm≈227ms, +127ms penalty
        9–16  robots → 26-tone  (⅛ BW)      T_comm≈567ms, +467ms penalty
        >16   robots → capped at 26-tone (max contention; still schedulable)

    NOTE: n_cell is capped at 16 to avoid the Overflow (ru_type=4) case which
    returns R_actual=0 → T_comm=inf → reward=-inf → NaN in training.
    With 32 AGVs and realistic movement, >16/AP occurs during dense initial
    placement; treating it as max-contention (26-tone) is physically sound.
    """
    n_cell = min(n_cell, 16)   # cap: >16 treated as max-contention (26-tone)
    if n_cell <= 2:
        return 0    # 242-tone
    elif n_cell <= 4:
        return 1    # 106-tone
    elif n_cell <= 8:
        return 2    # 52-tone
    else:
        return 3    # 26-tone


def compute_throughput(mcs: int, ru_type: int, n_streams: int = 1,
                       n_ofdm_symbols: int = None) -> float:
    """
    802.11ax PHY throughput (Mbps) for trigger-based UL MU-OFDMA.

    Accounts for HE PPDU preamble overhead:
        R = N_SD * bits_per_SC * N_s * N_sym / (T_preamble + N_sym * T_sym)

    n_ofdm_symbols: OFDM data symbols per PPDU (default: derive from L_DATA).
    """
    if ru_type == 4:
        return 0.0
    N_sd = RU_DATA_SUBCARRIERS[ru_type]
    _, _, bits_per_sc = MCS_TABLE[mcs]
    bits_per_sym = bits_per_sc * N_sd * n_streams    # bits per OFDM symbol

    if n_ofdm_symbols is None:
        # Estimate symbols needed to carry L_DATA bits (rounded up)
        import math
        n_ofdm_symbols = max(1, math.ceil((L_DATA * 8) / bits_per_sym))

    t_data = n_ofdm_symbols * T_SYM
    t_total = T_PREAMBLE + t_data                    # preamble + payload
    throughput_bps = (bits_per_sym * n_ofdm_symbols) / t_total
    return throughput_bps / 1e6


def compute_comm_delay(
    R_hat: float,
    mcs: int,
    l_data: float = L_DATA,
    t_penalty: float = T_PENALTY,
) -> float:
    """
    T_comm = L_data / R_hat + P_BLER / (1 - P_BLER) * T_penalty
    """
    if R_hat <= 0:
        return float("inf")
    T_tx = (l_data * 8) / (R_hat * 1e6)
    p_bler = get_bler(mcs, MCS_SINR_THRESHOLD[mcs] + 1.5)
    p_bler = min(p_bler, 0.99)
    T_retry = (p_bler / (1.0 - p_bler)) * t_penalty
    return T_tx + T_retry


def compute_tx_power_control(
    sinr_arr: np.ndarray,
    p_rx_target: float,
    noise_var: float,
    p_tx_ref: float,
    p_tx_min: float,
    p_tx_max: float,
) -> tuple:
    """
    802.11ax open-loop UL power control via Trigger Frame Target RSSI.

    Per the standard, the STA estimates path loss from the WIDEBAND RSSI of
    the received Trigger Frame (full 20 MHz), NOT from the specific allocated
    RU block.  This is because the TF is a wideband signal and the STA has no
    separate per-block path loss estimate before the TF arrives.

    Since sinr_map stores  SINR_k = G_k * p_tx_ref / noise_var,
    the wideband path gain is:
        G_wideband = mean(SINR_k, k=0..K-1) * noise_var / p_tx_ref

    Required TX power for target received power:
        P_TX_req = p_rx_target / G_wideband

    Actual TX power (FCC / device limited):
        P_TX_act = clip(P_TX_req, p_tx_min, p_tx_max)

    SINR scale factor (uniform across all subcarriers):
        scale = P_TX_act / p_tx_ref   (≤ 1.0)

    Near robots (high G_wideband): scale < 1 → SINR reduced (power backed off)
    Far  robots (low G_wideband):  clamped at p_tx_max → scale = 1.0

    The per-RU-block SINR for MCS selection is computed AFTER applying this
    scale:  sinr_arr_pc = sinr_arr * scale  (then select_best_ru_block on that)

    Args:
        sinr_arr   : (NUM_DATA_SC,) stored linear SINR per subcarrier
        p_rx_target: target received power at AP (W)
        noise_var  : thermal noise power used in precomputation (W)
        p_tx_ref   : reference TX power used in precomputation (W)
        p_tx_min   : minimum allowed STA TX power (W)
        p_tx_max   : maximum allowed STA TX power (W)

    Returns:
        (p_tx_act, sinr_scale)
    """
    g_wideband = float(np.mean(sinr_arr)) * noise_var / p_tx_ref
    if g_wideband <= 0.0:
        return p_tx_max, 1.0

    p_tx_req = p_rx_target / g_wideband
    p_tx_act = float(np.clip(p_tx_req, p_tx_min, p_tx_max))
    scale    = p_tx_act / p_tx_ref
    return p_tx_act, scale


def assign_scheduling_round(
    sinr_self_lin: float,
    sinr_peers_lin: list,
    n_cell: int,
    ru_type: int,
    method: str,
    sinr_noise_std_db: float = 0.0,
    rng=None,
    tp_history_self: float = 1.0,
    tp_history_peers: list = None,
) -> int:
    """
    802.11ax UL MU-OFDMA: assign this robot to a TF round (1-indexed).

    The AP schedules based on a "scheduling preference" (Access Category)
    signalled by each STA, plus the AP's own (noisy) SINR measurements.

    Stochasticity
    ─────────────
    AP estimates SINR from HE-NDP sounding — limited pilots + CQI quantisation
    introduce ~1.5 dB log-normal noise (802.11ax spec).  Applied to all rank-
    based methods (not round_robin) before sorting.  Makes outcomes partially
    unpredictable → RL must learn robust positions, not just exploit noise.

    Methods (EDCA AC mapping)
    ─────────────────────────
    'round_robin' (AC_BK):
        No channel info used. Conservative worst-case: last round.
        Deterministic. No noise applied.

    'proportional_fair' (AC_BE):
        PF metric = SINR_linear / avg_historical_throughput.
        Higher ratio → gets earlier round (catch up to fair share).
        Robots behind on throughput get compensated.  History-aware.

    'max_sinr' (AC_VI):
        Robots sorted by SINR descending → best channel → round 1.
        Near-AP robots benefit from early rounds.  Throughput-greedy.

    'deadline_aware' (AC_VO):
        Robots sorted by SINR ascending → worst channel → round 1.
        Protects latency-critical weak links.  Equalises deadline risk.

    Path planning cost:
        T_eff(v) = round_idx × T_comm_round(v)
        comm_penalty(v) = max(0, T_eff(v) − T_POLL)

    Args:
        sinr_self_lin      : this robot's mean wideband linear SINR (after PC)
        sinr_peers_lin     : same-AP peers' mean wideband linear SINR (after PC)
        n_cell             : total concurrent robots at this AP
        ru_type            : RU type index (0–3)
        method             : scheduling policy ('round_robin' | 'proportional_fair'
                             | 'max_sinr' | 'deadline_aware')
        sinr_noise_std_db  : AP-side SINR estimation noise std (dB, log-normal)
        rng                : numpy.random.Generator for reproducible noise
        tp_history_self    : this robot's average historical throughput (Mbps)
        tp_history_peers   : peers' average historical throughputs (Mbps); aligned
                             with sinr_peers_lin; None → uniform 1.0 assumed

    Returns:
        round_idx : int [1, n_rounds] — TF round this robot is assigned to
    """
    max_sim  = MAX_SIMULTANEOUS.get(ru_type, 1)
    n_rounds = get_n_rounds(n_cell, ru_type)

    if n_rounds == 1 or not sinr_peers_lin:
        return 1   # single round or no peers: always round 1

    # round_robin: deterministic, no SINR info → no noise applied
    if method == 'round_robin':
        return n_rounds

    # ── Apply AP-side SINR measurement noise (log-normal) ───────────────────
    # Simulates HE-NDP sounding quantisation + estimation error.
    # round_robin is exempt because it doesn't use SINR info.
    if sinr_noise_std_db > 0.0 and rng is not None:
        def _add_noise(s: float) -> float:
            return s * 10.0 ** (rng.normal(0.0, sinr_noise_std_db) / 10.0)
        sinr_self_lin  = _add_noise(sinr_self_lin)
        sinr_peers_lin = [_add_noise(s) for s in sinr_peers_lin]

    # ── Proportional Fair ────────────────────────────────────────────────────
    if method == 'proportional_fair':
        # PF metric = current_SINR / historical_avg_throughput
        # Higher PF score → gets earlier round (compensates throughput deficit)
        tp_peers = (tp_history_peers if tp_history_peers is not None
                    else [1.0] * len(sinr_peers_lin))
        pf_self  = sinr_self_lin / max(tp_history_self, 1e-6)
        all_pf   = [(pf_self, True)] + [
            (s / max(tp, 1e-6), False)
            for s, tp in zip(sinr_peers_lin, tp_peers)
        ]
        # Descending: highest PF ratio → round 1
        all_pf.sort(key=lambda x: -x[0])
        rank = next(i + 1 for i, (_, is_self) in enumerate(all_pf) if is_self)
        return min(math.ceil(rank / max_sim), n_rounds)

    # ── Rank-based SINR methods ──────────────────────────────────────────────
    all_sinrs = [(sinr_self_lin, True)] + [(s, False) for s in sinr_peers_lin]

    if method == 'max_sinr':
        # Descending: highest SINR → rank 1 → earliest round
        all_sinrs.sort(key=lambda x: -x[0])
    elif method == 'deadline_aware':
        # Ascending: lowest SINR → rank 1 → earliest round (most protection)
        all_sinrs.sort(key=lambda x: x[0])
    else:
        return n_rounds   # unknown method: conservative fallback

    rank = next(i + 1 for i, (_, is_self) in enumerate(all_sinrs) if is_self)
    return min(math.ceil(rank / max_sim), n_rounds)


def get_n_rounds(n_cell: int, ru_type: int) -> int:
    """
    Number of TF rounds needed to serve all n_cell robots in one AP.

    802.11ax Trigger Frame (TF) schedules at most MAX_SIMULTANEOUS[ru_type]
    STAs simultaneously.  If n_cell exceeds that, the AP issues additional
    TF rounds within the same polling interval.

    Examples (ru_type based on capped n_cell):
        n_cell=1, ru=0 (242-tone) → 1 round
        n_cell=2, ru=0 (242-tone) → 2 rounds  (1 user/round)
        n_cell=4, ru=1 (106-tone) → 2 rounds  (2 users/round)
        n_cell=8, ru=2 ( 52-tone) → 2 rounds  (4 users/round)
        n_cell=9, ru=3 ( 26-tone) → 1 round   (9 users fit)
        n_cell=16,ru=3 ( 26-tone) → 2 rounds
    """
    max_sim = MAX_SIMULTANEOUS.get(ru_type, 1)
    return max(1, math.ceil(n_cell / max_sim))


def sinr_to_throughput(sinr_db: float, n_cell: int) -> float:
    """Convenience: SINR (dB) + cell load → throughput (Mbps)."""
    mcs = select_mcs(sinr_db)
    ru  = get_ru_type(n_cell)
    return compute_throughput(mcs, ru)
