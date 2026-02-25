"""
IEEE 802.11ax protocol layer:
  - MCS table and RU allocation
  - BLER curves
  - Throughput and communication delay computation
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MCS_TABLE, RU_DATA_SUBCARRIERS, MCS_SINR_THRESHOLD,
    T_SYM, L_DATA, T_PENALTY, T_PREAMBLE,
)


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

        1–2  robots  → 242-tone (full BW)   T_comm≈28ms,  no penalty
        3–4  robots  → 106-tone (half BW)   T_comm≈80ms,  no penalty
        5–8  robots  → 52-tone  (¼ BW)      T_comm≈227ms, +127ms penalty
        9–16 robots  → 26-tone  (⅛ BW)      T_comm≈567ms, +467ms penalty
        >16  robots  → Overflow (0 BW)       T_comm=∞
    """
    if n_cell <= 2:
        return 0    # 242-tone
    elif n_cell <= 4:
        return 1    # 106-tone
    elif n_cell <= 8:
        return 2    # 52-tone
    elif n_cell <= 16:
        return 3    # 26-tone
    else:
        return 4    # Overflow


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


def sinr_to_throughput(sinr_db: float, n_cell: int) -> float:
    """Convenience: SINR (dB) + cell load → throughput (Mbps)."""
    mcs = select_mcs(sinr_db)
    ru  = get_ru_type(n_cell)
    return compute_throughput(mcs, ru)
