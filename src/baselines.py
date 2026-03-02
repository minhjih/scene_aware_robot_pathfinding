"""
Baseline path-planning algorithms for comparison with CA-MAPF.

Baselines implemented:
  1. shortest_path    — A* ignoring communication (distance only)
  2. snr_threshold    — A* avoiding zones with SINR < threshold
  3. random_walk      — uniformly random action selection
  4. paper1_mlp       — simple MLP on 1D CSI (supervised)
  5. ours_supervised  — DualBranchNet trained with supervised learning
  6. pa_star          — PA-STA* using trained f_theta (proposed)
"""
import heapq
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    AP_POSITIONS, NUM_DATA_SC, C_IN, T_WIN, LAMBDA_COMM, L_DATA, T_PENALTY,
    MCS_SINR_THRESHOLD, N_HUMANS, N_ROBOTS, MAX_THROUGHPUT_MBPS, T_POLL,
    P_TX, P_TX_MIN, P_RX_TARGET, NOISE_VAR, SCHEDULING_METHOD,
)
from src.wifi_layer import (
    select_mcs, get_ru_type, get_n_rounds,
    select_best_ru_block,
    compute_tx_power_control, assign_scheduling_round,
    compute_throughput, compute_comm_delay,
)
from src.models import DualBranchNet, MLPBaseline
from src.env import MOVE_DELTAS, GRID_RESOLUTION


def _nav_xy(pos: np.ndarray) -> np.ndarray:
    """Extract (world_x, world_y) from a [world_x, world_z, world_y] position."""
    return np.array([pos[0], pos[2]], dtype=np.float32)


def _nearest_ap(robot_pos: np.ndarray) -> int:
    rxy = _nav_xy(np.asarray(robot_pos))
    dists = [np.linalg.norm(rxy - _nav_xy(np.asarray(ap)))
             for ap in AP_POSITIONS]
    return int(np.argmin(dists))


def _move_time(pos_a: np.ndarray, pos_b: np.ndarray, v: float = 1.0) -> float:
    return float(np.linalg.norm(_nav_xy(pos_a) - _nav_xy(pos_b))) / v


def _heuristic(grid_positions, v_idx, goal_idx, v_rob=1.0):
    p_v    = _nav_xy(grid_positions[v_idx])
    p_goal = _nav_xy(grid_positions[goal_idx])
    return float(np.linalg.norm(p_v - p_goal)) / v_rob


def a_star_shortest(grid_positions, neighbors, start, goal):
    """
    Standard A* (distance only, no communication cost).
    Returns path as list of grid indices, or None.
    """
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    g_score  = {start: 0.0}
    came_from = {}

    while open_heap:
        f, u = heapq.heappop(open_heap)
        if u == goal:
            return _reconstruct(came_from, u)
        for v in neighbors.get(u, []):
            if v == u:
                continue
            g_new = g_score[u] + _move_time(grid_positions[u], grid_positions[v])
            if g_new < g_score.get(v, np.inf):
                g_score[v]    = g_new
                came_from[v]  = u
                h = _heuristic(grid_positions, v, goal)
                heapq.heappush(open_heap, (g_new + h, v))
    return None


def a_star_snr_threshold(grid_positions, neighbors, sinr_map, start, goal,
                          sinr_threshold_db: float = -1.0):
    """
    A* that avoids nodes where mean SINR < threshold.
    Uses per-AP SINR from nearest AP.
    """
    def node_ok(idx):
        sinr_dict = sinr_map.get(idx, {})
        ap_idx = _nearest_ap(grid_positions[idx])
        sinr = sinr_dict.get(ap_idx, np.zeros(NUM_DATA_SC))
        sinr_db = 10 * np.log10(np.mean(sinr) + 1e-12)
        return sinr_db >= sinr_threshold_db

    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    g_score   = {start: 0.0}
    came_from = {}

    while open_heap:
        f, u = heapq.heappop(open_heap)
        if u == goal:
            return _reconstruct(came_from, u)
        for v in neighbors.get(u, []):
            if v == u:
                continue
            if not node_ok(v):
                continue
            g_new = g_score[u] + _move_time(grid_positions[u], grid_positions[v])
            if g_new < g_score.get(v, np.inf):
                g_score[v]   = g_new
                came_from[v] = u
                h = _heuristic(grid_positions, v, goal)
                heapq.heappush(open_heap, (g_new + h, v))
    # Fallback: ignore SNR constraint if no path found
    return a_star_shortest(grid_positions, neighbors, start, goal)


def random_walk_path(grid_positions, neighbors, start, goal, max_steps=500):
    """
    Random walk until goal reached or max_steps exceeded.
    Returns the sequence of visited nodes.
    """
    path = [start]
    cur  = start
    for _ in range(max_steps):
        if cur == goal:
            break
        nbrs = [n for n in neighbors.get(cur, []) if n != cur]
        if not nbrs:
            break
        cur = int(np.random.choice(nbrs))
        path.append(cur)
    return path


def pa_star(grid_positions, neighbors, sinr_map, csi_map,
            model, start, goal, reservation=None,
            lambda_c=LAMBDA_COMM, v_rob=1.0, n_max=9, device=None,
            max_expansions=20000):
    """
    PA-STA*: Prediction-Aware Space-Time A* using trained f_theta.
    """
    if device is None:
        device = torch.device("cpu")
    if reservation is None:
        reservation = {}

    model.eval()
    # Zero human-obs tensor to satisfy DualBranchNet fusion dim when h is unused
    h_zero = torch.zeros(1, N_HUMANS * 2, dtype=torch.float32).to(device)

    open_heap = []
    heapq.heappush(open_heap, (0.0, start, 0))   # (f, node, time_step)
    g_score   = {(start, 0): 0.0}
    came_from = {}
    state_hist = {
        (start, 0): np.zeros((T_WIN, 3), dtype=np.float32)
    }

    expansions = 0
    while open_heap:
        f, u, t = heapq.heappop(open_heap)
        expansions += 1
        if expansions > max_expansions:
            break
        if u == goal:
            return _reconstruct_st(came_from, u, t)

        for v in neighbors.get(u, []):
            if v == u:
                continue
            T_move = _move_time(grid_positions[u], grid_positions[v], v_rob)
            t_arr  = t + max(1, int(T_move))

            # Network capacity pruning
            ap     = _nearest_ap(grid_positions[v])
            n_cell = len(reservation.get((ap, t_arr), set()))
            if n_cell > n_max:
                continue

            # f_theta inference
            C_v = torch.FloatTensor(
                csi_map.get(v, np.zeros((C_IN, NUM_DATA_SC, T_WIN)))
            ).unsqueeze(0).to(device)

            s_hist = state_hist.get((u, t), np.zeros((T_WIN, 3), np.float32))
            s_v    = torch.FloatTensor(s_hist).unsqueeze(0).to(device)

            with torch.no_grad():
                _, _, R_hat = model(C_v, s_v, h_zero)
            R_hat_val = max(R_hat.item(), 0.01)

            # Communication delay: multi-round TF + wideband PC + scheduling
            # PA-STAR has no real-time AP load or peer SINR info, so we use:
            #   n_cell_est  = mean load across 4 APs
            #   round_idx   = worst-case n_rounds (round_robin conservative)
            n_cell_est  = max(1, N_ROBOTS // len(AP_POSITIONS))
            ru_type_est = get_ru_type(n_cell_est)

            sinr_dict = sinr_map.get(v, {})
            sinr = sinr_dict.get(ap, np.zeros(NUM_DATA_SC, dtype=np.float32))
            # Wideband power control
            _, pc_scale = compute_tx_power_control(
                sinr, P_RX_TARGET, NOISE_VAR, P_TX, P_TX_MIN, P_TX,
            )
            sinr_pc = sinr * pc_scale
            # Best RU block after power control
            _, sinr_ru_db = select_best_ru_block(sinr_pc, ru_type_est)
            mcs = select_mcs(sinr_ru_db)
            # Round assignment: no peer info → round_robin worst-case
            round_idx = assign_scheduling_round(
                float(np.mean(sinr_pc)), [], n_cell_est, ru_type_est,
                SCHEDULING_METHOD,
            )
            T_comm_round = compute_comm_delay(R_hat_val, mcs)
            T_comm  = round_idx * T_comm_round

            # Edge weight — polling-deadline penalty (matches env.py semantics)
            w = T_move + lambda_c * max(0.0, T_comm - T_POLL)

            g_new = g_score.get((u, t), np.inf) + w
            if g_new < g_score.get((v, t_arr), np.inf):
                g_score[(v, t_arr)]   = g_new
                came_from[(v, t_arr)] = (u, t)
                h = _heuristic(grid_positions, v, goal, v_rob)
                heapq.heappush(open_heap, (g_new + h, v, t_arr))

                # Update state history for (v, t_arr) — normalized, matches env.py
                ru_v      = get_ru_type(n_cell)
                R_actual  = compute_throughput(mcs, ru_v)
                new_state = np.array([
                    mcs / 11.0,
                    n_cell / float(N_ROBOTS),
                    R_actual / MAX_THROUGHPUT_MBPS,
                ], dtype=np.float32)
                new_hist  = np.roll(s_hist, -1, axis=0)
                new_hist[-1] = new_state
                state_hist[(v, t_arr)] = new_hist

                # Update reservation
                reservation.setdefault((ap, t_arr), set())
                reservation[(ap, t_arr)].add(v)

    return None


def mlp_baseline_path(grid_positions, neighbors, csi_map, model,
                       start, goal, device=None):
    """
    Greedy path using MLP baseline (1D CSI → throughput prediction).
    At each step, move to the neighbor with highest predicted throughput.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()

    path = [start]
    cur  = start
    visited = {start}

    for _ in range(500):
        if cur == goal:
            break
        nbrs = [n for n in neighbors.get(cur, []) if n != cur]
        if not nbrs:
            break

        best_nbr, best_r = cur, -np.inf
        for v in nbrs:
            C_v = torch.FloatTensor(
                csi_map.get(v, np.zeros((C_IN, NUM_DATA_SC, T_WIN)))
            ).unsqueeze(0).to(device)
            x = MLPBaseline.csi_to_input(C_v)
            with torch.no_grad():
                r = model(x).item()
            if r > best_r:
                best_r, best_nbr = r, v

        # If best is already visited, fallback to random
        if best_nbr in visited:
            best_nbr = int(np.random.choice(nbrs))

        visited.add(best_nbr)
        path.append(best_nbr)
        cur = best_nbr

    return path


def supervised_dual_branch_path(grid_positions, neighbors, csi_map, sinr_map,
                                 model, start, goal, device=None):
    """
    Greedy path using supervised DualBranchNet (no RL, direct R_hat maximization).
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    # Zero human-obs tensor to satisfy DualBranchNet fusion dim when h is unused
    h_zero = torch.zeros(1, N_HUMANS * 2, dtype=torch.float32).to(device)

    path = [start]
    cur  = start
    state_hist = np.zeros((T_WIN, 3), dtype=np.float32)
    visited = {start}

    for _ in range(500):
        if cur == goal:
            break
        nbrs = [n for n in neighbors.get(cur, []) if n != cur]
        if not nbrs:
            break

        best_nbr, best_r = cur, -np.inf
        s_t = torch.FloatTensor(state_hist).unsqueeze(0).to(device)

        for v in nbrs:
            C_v = torch.FloatTensor(
                csi_map.get(v, np.zeros((C_IN, NUM_DATA_SC, T_WIN)))
            ).unsqueeze(0).to(device)
            with torch.no_grad():
                _, _, r_hat = model(C_v, s_t, h_zero)
            r = r_hat.item()
            if r > best_r:
                best_r, best_nbr = r, v

        if best_nbr in visited:
            best_nbr = int(np.random.choice(nbrs))

        visited.add(best_nbr)
        path.append(best_nbr)
        cur = best_nbr

        # Update state history — normalized, matches env.py
        sinr_dict = sinr_map.get(cur, {})
        ap_cur    = _nearest_ap(grid_positions[cur])
        sinr      = sinr_dict.get(ap_cur, np.zeros(NUM_DATA_SC, dtype=np.float32))
        sinr_db   = float(10 * np.log10(np.mean(sinr) + 1e-12))
        mcs      = select_mcs(sinr_db)
        n_cell   = 1
        ru       = get_ru_type(n_cell)
        R_actual = compute_throughput(mcs, ru)
        new_st   = np.array([
            mcs / 11.0,
            n_cell / float(N_ROBOTS),
            R_actual / MAX_THROUGHPUT_MBPS,
        ], dtype=np.float32)
        state_hist = np.roll(state_hist, -1, axis=0)
        state_hist[-1] = new_st

    return path


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _reconstruct_st(came_from, u, t):
    path = [u]
    key  = (u, t)
    while key in came_from:
        key = came_from[key]
        path.append(key[0])
    path.reverse()
    return path
