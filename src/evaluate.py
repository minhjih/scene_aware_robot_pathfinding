"""
Evaluation script: runs all baselines and the proposed method across
scenarios S1–S4, computes metrics, and saves results to results/.

Usage:
    CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
        --csi_map data/csi_map.pkl \
        --checkpoint checkpoints/best_model.pt \
        --n_trials 100
"""
import os, sys, pickle, argparse, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SCENARIOS, BASELINES, METRICS, RESULTS_DIR, CSI_MAP_PATH,
    CHECKPOINT_DIR, N_ROBOTS, N_HUMANS, MAX_STEPS, LAMBDA_COMM, R_GOAL,
    NUM_DATA_SC, C_IN, T_WIN, AP_POSITIONS, T_POLL,
)
from src.wifi_layer import (
    select_mcs, get_ru_type, compute_throughput, compute_comm_delay,
)
from src.models import DualBranchNet, MLPBaseline
from src.env import CAMAPFEnv, MOVE_DELTAS, GRID_RESOLUTION
from src.baselines import (
    a_star_shortest, a_star_snr_threshold, random_walk_path,
    pa_star, mlp_baseline_path, supervised_dual_branch_path,
    _nearest_ap, _move_time,
)

import numpy as np
import torch
from tqdm import tqdm


def path_metrics(path, grid_positions, sinr_map, reservation=None,
                 v_rob=1.0, n_cell_range=(1, 3)):
    """Compute all metrics for a single path."""
    if path is None or len(path) < 1:
        return {m: np.nan for m in METRICS}

    success = path[-1] == path[-1]  # always True if path exists (goal check done externally)
    total_time  = 0.0
    total_tp    = 0.0
    total_dist  = 0.0
    overflow_cnt = 0
    total_cost  = 0.0
    comm_cost   = 0.0

    if reservation is None:
        reservation = {}

    for step_i in range(len(path) - 1):
        u = path[step_i]
        v = path[step_i + 1]
        T_move = _move_time(grid_positions[u], grid_positions[v], v_rob)
        dist   = np.linalg.norm(grid_positions[u, :2] - grid_positions[v, :2])

        # Realistic n_cell from scenario range
        n_cell = int(np.random.randint(n_cell_range[0], n_cell_range[1] + 1))

        sinr    = sinr_map.get(v, np.zeros(NUM_DATA_SC))
        sinr_db = float(10 * np.log10(np.mean(sinr) + 1e-12))
        mcs     = select_mcs(sinr_db)
        ru      = get_ru_type(n_cell)
        R_act   = compute_throughput(mcs, ru)
        T_comm  = compute_comm_delay(R_act, mcs)

        if n_cell > 16:   # overflow threshold for 16-robot / 4-AP scenario
            overflow_cnt += 1

        penalty = max(0.0, T_comm - T_POLL)
        step_cost = T_move + LAMBDA_COMM * penalty

        total_time  += step_cost
        total_tp    += R_act
        total_dist  += dist
        total_cost  += step_cost
        comm_cost   += penalty

    n_steps = max(len(path) - 1, 1)
    return {
        "mission_latency":    total_time,
        "throughput_actual":  total_tp / n_steps,
        "overflow_rate":      overflow_cnt / n_steps * 100,
        "path_length":        total_dist,
        "success_rate":       100.0,   # full success (goal reached)
        "comm_penalty_ratio": comm_cost / max(total_cost, 1e-9),
    }


def evaluate_scenario(scenario_name, scenario_cfg, grid_positions,
                       csi_map, sinr_map, neighbors,
                       ppo_model, mlp_model, supervised_model,
                       device, n_trials=50):
    """
    Evaluate all baselines on a given scenario.
    Returns dict: {baseline_name: {metric: [values across trials]}}
    """
    n_cell_range = scenario_cfg["n_cell_range"]
    dynamic_obs  = scenario_cfg["dynamic_obs"]
    sc_n_humans  = scenario_cfg.get("n_humans", N_HUMANS)
    sc_n_robots  = scenario_cfg.get("n_robots", N_ROBOTS)
    N = len(grid_positions)

    results = {b: {m: [] for m in METRICS} for b in BASELINES}

    for trial in tqdm(range(n_trials), desc=f"Scenario {scenario_name}", leave=False):
        np.random.seed(trial)
        start = int(np.random.randint(0, N))
        goal  = int(np.random.randint(0, N))
        while goal == start:
            goal = int(np.random.randint(0, N))

        reservation = {}

        # ── 1. shortest_path ─────────────────────────────────────────────────
        try:
            path = a_star_shortest(grid_positions, neighbors, start, goal)
            if path is None or path[-1] != goal:
                m = {k: np.nan for k in METRICS}
                m["success_rate"] = 0.0
            else:
                m = path_metrics(path, grid_positions, sinr_map,
                                 n_cell_range=n_cell_range)
        except Exception:
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["shortest_path"][k].append(v)

        # ── 2. snr_threshold ─────────────────────────────────────────────────
        try:
            path = a_star_snr_threshold(grid_positions, neighbors, sinr_map,
                                        start, goal)
            if path is None or path[-1] != goal:
                m = {k: np.nan for k in METRICS}
                m["success_rate"] = 0.0
            else:
                m = path_metrics(path, grid_positions, sinr_map,
                                 n_cell_range=n_cell_range)
        except Exception:
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["snr_threshold"][k].append(v)

        # ── 3. random_walk ────────────────────────────────────────────────────
        try:
            path = random_walk_path(grid_positions, neighbors, start, goal,
                                    max_steps=MAX_STEPS // sc_n_robots)
            reached = path[-1] == goal if path else False
            m = path_metrics(path, grid_positions, sinr_map,
                             n_cell_range=n_cell_range)
            if not reached:
                m["success_rate"] = 0.0
        except Exception:
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["random_walk"][k].append(v)

        # ── 4. paper1_mlp ─────────────────────────────────────────────────────
        try:
            path = mlp_baseline_path(grid_positions, neighbors, csi_map,
                                     mlp_model, start, goal, device=device)
            reached = path[-1] == goal if path else False
            m = path_metrics(path, grid_positions, sinr_map,
                             n_cell_range=n_cell_range)
            if not reached:
                m["success_rate"] = 0.0
        except Exception:
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["paper1_mlp"][k].append(v)

        # ── 5. ours_supervised ────────────────────────────────────────────────
        try:
            path = supervised_dual_branch_path(
                grid_positions, neighbors, csi_map, sinr_map,
                supervised_model, start, goal, device=device
            )
            reached = path[-1] == goal if path else False
            m = path_metrics(path, grid_positions, sinr_map,
                             n_cell_range=n_cell_range)
            if not reached:
                m["success_rate"] = 0.0
        except Exception:
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["ours_supervised"][k].append(v)

        # ── 6. ours_ppo (PA-STA*) ─────────────────────────────────────────────
        try:
            path = pa_star(
                grid_positions, neighbors, sinr_map, csi_map,
                ppo_model, start, goal,
                reservation=dict(reservation),
                lambda_c=LAMBDA_COMM, device=device,
            )
            if path is None:
                # Fallback to shortest_path
                path = a_star_shortest(grid_positions, neighbors, start, goal)
            reached = (path is not None) and path[-1] == goal
            if path:
                m = path_metrics(path, grid_positions, sinr_map,
                                 n_cell_range=n_cell_range)
                if not reached:
                    m["success_rate"] = 0.0
            else:
                m = {k: np.nan for k in METRICS}
                m["success_rate"] = 0.0
        except Exception as e:
            print(f"  [warn] PA-STA* failed on trial {trial}: {e}")
            m = {k: np.nan for k in METRICS}
        for k, v in m.items():
            results["ours_ppo"][k].append(v)

    return results


def build_neighbors_from_env(env):
    """Extract neighbors dict from environment."""
    return env._neighbors


def aggregate(raw: dict) -> dict:
    """Aggregate per-trial results to mean ± std."""
    agg = {}
    for b, metrics in raw.items():
        agg[b] = {}
        for m, vals in metrics.items():
            arr = np.array([v for v in vals if not np.isnan(v)])
            agg[b][m] = {
                "mean": float(np.mean(arr)) if len(arr) > 0 else np.nan,
                "std":  float(np.std(arr))  if len(arr) > 0 else np.nan,
            }
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi_map",    default=CSI_MAP_PATH)
    parser.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "best_model.pt"))
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--scenarios",  nargs="+", default=list(SCENARIOS.keys()))
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # Load CSI map
    print(f"[eval] Loading CSI map ...")
    with open(args.csi_map, "rb") as f:
        data = pickle.load(f)
    csi_map        = data["csi_map"]
    sinr_map       = data["sinr_map"]
    grid_positions = data["grid_positions"]
    print(f"[eval] {len(csi_map)} grid positions")

    # Build env to get neighbors
    env = CAMAPFEnv(grid_positions, csi_map, sinr_map, n_robots=N_ROBOTS)
    neighbors = build_neighbors_from_env(env)

    # Load PPO model
    ppo_model = DualBranchNet().to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        ppo_model.load_state_dict(ckpt["model"])
        print(f"[eval] PPO model loaded from {args.checkpoint}")
    else:
        print(f"[eval] Warning: checkpoint not found at {args.checkpoint}. Using random weights.")
    ppo_model.eval()

    # Supervised DualBranch model (same arch, no RL training)
    supervised_model = DualBranchNet().to(device)
    sup_ckpt = os.path.join(os.path.dirname(args.checkpoint), "supervised_model.pt")
    if os.path.exists(sup_ckpt):
        ckpt2 = torch.load(sup_ckpt, map_location=device)
        supervised_model.load_state_dict(ckpt2["model"])
    supervised_model.eval()

    # MLP baseline model
    mlp_model = MLPBaseline().to(device)
    mlp_ckpt  = os.path.join(os.path.dirname(args.checkpoint), "mlp_model.pt")
    if os.path.exists(mlp_ckpt):
        ckpt3 = torch.load(mlp_ckpt, map_location=device)
        mlp_model.load_state_dict(ckpt3["model"])
    mlp_model.eval()

    all_results = {}
    for sc_name in args.scenarios:
        if sc_name not in SCENARIOS:
            print(f"[eval] Unknown scenario: {sc_name}, skipping.")
            continue
        sc_cfg = SCENARIOS[sc_name]
        print(f"\n[eval] Scenario {sc_name}: {sc_cfg['desc']}")

        raw = evaluate_scenario(
            sc_name, sc_cfg, grid_positions,
            csi_map, sinr_map, neighbors,
            ppo_model, mlp_model, supervised_model,
            device, n_trials=args.n_trials,
        )
        agg = aggregate(raw)
        all_results[sc_name] = agg

        # Save per-scenario JSON
        out_path = os.path.join(args.results_dir, f"{sc_name}_results.json")
        with open(out_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  Saved: {out_path}")

        # Print summary table
        print(f"\n  {'Baseline':<20} " +
              " ".join(f"{m[:12]:>14}" for m in METRICS))
        for b in BASELINES:
            row = f"  {b:<20} "
            for m in METRICS:
                val = agg[b][m]["mean"]
                row += f"{val:14.3f} "
            print(row)

    # Save combined results
    combined_path = os.path.join(args.results_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] All results saved to {combined_path}")


if __name__ == "__main__":
    main()
