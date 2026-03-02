"""
Visualization of evaluation results.

Generates:
  - Bar charts comparing baselines per scenario and metric
  - Training curve from train_log.csv
  - Heat maps of throughput/SINR across the factory grid

Usage:
    python src/plot_results.py --results_dir results/
"""
import os, sys, json, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    SCENARIOS, BASELINES, METRICS, RESULTS_DIR, CHECKPOINT_DIR,
    CSI_MAP_PATH, NUM_DATA_SC, AP_POSITIONS,
)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd


# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "shortest_path":   "#4e79a7",
    "snr_threshold":   "#f28e2b",
    "random_walk":     "#e15759",
    "paper1_mlp":      "#76b7b2",
    "ours_supervised": "#59a14f",
    "ours_ppo":        "#b07aa1",
}

METRIC_LABELS = {
    "mission_latency":    "Mission Latency (s)",
    "throughput_actual":  "Throughput (Mbps)",
    "overflow_rate":      "Overflow Rate (%)",
    "path_length":        "Path Length (m)",
    "success_rate":       "Success Rate (%)",
    "comm_penalty_ratio": "Comm. Penalty Ratio",
}

BASELINE_LABELS = {
    "shortest_path":   "Shortest Path",
    "snr_threshold":   "SNR Threshold",
    "random_walk":     "Random Walk",
    "paper1_mlp":      "MLP (1D CSI)",
    "ours_supervised": "Dual-Branch (SL)",
    "ours_ppo":        "CA-MAPF (PPO)",
}


def _nearest_ap_idx(robot_pos) -> int:
    """Return index of nearest AP given a [world_x, world_z, world_y] position."""
    rxy = np.array([float(robot_pos[0]), float(robot_pos[2])], dtype=np.float32)
    dists = [np.linalg.norm(rxy - np.array([ap[0], ap[2]], dtype=np.float32))
             for ap in AP_POSITIONS]
    return int(np.argmin(dists))


def load_results(results_dir):
    combined = os.path.join(results_dir, "all_results.json")
    if os.path.exists(combined):
        with open(combined) as f:
            return json.load(f)
    # Fallback: load per-scenario files
    data = {}
    for sc in SCENARIOS:
        path = os.path.join(results_dir, f"{sc}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                data[sc] = json.load(f)
    return data


def plot_metric_bars(all_results, metric, results_dir):
    """Bar chart for one metric across all scenarios and baselines."""
    scenarios = [s for s in SCENARIOS if s in all_results]
    n_sc = len(scenarios)
    if n_sc == 0:
        return

    fig, axes = plt.subplots(1, n_sc, figsize=(4 * n_sc, 5), sharey=False)
    if n_sc == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        means = []
        stds  = []
        labels = []
        colors = []
        for b in BASELINES:
            if b not in all_results[sc]:
                continue
            val = all_results[sc][b].get(metric, {})
            means.append(val.get("mean", np.nan))
            stds.append(val.get("std", 0.0))
            labels.append(BASELINE_LABELS.get(b, b))
            colors.append(COLORS.get(b, "gray"))

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, color=colors, width=0.6,
                      capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{sc}: {SCENARIOS[sc]['desc']}", fontsize=9)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(METRIC_LABELS.get(metric, metric), fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(results_dir, f"bar_{metric}.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_all_metrics_summary(all_results, results_dir):
    """Combined heatmap: scenario × baseline for each metric."""
    scenarios = [s for s in SCENARIOS if s in all_results]
    if not scenarios:
        return

    for metric in METRICS:
        df_mean = pd.DataFrame(index=[BASELINE_LABELS[b] for b in BASELINES],
                               columns=scenarios)
        for sc in scenarios:
            for b in BASELINES:
                val = all_results.get(sc, {}).get(b, {}).get(metric, {})
                df_mean.loc[BASELINE_LABELS[b], sc] = val.get("mean", np.nan)

        df_mean = df_mean.astype(float)

        fig, ax = plt.subplots(figsize=(max(4, len(scenarios) * 1.5), 4))
        sns.heatmap(df_mean, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5)
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Method")
        plt.tight_layout()
        out = os.path.join(results_dir, f"heatmap_{metric}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_training_curve(results_dir, checkpoint_dir):
    log_path = os.path.join(checkpoint_dir, "train_log.csv")
    if not os.path.exists(log_path):
        print("  [skip] train_log.csv not found.")
        return

    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(df["episode"], df["mean_return"], color="#b07aa1", lw=1.5)
    axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Mean Return")
    axes[0].set_title("Training Return"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["episode"], df["loss"], color="#4e79a7", lw=1.5)
    axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Total Loss")
    axes[1].set_title("PPO Loss"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["episode"], df["tp_loss"], color="#f28e2b", lw=1.5,
                 label="TP Loss")
    axes[2].plot(df["episode"], df["entropy"], color="#59a14f", lw=1.5,
                 label="Entropy", linestyle="--")
    axes[2].set_xlabel("Episode"); axes[2].set_ylabel("Value")
    axes[2].set_title("Throughput Loss & Entropy")
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(results_dir, "training_curve.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_sinr_heatmap(results_dir):
    """Heatmap of mean SINR across the factory grid."""
    if not os.path.exists(CSI_MAP_PATH):
        print("  [skip] csi_map.pkl not found for SINR heatmap.")
        return

    with open(CSI_MAP_PATH, "rb") as f:
        data = pickle.load(f)
    sinr_map       = data["sinr_map"]
    grid_positions = data["grid_positions"]

    # grid_positions columns: [world_x, world_z, world_y]
    # Navigation plane: col 0 = world_x, col 2 = world_y
    xs = np.unique(grid_positions[:, 0])
    ys = np.unique(grid_positions[:, 2])
    grid = np.full((len(ys), len(xs)), np.nan)

    for idx, pos in enumerate(grid_positions):
        xi = np.searchsorted(xs, pos[0])
        yi = np.searchsorted(ys, pos[2])
        sinr_dict = sinr_map.get(idx, {})
        ap_idx = _nearest_ap_idx(pos)
        sinr = sinr_dict.get(ap_idx, np.zeros(NUM_DATA_SC, dtype=np.float32))
        sinr_db = float(10 * np.log10(np.mean(sinr) + 1e-12))
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            grid[yi, xi] = sinr_db

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean SINR (dB)")
    ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")
    ax.set_title("Factory SINR Map (Nearest AP, Mean over Subcarriers)")

    for i, ap in enumerate(AP_POSITIONS):
        # ap = [world_x, world_z, world_y]; plot in (x, y) plane
        ax.plot(ap[0], ap[2], "k^", markersize=10)
        ax.annotate(f"AP{i+1}", (ap[0], ap[2]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    plt.tight_layout()
    out = os.path.join(results_dir, "sinr_heatmap.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_throughput_heatmap(results_dir):
    """Heatmap of best-case throughput (N_cell=1) across the factory grid."""
    if not os.path.exists(CSI_MAP_PATH):
        return

    from src.wifi_layer import sinr_to_throughput

    with open(CSI_MAP_PATH, "rb") as f:
        data = pickle.load(f)
    sinr_map       = data["sinr_map"]
    grid_positions = data["grid_positions"]

    # grid_positions: [world_x, world_z, world_y]; nav plane = cols 0 and 2
    xs = np.unique(grid_positions[:, 0])
    ys = np.unique(grid_positions[:, 2])
    grid = np.full((len(ys), len(xs)), np.nan)

    for idx, pos in enumerate(grid_positions):
        xi = np.searchsorted(xs, pos[0])
        yi = np.searchsorted(ys, pos[2])
        sinr_dict = sinr_map.get(idx, {})
        ap_idx = _nearest_ap_idx(pos)
        sinr = sinr_dict.get(ap_idx, np.zeros(NUM_DATA_SC, dtype=np.float32))
        sinr_db = float(10 * np.log10(np.mean(sinr) + 1e-12))
        tp = sinr_to_throughput(sinr_db, n_cell=1)
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            grid[yi, xi] = tp

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Throughput (Mbps)")
    ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")
    ax.set_title("Factory Throughput Map (N_cell=1)")

    for i, ap in enumerate(AP_POSITIONS):
        # ap = [world_x, world_z, world_y]
        ax.plot(ap[0], ap[2], "w^", markersize=10)
        ax.annotate(f"AP{i+1}", (ap[0], ap[2]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="white")

    plt.tight_layout()
    out = os.path.join(results_dir, "throughput_heatmap.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",    default=RESULTS_DIR)
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("[plot] Loading results ...")
    all_results = load_results(args.results_dir)

    if not all_results:
        print("[plot] No results found. Run evaluate.py first.")
        return

    print("[plot] Generating bar charts ...")
    for metric in METRICS:
        plot_metric_bars(all_results, metric, args.results_dir)

    print("[plot] Generating heatmaps ...")
    plot_all_metrics_summary(all_results, args.results_dir)

    print("[plot] Generating training curve ...")
    plot_training_curve(args.results_dir, args.checkpoint_dir)

    print("[plot] Generating SINR heatmap ...")
    plot_sinr_heatmap(args.results_dir)

    print("[plot] Generating throughput heatmap ...")
    plot_throughput_heatmap(args.results_dir)

    print("\n[plot] Done. All figures saved to:", args.results_dir)


if __name__ == "__main__":
    main()
