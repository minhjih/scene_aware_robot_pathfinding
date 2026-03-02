"""
PPO training loop for CA-MAPF (MAPPO + p_t + Option-A batch inference).

Key changes vs. original:
  - collect_rollout: batched forward pass per round (all N_ROBOTS at once)
  - p_t (position+goal) added to actor input
  - CentralizedCritic for value estimation (MAPPO CTDE)

Usage:
    CUDA_VISIBLE_DEVICES=0 python src/train_ppo.py
"""
import os, sys, pickle, argparse, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    LR, GAMMA, GAE_LAMBDA, CLIP_EPS, VALUE_COEF, ENTROPY_COEF,
    N_EPOCHS, BATCH_SIZE, ROLLOUT_LEN, MAX_EPISODES,
    N_ROBOTS, MAX_STEPS, LAMBDA_COMM, R_GOAL,
    N_HUMANS, CHECKPOINT_DIR, CSI_MAP_PATH,
    MAX_THROUGHPUT_MBPS,
)
from src.models import DualBranchNet, CentralizedCritic
from src.env import CAMAPFEnv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam


# ── GAE ───────────────────────────────────────────────────────────────────────

def compute_gae(rewards, values, dones, gamma, lam):
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    g   = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t < T - 1 else 0.0
        mask   = 1.0 - float(dones[t])
        delta  = rewards[t] + gamma * next_v * mask - values[t]
        g      = delta + gamma * lam * mask * g
        adv[t] = g
    return adv


# ── Option-A: batch-forward all robots at the start of each round ─────────────

def _batch_round(env, model, centralized_critic, device, n_humans, n_robots):
    """
    Collect observations for ALL n_robots, run a single batched forward pass,
    and return a dict  {robot_id: {obs, action, value, logp, gs}}.

    Called once per round (every n_robots env steps), replacing n_robots
    separate batch=1 forward passes with one batch=n_robots pass.
    """
    gs_np = env.get_global_state()                             # (n_robots*4,)
    gs_t  = torch.FloatTensor(gs_np).unsqueeze(0) \
                 .expand(n_robots, -1).to(device)              # (N, n_robots*4)

    all_obs = [env._get_obs(i) for i in range(n_robots)]

    from src.config import MAX_AGENTS_PER_AP
    C  = torch.FloatTensor(
            np.array([o["csi"]   for o in all_obs])).to(device)   # (N,4,234,16)
    s  = torch.FloatTensor(
            np.array([o["state"] for o in all_obs])).to(device)   # (N,16,3)
    p  = torch.FloatTensor(
            np.array([o["p_t"]   for o in all_obs])).to(device)   # (N,4)
    ap = torch.FloatTensor(
            np.array([o.get("ap_obs", np.zeros(MAX_AGENTS_PER_AP * 2))
                      for o in all_obs])).to(device)              # (N,16)
    h  = torch.FloatTensor(
            np.array([o.get("human_obs", np.zeros(n_humans * 2))
                      for o in all_obs])).to(device) if n_humans > 0 else None

    with torch.no_grad():
        logits_b, _, _, fused_b = model(C, s, h, p, ap, return_features=True)
        value_b  = centralized_critic(fused_b, gs_t)
        dist_b   = Categorical(logits=logits_b)
        actions  = dist_b.sample()
        logps    = dist_b.log_prob(actions)

    return {
        i: {
            "obs":    all_obs[i],
            "action": actions[i].item(),
            "value":  value_b[i].squeeze().item(),
            "logp":   logps[i].item(),
            "gs":     gs_np,
        }
        for i in range(n_robots)
    }


# ── Rollout collection ────────────────────────────────────────────────────────

def collect_rollout(env, model, centralized_critic, device,
                    rollout_len, n_humans, n_robots=N_ROBOTS):
    """
    Collect rollout_len transitions.

    Every time robot_id resets to 0 (start of a new round), a single
    batch=n_robots forward pass is executed for all robots simultaneously
    (Option A).  Per-step overhead = env.step() only.
    """
    csi_buf, state_buf, human_buf, pt_buf, ap_buf = [], [], [], [], []
    global_state_buf                               = []
    act_buf, rew_buf, val_buf                      = [], [], []
    logp_buf, done_buf, R_buf                      = [], [], []

    env.reset()
    cache = {}   # {robot_id: {...}} — rebuilt each round

    for _ in range(rollout_len):
        robot_id = env._current_robot_id

        # Rebuild batch cache at the start of each new round
        if robot_id == 0:
            cache = _batch_round(env, model, centralized_critic,
                                 device, n_humans, n_robots)

        c     = cache[robot_id]
        r_obs = c["obs"]

        # Store pre-computed observation and policy outputs
        from src.config import MAX_AGENTS_PER_AP
        csi_buf.append(r_obs["csi"])
        state_buf.append(r_obs["state"])
        human_buf.append(r_obs.get("human_obs", np.zeros(n_humans * 2)))
        pt_buf.append(r_obs["p_t"])
        ap_buf.append(r_obs.get("ap_obs", np.zeros(MAX_AGENTS_PER_AP * 2)))
        global_state_buf.append(c["gs"])
        act_buf.append(c["action"])
        val_buf.append(c["value"])
        logp_buf.append(c["logp"])

        _, reward, term, trunc, info = env.step(c["action"])
        done = term or trunc

        rew_buf.append(reward)
        done_buf.append(done)
        R_buf.append(info["R_actual"])

        if done:
            env.reset()
            cache = {}   # invalidate; robot_id will be 0 → rebuild next step

    return (
        np.array(csi_buf,          dtype=np.float32),  # (T,4,234,16)
        np.array(state_buf,        dtype=np.float32),  # (T,16,3)
        np.array(human_buf,        dtype=np.float32),  # (T,n_h*2)
        np.array(pt_buf,           dtype=np.float32),  # (T,4)
        np.array(ap_buf,           dtype=np.float32),  # (T,MAX_AP*2)
        np.array(global_state_buf, dtype=np.float32),  # (T,n_r*4)
        np.array(act_buf,          dtype=np.int64),    # (T,)
        np.array(rew_buf,          dtype=np.float32),  # (T,)
        np.array(val_buf,          dtype=np.float32),  # (T,)
        np.array(logp_buf,         dtype=np.float32),  # (T,)
        np.array(done_buf,         dtype=np.float32),  # (T,)
        np.array(R_buf,            dtype=np.float32),  # (T,)
    )


# ── PPO update ────────────────────────────────────────────────────────────────

def ppo_update(model, centralized_critic, optimizer, device, n_humans,
               csi_b, state_b, human_b, pt_b, ap_b, global_state_b, act_b,
               old_logp_b, adv_b, ret_b, R_b):
    C      = torch.FloatTensor(csi_b).to(device)
    s      = torch.FloatTensor(state_b).to(device)
    h_in   = torch.FloatTensor(human_b).to(device) if n_humans > 0 else None
    p_in   = torch.FloatTensor(pt_b).to(device)
    ap_in  = torch.FloatTensor(ap_b).to(device)
    gs     = torch.FloatTensor(global_state_b).to(device)
    a      = torch.LongTensor(act_b).to(device)
    lp_old = torch.FloatTensor(old_logp_b).to(device)
    adv    = torch.FloatTensor(adv_b).to(device)
    ret    = torch.FloatTensor(ret_b).to(device)
    R_ref  = torch.FloatTensor(R_b).to(device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    logits, _, R_hat, fused = model(C, s, h_in, p_in, ap_in, return_features=True)

    # NaN guard: model weights may have diverged → skip this mini-batch
    if not torch.isfinite(logits).all():
        return {"loss": float("nan"), "actor": 0.0, "critic": 0.0,
                "tp": 0.0, "entropy": 0.0}

    values   = centralized_critic(fused, gs)
    dist     = Categorical(logits=logits)
    new_logp = dist.log_prob(a)
    entropy  = dist.entropy().mean()

    # Clamp log-ratio BEFORE exp to prevent inf / NaN when policy drifts.
    # exp(±5) ≈ [0.007, 148] — well within float32 range.
    ratio  = (new_logp - lp_old).clamp(-5.0, 5.0).exp()
    surr1  = ratio * adv
    surr2  = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    actor_loss  = -torch.min(surr1, surr2).mean()

    # Normalise returns per mini-batch for critic stability.
    # Prevents MSE explosion when critic values start far from accumulated returns
    # (returns ≈ -30 to -100 at init; MSE would be 1k-10k without normalisation).
    ret_n       = (ret - ret.mean()) / (ret.std() + 1e-8)
    critic_loss = F.mse_loss(values.squeeze(), ret_n)

    # Normalise R_ref to [0,1] before Huber loss to prevent tp_loss explosions.
    R_hat_norm = F.relu(R_hat.squeeze()) / MAX_THROUGHPUT_MBPS
    R_ref_norm = R_ref / MAX_THROUGHPUT_MBPS
    tp_loss    = F.huber_loss(R_hat_norm, R_ref_norm, delta=0.1)

    loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy + 0.1 * tp_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(model.parameters()) + list(centralized_critic.parameters()), 0.3
    )
    optimizer.step()
    return {"loss": loss.item(), "actor": actor_loss.item(),
            "critic": critic_loss.item(), "tp": tp_loss.item(),
            "entropy": entropy.item()}


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"[train] Loading CSI map: {args.csi_map}")
    with open(args.csi_map, "rb") as f:
        data = pickle.load(f)
    csi_map        = data["csi_map"]
    sinr_map       = data["sinr_map"]
    grid_positions = data["grid_positions"]
    print(f"[train] {len(csi_map)} grid positions loaded")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    env = CAMAPFEnv(
        grid_positions, csi_map, sinr_map,
        n_robots=N_ROBOTS, max_steps=MAX_STEPS,
        n_humans=N_HUMANS, dynamic_obs=True,
        sched_policy=args.sched_policy,
        sched_noise_std_db=args.sched_noise,
    )
    print(f"[train] AP scheduling policy : {args.sched_policy}"
          f"  (noise={args.sched_noise} dB)")
    model              = DualBranchNet(n_humans=N_HUMANS).to(device)
    centralized_critic = CentralizedCritic(n_robots=N_ROBOTS).to(device)
    optimizer = Adam(
        list(model.parameters()) + list(centralized_critic.parameters()), lr=LR
    )

    # Resume — strict=False allows new pos_branch to be randomly initialised
    # when loading an older checkpoint that pre-dates this parameter.
    ckpt_file = os.path.join(args.save_dir, "latest.pt")
    start_ep, best_ret = 0, -np.inf
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[train] New model params (random init): {missing}")
        if "critic" in ckpt:
            centralized_critic.load_state_dict(ckpt["critic"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError, RuntimeError) as e:
            print(f"[train] Optimizer state incompatible ({e}), using fresh optimizer")
        start_ep = ckpt.get("episode", 0)
        best_ret = ckpt.get("best_ret", -np.inf)
        print(f"[train] Resumed from episode {start_ep}")

    log_path = os.path.join(args.save_dir, "train_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("episode,mean_return,loss,tp_loss,entropy,sched_policy\n")

    print(f"[train] Starting PPO for {args.max_episodes} episodes ...")
    t0 = time.time()

    for ep in range(start_ep, args.max_episodes):
        (csi_arr, state_arr, human_arr, pt_arr, ap_arr, gs_arr,
         act_arr, rew_arr, val_arr, logp_arr, done_arr, R_arr) = collect_rollout(
             env, model, centralized_critic, device,
             ROLLOUT_LEN, N_HUMANS, N_ROBOTS,
        )

        adv_arr  = compute_gae(rew_arr, val_arr, done_arr, GAMMA, GAE_LAMBDA)
        ret_arr  = adv_arr + val_arr
        mean_ret = float(rew_arr.sum())

        last_m = {}
        for _ in range(N_EPOCHS):
            idxs = np.random.permutation(ROLLOUT_LEN)
            for start in range(0, ROLLOUT_LEN, BATCH_SIZE):
                b = idxs[start:start + BATCH_SIZE]
                last_m = ppo_update(
                    model, centralized_critic, optimizer, device, N_HUMANS,
                    csi_arr[b], state_arr[b], human_arr[b],
                    pt_arr[b], ap_arr[b], gs_arr[b], act_arr[b],
                    logp_arr[b], adv_arr[b], ret_arr[b], R_arr[b],
                )

        if ep % 10 == 0:
            elapsed = time.time() - t0
            print(f"Ep {ep:5d} | ret={mean_ret:8.2f} | "
                  f"loss={last_m.get('loss',0):.4f} | "
                  f"tp={last_m.get('tp',0):.4f} | "
                  f"entropy={last_m.get('entropy',0):.3f} | "
                  f"sched={args.sched_policy} | t={elapsed:.0f}s")
            with open(log_path, "a") as f:
                f.write(f"{ep},{mean_ret:.4f},"
                        f"{last_m.get('loss',0):.6f},"
                        f"{last_m.get('tp',0):.6f},"
                        f"{last_m.get('entropy',0):.6f},"
                        f"{args.sched_policy}\n")

        # NaN guard: never persist a corrupt checkpoint.
        model_nan = any(not torch.isfinite(v).all()
                        for v in model.state_dict().values())
        if model_nan:
            print(f"[train] Ep {ep}: NaN detected in model — skipping checkpoint save")
            continue

        ckpt_data = {
            "model":     model.state_dict(),
            "critic":    centralized_critic.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode":   ep + 1,
            "best_ret":  best_ret,
        }
        torch.save(ckpt_data, ckpt_file)
        if mean_ret > best_ret:
            best_ret = mean_ret
            torch.save(ckpt_data, os.path.join(args.save_dir, "best_model.pt"))

    print(f"[train] Done. Best return: {best_ret:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi_map",      default=CSI_MAP_PATH)
    parser.add_argument("--max_episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--save_dir",     default=CHECKPOINT_DIR)
    # 802.11ax AP scheduling policy for this training run.
    # 'proportional_fair' : PF — de-facto enterprise Wi-Fi 6 standard.
    #                        RL learns optimal movement under fairness-driven AP.
    # 'auto'              : congestion-adaptive (max_sinr→PF→deadline_aware→rr).
    #                        RL learns to handle varying AP behaviour.
    # specific methods    : 'round_robin'|'max_sinr'|'deadline_aware' for ablations.
    parser.add_argument("--sched_policy", default="proportional_fair",
                        choices=["proportional_fair", "auto",
                                 "round_robin", "max_sinr", "deadline_aware"],
                        help="AP UL MU-OFDMA scheduling policy")
    parser.add_argument("--sched_noise",  type=float, default=1.5,
                        help="AP-side SINR estimation noise std (dB, 0=deterministic)")
    args = parser.parse_args()

    # Auto-suffix save_dir with scheduling policy to keep runs separate.
    if args.save_dir == CHECKPOINT_DIR:
        args.save_dir = os.path.join(CHECKPOINT_DIR, args.sched_policy)

    train(args)


if __name__ == "__main__":
    main()
