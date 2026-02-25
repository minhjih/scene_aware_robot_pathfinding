"""
CA-MAPF Gymnasium environment with dynamic human agents and robot FoV.

Coordinate convention (Sionna):  X = right, Y = forward, Z = up.
Grid index [xi, yi] → world pos (xi, yi, ROBOT_HEIGHT).

Observation (per robot):
    csi      : (C_in, K, T)           MIMO channel magnitude
    state    : (T_win, 3)             [mcs/11, n_cell/N_R, R_actual/R_max] history
    human_obs: (N_humans * 2,)        visible human [dist, angle] pairs

Action: Discrete(9) — 8 cardinal/diagonal directions + wait (index 8)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import KDTree
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    C_IN, NUM_DATA_SC, T_WIN, AP_POSITIONS,
    GRID_RESOLUTION, ROBOT_HEIGHT,
    N_ROBOTS, MAX_STEPS, LAMBDA_COMM, R_GOAL,
    N_HUMANS, HUMAN_DETECTION_RANGE, ROBOT_FOV_DEG,
    HUMAN_SHADOW_DB, HUMAN_SHADOW_RADIUS,
    T_POLL, MAX_THROUGHPUT_MBPS, MAX_AGENTS_PER_AP,
)
from src.wifi_layer import (
    select_mcs, get_ru_type, compute_throughput, compute_comm_delay,
)
from src.human_agent import HumanManager

# 8 movement directions (dx, dy) in world coords + wait
MOVE_DELTAS = [
    ( 1,  0), ( 1,  1), ( 0,  1), (-1,  1),
    (-1,  0), (-1, -1), ( 0, -1), ( 1, -1),
    ( 0,  0),   # wait
]


class CAMAPFEnv(gym.Env):
    """
    Multi-agent CA-MAPF environment with:
    - 802.11ax communication-aware rewards
    - Dynamic human pedestrians (random-path walkers)
    - Robot 120° FoV human detection
    - RF shadowing penalty from human-body blockage
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        grid_positions: np.ndarray,    # (N, 3) float32, world coords
        csi_map: dict,                  # {idx: (C_in, K, T)}
        sinr_map: dict,                 # {idx: (K,)}
        n_robots: int = N_ROBOTS,
        max_steps: int = MAX_STEPS,
        lambda_comm: float = LAMBDA_COMM,
        r_goal: float = R_GOAL,
        n_humans: int = N_HUMANS,
        n_cell_override: int = None,    # fixed cell load for scenario control
        v_robot: float = 1.0,           # robot speed m/s
        dynamic_obs: bool = True,       # enable dynamic human agents
    ):
        super().__init__()
        self.grid_positions  = grid_positions    # (N, 3)
        self.csi_map         = csi_map
        self.sinr_map        = sinr_map
        self.n_robots        = n_robots
        self.max_steps       = max_steps
        self.lambda_comm     = lambda_comm
        self.r_goal          = r_goal
        self.n_humans        = n_humans
        self.n_cell_override = n_cell_override
        self.v_robot         = v_robot
        self.dynamic_obs     = dynamic_obs
        self.N               = len(grid_positions)

        # grid_positions columns: [world_x, world_z, world_y]
        # Navigation plane = (world_x, world_y) = columns 0 and 2
        self._nav_xy = grid_positions[:, [0, 2]]   # (N, 2) for KDTree
        self._tree   = KDTree(self._nav_xy)
        self._neighbors = self._build_adjacency()

        # Nearest-AP lookup uses same (world_x, world_y) plane
        ap_arr = np.array(AP_POSITIONS)[:, [0, 2]]  # col 0=x, col 2=y
        self._ap_tree = KDTree(ap_arr)

        # Human manager (handles N pedestrians)
        self.human_manager = HumanManager(
            n_humans=n_humans,
            grid_positions=grid_positions,
            neighbors=self._neighbors,
        ) if n_humans > 0 else None

        # ── Observation / action spaces ────────────────────────────────────
        human_obs_dim = n_humans * 2   # [dist, angle] per human
        self.observation_space = spaces.Dict({
            "csi":       spaces.Box(0, np.inf, (C_IN, NUM_DATA_SC, T_WIN), np.float32),
            "state":     spaces.Box(-np.inf, np.inf, (T_WIN, 3), np.float32),
            "human_obs": spaces.Box(0, 1, (human_obs_dim,), np.float32),
            "p_t":       spaces.Box(-1.0, 1.0, (4,), np.float32),
            "ap_obs":    spaces.Box(-1.0, 1.0, (MAX_AGENTS_PER_AP * 2,), np.float32),
        })
        self.action_space = spaces.Discrete(9)

        # State variables (initialised in reset)
        self.positions         = None
        self.goals             = None
        self.headings          = None   # (n_robots, 2) robot facing directions
        self.reservation       = None
        self.state_history     = None
        self.step_count        = 0
        self._current_robot_id = 0

    # ── Adjacency ──────────────────────────────────────────────────────────────
    def _build_adjacency(self) -> dict:
        """
        Build 8-direction adjacency in world (x, y) space.

        Grid index offset (dxi, dyi) maps to world delta:
            world_dx = dxi * 2          (x = xi*2+1)
            world_dy = dyi * (-2)       (y = -1.5 + yi*(-2))

        Neighbour lookup: find the closest valid grid node to the target
        world position; accept if within 1.5 m (< one grid step).
        """
        neighbors = {}
        THRESH = 1.5   # metres; one grid step = 2 m, diagonal = 2√2 ≈ 2.83 m
        for idx in range(self.N):
            pos_xz = self._nav_xy[idx]   # (world_x, world_y)
            nbrs = []
            for dxi, dyi in MOVE_DELTAS[:-1]:   # skip wait
                # world delta for this grid-index step
                target = pos_xz + np.array([dxi * 2.0, dyi * (-2.0)],
                                           dtype=np.float32)
                dist, nidx = self._tree.query(target, k=1)
                if dist < THRESH and int(nidx) != idx:
                    nbrs.append(int(nidx))
                else:
                    nbrs.append(idx)   # blocked or out-of-bounds → stay
            neighbors[idx] = nbrs
        return neighbors

    def _nearest_ap(self, pos_idx: int) -> int:
        # Use (world_x, world_y) plane
        robot_xz = self._nav_xy[pos_idx]
        _, ap_idx = self._ap_tree.query(robot_xz)
        return int(ap_idx)

    # ── Reset ──────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        MIN_DIST = 5   # minimum Manhattan distance in grid cells (= 10 m at 2 m/cell)
        while True:
            idxs   = rng.choice(self.N, self.n_robots * 2, replace=False)
            starts = idxs[:self.n_robots]
            goals  = idxs[self.n_robots:]
            dists  = [
                (abs(self._nav_xy[s, 0] - self._nav_xy[g, 0]) +
                 abs(self._nav_xy[s, 1] - self._nav_xy[g, 1])) / 2.0
                for s, g in zip(starts, goals)
            ]
            if all(d >= MIN_DIST for d in dists):
                break
        self.positions = list(starts)
        self.goals     = list(goals)
        # Initial headings: all robots face +Y (forward)
        self.headings  = [np.array([0.0, 1.0], dtype=np.float32)
                          for _ in range(self.n_robots)]
        self.reservation   = {}
        self.state_history = {
            i: np.zeros((T_WIN, 3), dtype=np.float32)
            for i in range(self.n_robots)
        }
        self.step_count        = 0
        self._current_robot_id = 0

        if self.human_manager:
            self.human_manager.reset(rng=rng)

        obs = self._get_obs(0)
        return obs, {}

    # ── Step ───────────────────────────────────────────────────────────────────
    def step(self, action: int):
        robot_id = self._current_robot_id
        cur_pos  = self.positions[robot_id]

        # Apply action
        if action == 8:
            next_pos = cur_pos
        else:
            next_pos = self._neighbors[cur_pos][action]

        # Update heading in (world_x, world_y) plane
        if next_pos != cur_pos:
            dxi, dyi = MOVE_DELTAS[action]
            # world deltas:  dx_world = dxi*2,  dy_world = dyi*(-2)
            d = np.array([dxi * 2.0, dyi * (-2.0)], dtype=np.float32)
            norm = np.linalg.norm(d)
            if norm > 0:
                self.headings[robot_id] = d / norm

        T_move = self._move_time(cur_pos, next_pos)

        # ── 802.11ax channel + comm ─────────────────────────────────────────
        if self.n_cell_override is not None:
            n_cell = self.n_cell_override
        else:
            n_cell = self._get_n_cell(next_pos)

        sinr_arr = self.sinr_map.get(next_pos, np.zeros(NUM_DATA_SC))

        # Apply RF shadowing from nearby humans
        if self.human_manager:
            robot_world = self.grid_positions[next_pos]
            shadow_db   = self.human_manager.sinr_penalty_db(robot_world)
            shadow_lin  = 10 ** (-shadow_db / 10.0)
            sinr_arr    = sinr_arr * shadow_lin

        sinr_db  = float(10 * np.log10(np.mean(sinr_arr) + 1e-12))
        mcs      = select_mcs(sinr_db)
        ru_type  = get_ru_type(n_cell)
        R_actual = compute_throughput(mcs, ru_type)
        T_comm   = compute_comm_delay(R_actual, mcs)

        # ── Reward ─────────────────────────────────────────────────────────
        # Polling-deadline penalty: if T_comm exceeds the AP's polling interval
        # (T_POLL = 1/upload_rate), the robot misses its next scheduled UL slot.
        # T_move >> T_POLL is irrelevant here; what matters is the uplink deadline.
        comm_penalty = max(0.0, T_comm - T_POLL)
        reward = -T_move - self.lambda_comm * comm_penalty
        if next_pos == self.goals[robot_id]:
            reward += self.r_goal

        # Penalty for moving into a human-occupied cell
        if self.human_manager:
            robot_world = self.grid_positions[next_pos]
            for agent in self.human_manager.agents:
                if agent.pos_idx == next_pos:
                    reward -= 2.0   # collision penalty

        # ── Update state ────────────────────────────────────────────────────
        self.positions[robot_id] = next_pos
        self._update_state_history(robot_id, mcs, n_cell, R_actual)
        self._update_reservation(next_pos)

        # Advance all humans one step (dynamic obstacles)
        if self.human_manager and self.dynamic_obs:
            self.human_manager.step()

        self.step_count += 1
        self._current_robot_id = (robot_id + 1) % self.n_robots

        all_done  = all(p == g for p, g in zip(self.positions, self.goals))
        truncated = self.step_count >= self.max_steps
        info = {
            "mcs": mcs, "R_actual": R_actual,
            "T_comm": T_comm, "n_cell": n_cell,
            "T_move": T_move, "comm_penalty": comm_penalty,
        }
        return self._get_obs(self._current_robot_id), reward, all_done, truncated, info

    # ── Observation ────────────────────────────────────────────────────────────
    def _get_obs(self, robot_id: int) -> dict:
        pos_idx  = self.positions[robot_id]
        goal_idx = self.goals[robot_id]

        csi = self.csi_map.get(pos_idx,
              np.zeros((C_IN, NUM_DATA_SC, T_WIN), np.float32))

        human_obs = np.zeros(self.n_humans * 2, dtype=np.float32)
        if self.human_manager and self.dynamic_obs:
            robot_world   = self.grid_positions[pos_idx]
            robot_heading = self.headings[robot_id]
            human_obs = self.human_manager.obs_vector(
                robot_world, robot_heading, n_max_humans=self.n_humans
            )

        # p_t: (current_x, current_y, goal_x, goal_y) normalised by 80 m
        p_t = np.concatenate([
            self._nav_xy[pos_idx]  / 80.0,
            self._nav_xy[goal_idx] / 80.0,
        ]).astype(np.float32)   # (4,)

        return {
            "csi":       csi.astype(np.float32),
            "state":     self.state_history[robot_id].copy(),
            "human_obs": human_obs,
            "p_t":       p_t,
            "ap_obs":    self._get_ap_obs(robot_id),
        }

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _update_state_history(self, robot_id: int, mcs: int,
                               n_cell: int, R_actual: float):
        # Normalize all features to [0, 1] — consistent scale for LSTM input.
        # Third feature is actual throughput (Mbps), replacing the old ru_type
        # which was a deterministic step-function of n_cell (redundant).
        new_st = np.array([
            mcs / 11.0,                          # MCS index  → [0, 1]
            n_cell / float(N_ROBOTS),            # congestion ratio → [0, 1]
            R_actual / MAX_THROUGHPUT_MBPS,      # achieved throughput → [0, 1]
        ], dtype=np.float32)
        hist   = self.state_history[robot_id]
        self.state_history[robot_id] = np.roll(hist, -1, axis=0)
        self.state_history[robot_id][-1] = new_st

    def _get_n_cell(self, pos_idx: int) -> int:
        ap_idx   = self._nearest_ap(pos_idx)
        t_bucket = self.step_count // self.n_robots
        occupied = self.reservation.get((ap_idx, t_bucket), set())
        return max(1, len(occupied))

    def _update_reservation(self, pos_idx: int):
        ap_idx   = self._nearest_ap(pos_idx)
        t_bucket = self.step_count // self.n_robots
        self.reservation.setdefault((ap_idx, t_bucket), set())
        self.reservation[(ap_idx, t_bucket)].add(pos_idx)

    def _move_time(self, cur: int, nxt: int) -> float:
        if cur == nxt:
            return 0.1   # small cost for waiting
        # Euclidean distance in (world_x, world_y) navigation plane
        p1 = self._nav_xy[cur]   # (world_x, world_y)
        p2 = self._nav_xy[nxt]
        return float(np.linalg.norm(p1 - p2)) / self.v_robot

    def _get_ap_obs(self, robot_id: int) -> np.ndarray:
        """
        Same-AP peer observation: relative positions (in nav plane) of up to
        MAX_AGENTS_PER_AP other robots currently associated with the same AP.

        Output: (MAX_AGENTS_PER_AP * 2,) float32
            Each pair = (rel_x, rel_y) / 80.0  ∈ [-1, 1]
            Zero-padded when fewer peers are present.
        Sorted by Euclidean distance (nearest peers first).
        """
        my_pos = self.positions[robot_id]
        my_ap  = self._nearest_ap(my_pos)
        my_xy  = self._nav_xy[my_pos]

        peers = []
        for i in range(self.n_robots):
            if i == robot_id:
                continue
            if self._nearest_ap(self.positions[i]) == my_ap:
                rel   = (self._nav_xy[self.positions[i]] - my_xy) / 80.0
                dist2 = float(np.dot(rel, rel))
                peers.append((dist2, rel))

        peers.sort(key=lambda x: x[0])
        peers = peers[:MAX_AGENTS_PER_AP]

        out = np.zeros(MAX_AGENTS_PER_AP * 2, dtype=np.float32)
        for k, (_, rel) in enumerate(peers):
            out[k * 2]     = rel[0]
            out[k * 2 + 1] = rel[1]
        return out

    def get_visible_humans(self, robot_id: int):
        """Return visible human agents for a specific robot."""
        if not self.human_manager:
            return []
        pos_idx = self.positions[robot_id]
        return self.human_manager.visible_from(
            self.grid_positions[pos_idx],
            self.headings[robot_id]
        )

    def get_global_state(self) -> np.ndarray:
        """Return global state vector for centralized critic (MAPPO).

        Concatenates normalized (world_x, world_y) position and goal
        for every robot.  Shape: (n_robots * 4,) = (64,) for 16 robots.
        """
        segs = []
        for i in range(self.n_robots):
            p = self._nav_xy[self.positions[i]] / 80.0   # normalize world coords
            g = self._nav_xy[self.goals[i]]      / 80.0
            segs.append(p); segs.append(g)
        return np.concatenate(segs).astype(np.float32)   # (n_robots*4,)
