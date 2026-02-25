"""
Human agent simulation for the CA-MAPF environment.

Each HumanAgent walks a random path on the valid factory grid.
Robots can detect humans within their 120° field-of-view (FoV).
Human presence causes RF shadowing when they lie between an AP and a robot.

Coordinate convention (Sionna):  X = right, Y = forward, Z = up.
Grid indices directly equal world X/Y coordinates (1 m resolution).
"""
import numpy as np
from scipy.spatial import KDTree
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    HUMAN_SPEED, HUMAN_DETECTION_RANGE, ROBOT_FOV_DEG,
    HUMAN_SHADOW_DB, HUMAN_SHADOW_RADIUS, GRID_RESOLUTION,
    AP_POSITIONS, ROBOT_HEIGHT, HUMAN_HEIGHT,
)


# ── Utility ───────────────────────────────────────────────────────────────────

def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between 2D vectors v1 and v2."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _dist2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a)[:2] - np.asarray(b)[:2]))


# ── HumanAgent class ──────────────────────────────────────────────────────────

class HumanAgent:
    """
    A pedestrian that walks a random path on the valid factory grid.

    State:
        pos_idx  : int   — current grid index in the global grid_positions array
        heading  : (2,)  — current facing direction (unit 2D vector)
    """

    def __init__(self, grid_positions: np.ndarray, neighbors: dict,
                 start_idx: int = None, rng: np.random.Generator = None):
        self.grid_positions = grid_positions   # (N, 3) float32
        self.neighbors      = neighbors        # {idx: [nbr_idx, ...]}
        self.rng            = rng or np.random.default_rng()

        N = len(grid_positions)
        self.pos_idx = int(self.rng.integers(0, N)) if start_idx is None else start_idx
        self.heading = np.array([1.0, 0.0], dtype=np.float32)  # initial heading

        # Random waypoint the human is walking towards
        self._waypoint_idx = self._random_waypoint()
        self._steps_at_waypoint = 0

    # ── Movement ──────────────────────────────────────────────────────────────

    def _random_waypoint(self) -> int:
        N = len(self.grid_positions)
        for _ in range(20):
            w = int(self.rng.integers(0, N))
            if w != self.pos_idx:
                return w
        return self.pos_idx

    def step(self):
        """Move one step towards the current waypoint (greedy grid walk)."""
        if self.pos_idx == self._waypoint_idx or self._steps_at_waypoint > 30:
            self._waypoint_idx    = self._random_waypoint()
            self._steps_at_waypoint = 0

        nbrs = self.neighbors.get(self.pos_idx, [])
        if not nbrs:
            return

        # Move towards waypoint: pick neighbor that minimises distance
        # Use nav plane (world_x, world_y) = columns 0 and 2
        goal_pos = self.grid_positions[self._waypoint_idx][[0, 2]]
        best_nbr = None
        best_dist = np.inf
        for n in nbrs:
            if n == self.pos_idx:
                continue
            d = np.linalg.norm(self.grid_positions[n][[0, 2]] - goal_pos)
            if d < best_dist:
                best_dist = d
                best_nbr  = n

        # Small probability of random step (avoid getting stuck)
        if self.rng.random() < 0.15 or best_nbr is None:
            candidates = [n for n in nbrs if n != self.pos_idx]
            if candidates:
                best_nbr = int(self.rng.choice(candidates))

        if best_nbr is not None and best_nbr != self.pos_idx:
            new_pos = self.grid_positions[best_nbr][[0, 2]]
            cur_pos = self.grid_positions[self.pos_idx][[0, 2]]
            delta   = new_pos - cur_pos
            if np.linalg.norm(delta) > 1e-6:
                self.heading = delta / np.linalg.norm(delta)
            self.pos_idx = best_nbr

        self._steps_at_waypoint += 1

    @property
    def world_pos(self) -> np.ndarray:
        """Sionna world position [world_x, world_z, world_y] for this human."""
        p = self.grid_positions[self.pos_idx].copy()
        p[1] = HUMAN_HEIGHT   # column 1 = world_z (height)
        return p

    @property
    def grid_xy(self) -> np.ndarray:
        """
        2D navigation-plane position (world_x, world_y).
        grid_positions columns: [world_x, world_z, world_y]
          → navigation plane uses columns 0 (world_x) and 2 (world_y).
        """
        p = self.grid_positions[self.pos_idx]
        return np.array([p[0], p[2]], dtype=np.float32)


# ── Multi-human manager ───────────────────────────────────────────────────────

class HumanManager:
    """
    Manages N pedestrians in the scene.
    Provides:
      - step()        : advance all humans one time step
      - visible_from(robot_pos, robot_heading) → list[HumanAgent]
      - sinr_penalty_db(robot_pos, ap_pos)     → dB shadowing loss
    """

    def __init__(self, n_humans: int, grid_positions: np.ndarray,
                 neighbors: dict, rng: np.random.Generator = None):
        self.rng = rng or np.random.default_rng()
        self.agents = [
            HumanAgent(grid_positions, neighbors, rng=self.rng)
            for _ in range(n_humans)
        ]

    def reset(self, rng: np.random.Generator = None):
        """Reseed and re-randomise all human positions."""
        if rng:
            self.rng = rng
        N = len(self.agents[0].grid_positions)
        for agent in self.agents:
            agent.rng      = self.rng
            agent.pos_idx  = int(self.rng.integers(0, N))
            agent.heading  = np.array([1.0, 0.0], dtype=np.float32)
            agent._waypoint_idx     = agent._random_waypoint()
            agent._steps_at_waypoint = 0

    def step(self):
        for agent in self.agents:
            agent.step()

    # ── Visibility (FoV) ──────────────────────────────────────────────────────

    def visible_from(self, robot_pos: np.ndarray,
                     robot_heading: np.ndarray) -> list:
        """
        Return list of HumanAgents visible from robot_pos within 120° FoV.

        robot_pos     : Sionna [world_x, world_z, world_y] or plain (x,y)
        robot_heading : 2D direction in (world_x, world_y) plane
        Detection cone: ±60° around robot_heading, ≤ HUMAN_DETECTION_RANGE m.
        """
        fov_half = ROBOT_FOV_DEG / 2.0
        rp_arr = np.asarray(robot_pos)
        # Support both 3-element [x,z,y] and 2-element [x,y] inputs
        if rp_arr.shape[0] >= 3:
            rp2 = np.array([rp_arr[0], rp_arr[2]], dtype=np.float32)
        else:
            rp2 = rp_arr[:2].astype(np.float32)
        rh2 = np.asarray(robot_heading)[:2]

        visible = []
        for agent in self.agents:
            hp2  = agent.grid_xy          # (world_x, world_y)
            diff = hp2 - rp2
            dist = float(np.linalg.norm(diff))
            if dist > HUMAN_DETECTION_RANGE:
                continue
            if _angle_between(rh2, diff) <= fov_half:
                visible.append(agent)
        return visible

    # ── RF shadowing ──────────────────────────────────────────────────────────

    def sinr_penalty_db(self, robot_pos: np.ndarray,
                        ap_positions: list = None) -> float:
        """
        Compute total RF shadowing penalty (dB) from humans blocking
        the AP→robot line of sight.

        All positions projected into the (world_x, world_y) navigation plane:
          robot_pos columns: [world_x, world_z, world_y] → use [0] and [2]
          ap_positions columns: same format
        """
        if ap_positions is None:
            ap_positions = AP_POSITIONS

        # Extract (world_x, world_y) for robot
        rp  = np.array(robot_pos)
        rp2 = np.array([rp[0], rp[2]], dtype=np.float32)   # (x, y) plane

        penalty_db = 0.0
        for agent in self.agents:
            hp = agent.grid_xy    # already (world_x, world_y)
            for ap in ap_positions:
                ap_arr = np.asarray(ap)
                ap2 = np.array([ap_arr[0], ap_arr[2]], dtype=np.float32)
                seg     = rp2 - ap2
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-6:
                    continue
                t       = np.clip(np.dot(hp - ap2, seg) / seg_len**2, 0.0, 1.0)
                closest = ap2 + t * seg
                if np.linalg.norm(hp - closest) < HUMAN_SHADOW_RADIUS:
                    penalty_db += HUMAN_SHADOW_DB
                    break   # count each human at most once per robot

        return penalty_db

    # ── Observation vector for RL ─────────────────────────────────────────────

    def obs_vector(self, robot_pos: np.ndarray,
                   robot_heading: np.ndarray,
                   n_max_humans: int = None) -> np.ndarray:
        """
        Return a fixed-size observation vector of visible humans:
          [dist_1, angle_1, dist_2, angle_2, ..., dist_K, angle_K]
        Padded with zeros for unseen slots.

        n_max_humans defaults to the total number of agents.
        """
        if n_max_humans is None:
            n_max_humans = len(self.agents)
        vis = self.visible_from(robot_pos, robot_heading)
        rp_arr = np.asarray(robot_pos)
        rp2 = np.array([rp_arr[0], rp_arr[2]], dtype=np.float32)  # (x,y) plane
        rh2 = np.asarray(robot_heading)[:2]

        obs = np.zeros(n_max_humans * 2, dtype=np.float32)
        for i, agent in enumerate(vis[:n_max_humans]):
            diff = agent.grid_xy - rp2
            dist = float(np.linalg.norm(diff))
            ang  = _angle_between(rh2, diff) / 90.0   # normalise to [-1, 1]
            obs[2 * i]     = dist / HUMAN_DETECTION_RANGE   # normalised dist
            obs[2 * i + 1] = ang
        return obs

    @property
    def positions(self) -> list:
        """List of 3D world positions for all human agents."""
        return [a.world_pos for a in self.agents]
