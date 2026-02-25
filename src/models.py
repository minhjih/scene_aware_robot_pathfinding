"""
Neural network architectures for CA-MAPF.

  DualBranchNet : CNN (CSI) + LSTM (net-state) → learnable weighted-sum fusion
                  → actor / critic / throughput head

  Architecture follows "Adaptive AI Model Partitioning over 5G Networks"
  Nguyen et al., arXiv:2509.01906v1, 2025  (Fig. 3 + Table I):
    Branch 1 — LSTM  : temporal relation of KPM-equivalent state [MCS, N_cell, rho]
                        hidden_size=124 per paper, window=T_WIN
    Branch 2 — CNN   : spatial CSI features (analogous to IQ spectrogram)
                        2×(Conv2d → ReLU → MaxPool2d) + Flatten + Linear + Dropout
    Fusion   — weighted sum:  f = σ(α)·f_cnn + (1−σ(α))·f_lstm
                        α is a learnable scalar (replaces paper's explicit gNB ratio)
    Human branch added separately as project-specific context (not in paper).

  MLPBaseline : Simple MLP on 1D CSI (paper1_mlp baseline)
"""
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import C_IN, NUM_DATA_SC, T_WIN, N_HUMANS, N_ROBOTS, MAX_AGENTS_PER_AP


class CNNBranch(nn.Module):
    """
    Spatial CSI encoder — adapted from Table I of Nguyen et al. 2025.

    Paper Table I structure (IQ input):
        Conv2d → ReLU → MaxPool2d(2,2)
        Conv2d → ReLU → MaxPool2d(2,2)
        Flatten → Linear → ReLU → Dropout

    Our input: (batch, C_in=4, K=234, T=16)  — MIMO CSI magnitude tensor
      After 2× MaxPool2d(2,2): spatial dims K=234→117→58, T=16→8→4
      Flatten: 32 × 58 × 4 = 7424

    Output: (batch, d_out)
    """
    def __init__(self, c_in=C_IN, K=NUM_DATA_SC, T=T_WIN, d_out=256):
        super().__init__()
        K_out    = K // 4          # 234 // 4 = 58  (two MaxPool2d floor-div)
        T_out    = T // 4          # 16  // 4 = 4
        flat_dim = 32 * K_out * T_out   # 7424

        self.net = nn.Sequential(
            # Block 1 — matches first Conv2d→ReLU→MaxPool in Table I
            nn.Conv2d(c_in, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            # Block 2 — matches second Conv2d→ReLU→MaxPool in Table I
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, d_out),
            nn.ReLU(),
            nn.Dropout(0.1),        # Dropout as in Table I
        )

    def forward(self, x):
        return self.fc(self.net(x))


class LSTMBranch(nn.Module):
    """
    Temporal KPM encoder — Branch 1 of Nguyen et al. 2025 (Fig. 3).

    Paper: hidden_size=124, window=30, captures RSRP/RSRQ/SINR/CQI/RI/MCS/BLER...
    Here:  hidden_size=124 (matching paper), window=T_WIN=16,
           input = [MCS, N_cell, rho] per time step (KPM-equivalent state).

    Output: (batch, d_out)  — must equal CNNBranch d_out for weighted sum.
    """
    def __init__(self, input_dim=3, hidden_dim=124, num_layers=2, d_out=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc   = nn.Linear(hidden_dim, d_out)

    def forward(self, s):
        _, (h_n, _) = self.lstm(s)
        return self.fc(h_n[-1])


class HumanObsBranch(nn.Module):
    """
    Visible-human observation encoder (project-specific, not in Nguyen et al.).
    Input:  (batch, N_humans * 2)  — [dist_norm, angle_norm] per human
    Output: (batch, d_out=64)
    """
    def __init__(self, n_humans=N_HUMANS, d_out=64):
        super().__init__()
        in_dim = max(n_humans * 2, 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, d_out), nn.ReLU(),
        )

    def forward(self, h):
        return self.net(h)


class DualBranchNet(nn.Module):
    """
    f_theta: shared backbone for PPO actor/critic + auxiliary throughput head.

    Follows Nguyen et al. 2025 dual-branch + weighted-sum architecture (Fig. 3):
      f_cnn  = CNNBranch(C)                          # spatial CSI features
      f_lstm = LSTMBranch(s)                          # temporal KPM features
      w      = σ(α)                                   # learnable scalar ∈ (0,1)
      feat   = w·f_cnn + (1−w)·f_lstm                # weighted sum (paper Eq.)
      [+ human obs appended if available]
      fused  = FC(feat) → actor / critic / R_hat

    α replaces paper's explicit gNB resource-ratio weight; both are task-dependent
    scalars that balance spatial vs temporal channel representations.

    Inputs:
        C  : (batch, C_in, K, T)          MIMO CSI magnitude tensor
        s  : (batch, T, 3)                network-state sequence [MCS, N_cell, rho]
        h  : (batch, N_humans*2)          visible-human observation  [optional]
        p  : (batch, 4)                   [cur_x, cur_y, goal_x, goal_y]  [optional]
        ap : (batch, MAX_AGENTS_PER_AP*2) same-AP peer relative positions [optional]
             Each pair = (rel_x, rel_y)/80 for one peer; zero-padded if absent.
             Gives model visibility into near-future n_cell (who contends for the AP).

    Outputs:
        logits : (batch, action_dim)  actor logits
        value  : (batch, 1)           critic state value
        R_hat  : (batch, 1)           auxiliary throughput prediction (Mbps)
    """
    def __init__(self, c_in=C_IN, K=NUM_DATA_SC, T=T_WIN,
                 action_dim=9, d_shared=256, d_human=64,
                 n_humans=N_HUMANS, d_ap=32):
        super().__init__()
        self.cnn_branch  = CNNBranch(c_in=c_in, K=K, T=T, d_out=d_shared)
        self.lstm_branch = LSTMBranch(d_out=d_shared)
        self.use_human   = n_humans > 0
        if self.use_human:
            self.human_branch = HumanObsBranch(n_humans=n_humans, d_out=d_human)

        # Learnable fusion weight α — sigmoid-constrained to (0,1)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Position branch: (cur_x, cur_y, goal_x, goal_y) → 32-dim
        self.pos_branch = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(),
        )

        # Same-AP peer branch: relative positions of up to MAX_AGENTS_PER_AP
        # other robots in the same AP cell → d_ap-dim.
        # ap=None at inference → zero-padded (backward compatible with pa_star).
        self.d_ap = d_ap
        self.ap_branch = nn.Sequential(
            nn.Linear(MAX_AGENTS_PER_AP * 2, 64), nn.ReLU(),
            nn.Linear(64, d_ap),                  nn.ReLU(),
        )

        d_fused = d_shared + (d_human if self.use_human else 0) + 32 + d_ap
        self.fusion = nn.Sequential(
            nn.Linear(d_fused, 256), nn.ReLU(),
            nn.Linear(256, 128),    nn.ReLU(),
        )
        self.actor   = nn.Linear(128, action_dim)
        self.critic  = nn.Linear(128, 1)
        self.tp_head = nn.Linear(128, 1)

    def forward(self, C, s, h=None, p=None, ap=None, return_features=False):
        f_cnn  = self.cnn_branch(C)    # (batch, d_shared)
        f_lstm = self.lstm_branch(s)   # (batch, d_shared)

        # Weighted-sum fusion (Nguyen et al. Fig. 3)
        w    = torch.sigmoid(self.alpha)
        feat = w * f_cnn + (1.0 - w) * f_lstm     # (batch, d_shared)

        if self.use_human and h is not None:
            f_h  = self.human_branch(h)
            feat = torch.cat([feat, f_h], dim=-1)

        # Position/goal branch
        if p is not None:
            f_p = self.pos_branch(p)
        else:
            f_p = torch.zeros(feat.shape[0], 32, dtype=feat.dtype, device=feat.device)
        feat = torch.cat([feat, f_p], dim=-1)

        # Same-AP peer branch — zeros when ap is None (pa_star / baselines)
        if ap is not None:
            f_ap = self.ap_branch(ap)
        else:
            f_ap = torch.zeros(feat.shape[0], self.d_ap, dtype=feat.dtype, device=feat.device)
        feat = torch.cat([feat, f_ap], dim=-1)     # (batch, d_fused)

        fused  = self.fusion(feat)
        logits = self.actor(fused)
        value  = self.critic(fused)
        R_hat  = self.tp_head(fused)
        if return_features:
            return logits, value, R_hat, fused
        return logits, value, R_hat


class CentralizedCritic(nn.Module):
    """
    MAPPO centralized critic (CTDE — Centralized Training, Decentralized Execution).

    Combines a local observation feature vector (from DualBranchNet fusion layer)
    with the global state (all robots' positions and goals) to produce a
    global value estimate that removes non-stationarity from multi-agent learning.

    Inputs:
        f_local      : (batch, d_local=128)   fused features from DualBranchNet
        global_state : (batch, n_robots*4)    normalized [pos_x, pos_y, goal_x, goal_y]
                                               for every robot (64-dim for 16 robots)
    Output:
        value : (batch, 1)
    """
    def __init__(self, d_local=128, n_robots=N_ROBOTS):
        super().__init__()
        d_global = n_robots * 4   # 16*4 = 64
        self.global_enc = nn.Sequential(
            nn.Linear(d_global, 128), nn.ReLU(),
            nn.Linear(128, 64),       nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_local + 64, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, f_local, global_state):
        g_enc    = self.global_enc(global_state)              # (batch, 64)
        combined = torch.cat([f_local, g_enc], dim=-1)        # (batch, 192)
        return self.value_head(combined)                       # (batch, 1)


class MLPBaseline(nn.Module):
    """
    paper1_mlp baseline: MLP on mean CSI magnitude (1D, per subcarrier).
    Input:  (batch, K)
    Output: (batch, 1)   predicted throughput
    """
    def __init__(self, K=NUM_DATA_SC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def csi_to_input(C: torch.Tensor) -> torch.Tensor:
        """C: (batch, C_in, K, T) → mean over antennas and time → (batch, K)."""
        return C.mean(dim=[1, 3])
