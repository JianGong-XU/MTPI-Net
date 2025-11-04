import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# 基础损失函数: Charbonnier
# ---------------------------------------------------------------------
class CharbonnierLoss(nn.Module):
    """
    Charbonnier (L1-smooth) loss used for numerical fidelity.
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, mask=None):
        diff = x - y
        if mask is not None:
            diff = diff * mask
        loss = torch.sqrt(diff.pow(2) + self.eps ** 2)
        return loss.mean()

# ---------------------------------------------------------------------
# Laplacian 边缘提取算子
# ---------------------------------------------------------------------
def laplacian_conv(x):
    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]], dtype=torch.float32, device=x.device
    ).view(1, 1, 3, 3)
    B, C, H, W = x.shape
    out = []
    for c in range(C):
        out.append(F.conv2d(x[:, c:c+1], kernel, padding=1))
    return torch.cat(out, dim=1)

# ---------------------------------------------------------------------
# 协同优化损失 (Collaborative Optimization Loss)
# ---------------------------------------------------------------------
class CollaborativeLoss(nn.Module):
    """
    Implements the joint optimization loss described in the paper:
    L_total = Σ_i [ (1/(2σ_i^2)) * L_i + log σ_i ]
    where:
        L_n: numerical fidelity term
        L_e: edge-aware Laplacian term
        L_t: temporal coherence term (optional)
    """
    def __init__(self):
        super().__init__()
        self.charb = CharbonnierLoss()
        # learnable log variances for uncertainty weighting
        self.log_sigma_n = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_e = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_t = nn.Parameter(torch.tensor(0.0))

    def forward(self, pred, ref, mask=None, temporal_pairs=None, weight_t=0.1):
        """
        Args:
            pred: predicted despeckled tensor (B,C,H,W)
            ref: noisy reference tensor (B,C,H,W)
            mask: change detection mask (B,1,H,W)
            temporal_pairs: optional list of tuples [(pred_t, pred_t+1), ...]
            weight_t: temporal loss scaling factor
        """
        # Numerical fidelity
        L_n = self.charb(pred, ref, mask)

        # Edge-aware Laplacian loss
        lap_pred = laplacian_conv(pred)
        lap_ref = laplacian_conv(ref)
        L_e = self.charb(lap_pred, lap_ref, mask)

        # Temporal coherence (if available)
        L_t = torch.tensor(0.0, device=pred.device)
        if temporal_pairs is not None:
            temporal_diffs = []
            for (f1, f2) in temporal_pairs:
                diff = torch.abs(f1 - f2)
                temporal_diffs.append(diff.mean())
            L_t = torch.stack(temporal_diffs).mean() * weight_t

        # Homoscedastic uncertainty weighting
        sigma_n = torch.exp(self.log_sigma_n)
        sigma_e = torch.exp(self.log_sigma_e)
        sigma_t = torch.exp(self.log_sigma_t)

        loss = (
            (0.5 / (sigma_n ** 2)) * L_n + torch.log(sigma_n + 1e-8)
            + (0.5 / (sigma_e ** 2)) * L_e + torch.log(sigma_e + 1e-8)
            + (0.5 / (sigma_t ** 2)) * L_t + torch.log(sigma_t + 1e-8)
        )
        return loss, {"L_n": L_n.item(), "L_e": L_e.item(), "L_t": L_t.item()}

# ---------------------------------------------------------------------
# Temporal Consistency Constraint (辅助函数)
# ---------------------------------------------------------------------
def compute_temporal_pairs(sequence_batch):
    """
    Utility to construct temporal pairs for multi-temporal sequences.
    Input: list or tensor of frames [B,T,C,H,W]
    Output: list of (frame_t, frame_t+1)
    """
    pairs = []
    if isinstance(sequence_batch, (list, tuple)):
        for i in range(len(sequence_batch) - 1):
            pairs.append((sequence_batch[i], sequence_batch[i + 1]))
    else:
        # assume tensor [B,T,C,H,W]
        T = sequence_batch.size(1)
        for i in range(T - 1):
            pairs.append((sequence_batch[:, i], sequence_batch[:, i + 1]))
    return pairs

# ---------------------------------------------------------------------
# 测试示例
# ---------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 7, 64, 64
    pred = torch.randn(B, C, H, W)
    ref = torch.randn(B, C, H, W)
    mask = torch.ones(B, 1, H, W)

    criterion = CollaborativeLoss()
    loss, parts = criterion(pred, ref, mask)
    print("Loss total:", loss.item())
    print("Loss parts:", parts)
