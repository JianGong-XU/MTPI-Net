import torch
import torch.nn as nn
import torch.nn.functional as F
from module.SPP import SFS_Conv
from module.AFG import AdaptiveFeatureGating
from module.HFA import HFA_Module

# ---------------------------------------------------------------------
# Residual Block (用于结构细化)
# ---------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.relu(res + x)

# ---------------------------------------------------------------------
# Dual-Domain Collaborative Refinement (DDCR)
# ---------------------------------------------------------------------
class DDCR_Unit(nn.Module):
    """
    DDCR integrates SPP + AFG + HFA with residual learning.
    Args:
        ch (int): number of feature channels
        spp_filter (str): 'FrGT' or 'FrFT' for SPP
    """
    def __init__(self, ch, spp_filter='FrGT'):
        super().__init__()
        self.spp = SFS_Conv(ch, ch, filter=spp_filter)
        self.afg_tar = AdaptiveFeatureGating(ch)
        self.afg_adj = AdaptiveFeatureGating(ch)
        self.hfa = HFA_Module(ch)
        self.refine = ResidualBlock(ch)

    def forward(self, fused, F_tar, F_adj):
        # Step 1: spatial-frequency perception (SPP)
        O_spp = self.spp(fused)

        # Step 2: adaptive gating for target and adjacent frames
        O_tar = self.afg_tar(F_tar + O_spp)
        O_adj = self.afg_adj(F_adj + O_spp)

        # Step 3: holistic feature aggregation
        O_hfa = self.hfa(O_tar, O_adj)

        # Step 4: residual refinement
        out = self.refine(O_hfa + fused)
        return out

# ---------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ch = 128
    H = W = 64
    model = DDCR_Unit(ch, spp_filter='FrGT')
    fused = torch.randn(1, ch, H, W)
    F_tar = torch.randn(1, ch, H, W)
    F_adj = torch.randn(1, ch, H, W)

    out = model(fused, F_tar, F_adj)
    print("Output shape:", out.shape)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Params: {total_params/1e6:.3f} M")
