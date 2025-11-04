import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import thop

# -----------------------------------------------------------------------------
# Utility convolution block
# -----------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -----------------------------------------------------------------------------
# Adaptive Frequency-based Gating (AFG) Module
# -----------------------------------------------------------------------------
class FrequencyBasis(nn.Module):
    """Learnable frequency bases (Φ) for adaptive frequency modulation."""
    def __init__(self, n_bases, ch):
        super().__init__()
        self.n_bases = n_bases
        self.bases = nn.Parameter(torch.randn(n_bases, ch))
        nn.init.xavier_uniform_(self.bases)

    def forward(self, coef):
        # coef: (B, n_bases)
        # combine coefficients with bases -> (B, ch)
        return torch.matmul(coef, self.bases)

class AdaptiveFeatureGating(nn.Module):
    """Implements the AFG module in MTPI-Net.
    Adaptive frequency modulation guided by channel descriptors.
    """
    def __init__(self, ch, n_bases=8, reduction=2):
        super().__init__()
        self.ch = ch
        self.n_bases = n_bases
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, n_bases)
        )
        self.freq_basis = FrequencyBasis(n_bases, ch)
        self.reproj = nn.Conv2d(ch, ch, 1)
        self.norm = nn.BatchNorm2d(ch)

    def forward(self, x):
        B, C, H, W = x.shape
        desc = self.pool(x).view(B, C)
        coef = F.softmax(self.mlp(desc), dim=-1)  # (B, n_bases)
        DF = self.freq_basis(coef).view(B, C, 1, 1)  # (B, C, 1, 1)

        # Transform to frequency domain
        Xf = torch.fft.fft2(x, norm='ortho')
        # Apply modulation across channels
        gated = torch.fft.ifft2(Xf * (1 + DF), norm='ortho').real
        out = self.reproj(gated)
        return self.norm(out + x)

# -----------------------------------------------------------------------------
# Temporal cross-frame variant (for multi-temporal fusion)
# -----------------------------------------------------------------------------
class AFGTemporal(nn.Module):
    """Cross-frame adaptive feature gating for adjacent-frame guidance."""
    def __init__(self, ch, n_bases=8):
        super().__init__()
        self.afg_tar = AdaptiveFeatureGating(ch, n_bases)
        self.afg_adj = AdaptiveFeatureGating(ch, n_bases)
        self.merge = Conv(ch * 2, ch, 1)

    def forward(self, F_tar, F_adj):
        O_tar = self.afg_tar(F_tar)
        O_adj = self.afg_adj(F_adj)
        out = self.merge(torch.cat([O_tar, O_adj], dim=1))
        return out

# -----------------------------------------------------------------------------
# Unit test / complexity test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    ch, H, W = 256, 32, 32
    model = AdaptiveFeatureGating(ch)
    x = torch.randn(1, ch, H, W)
    y = model(x)
    print('Output:', y.shape)

    flops, params = thop.profile(model, inputs=(x,), verbose=False)
    print(f"FLOPs: {flops/1e9:.3f}G, Params: {params/1e6:.3f}M")
