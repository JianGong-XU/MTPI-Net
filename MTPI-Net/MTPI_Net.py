import torch
import torch.nn as nn
import torch.nn.functional as F
from DDCR_Unit import DDCR_Unit

# ---------------------------------------------------------------------
# Feature Extraction and Normalization (FEN)
# ---------------------------------------------------------------------
class FEN(nn.Module):
    """
    Feature extraction for each temporal frame.
    Incorporates initial convolution, normalization, and activation.
    """
    def __init__(self, in_ch=7, out_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.head(x)

# ---------------------------------------------------------------------
# R²A Group (Recursive-Residual Attention Group)
# ---------------------------------------------------------------------
class R2A_Group(nn.Module):
    """
    Stack multiple DDCR units recursively with residual aggregation.
    """
    def __init__(self, ch, n_ddcr=3, spp_filter='FrGT'):
        super().__init__()
        self.blocks = nn.ModuleList([DDCR_Unit(ch, spp_filter) for _ in range(n_ddcr)])
        self.merge = nn.Conv2d(ch, ch, kernel_size=1)

    def forward(self, fused, F_tar, F_adj):
        out = fused
        for ddcr in self.blocks:
            out = ddcr(out, F_tar, F_adj) + out
        return self.merge(out)

# ---------------------------------------------------------------------
# MTPI-Net (Multitemporal Progressive Interaction Network)
# ---------------------------------------------------------------------
class MTPI_Net(nn.Module):
    """
    Complete architecture implementing:
    - Feature extraction (FEN)
    - Progressive multitemporal interaction (R2A Group)
    - Collaborative reconstruction (output head)
    """
    def __init__(self, in_ch=7, base_ch=64, n_ddcr=3, spp_filter='FrGT'):
        super().__init__()
        # Multi-temporal feature encoders
        self.fen_tar = FEN(in_ch, base_ch)
        self.fen_adj = FEN(in_ch, base_ch)
        self.fen_next = FEN(in_ch, base_ch)

        # Channel reduction after concatenation
        self.reduce = nn.Conv2d(base_ch * 3, base_ch, kernel_size=1)

        # Progressive interaction group
        self.r2a = R2A_Group(base_ch, n_ddcr, spp_filter)

        # Reconstruction head
        self.recon = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )

    def forward(self, I_tar, I_prev, I_next):
        # Step 1: feature extraction
        F_tar = self.fen_tar(I_tar)
        F_prev = self.fen_adj(I_prev)
        F_next = self.fen_next(I_next)

        # Step 2: feature fusion
        fused = torch.cat([F_tar, F_prev, F_next], dim=1)
        fused = self.reduce(fused)

        # Step 3: multitemporal progressive interaction
        enhanced = self.r2a(fused, F_tar, F_prev)

        # Step 4: reconstruction (residual learning)
        residual = self.recon(enhanced)
        despeckled = I_tar - residual
        return despeckled

# ---------------------------------------------------------------------
# Example Test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 1, 7, 128, 128
    model = MTPI_Net(in_ch=C, base_ch=64, n_ddcr=3, spp_filter='FrGT')
    I_tar = torch.randn(B, C, H, W)
    I_prev = torch.randn(B, C, H, W)
    I_next = torch.randn(B, C, H, W)

    out = model(I_tar, I_prev, I_next)
    print("Output shape:", out.shape)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {params / 1e6:.3f} M")

