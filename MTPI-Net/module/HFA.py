import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import thop

# ---------------------------------------------------------------------
# Utility Layers
# ---------------------------------------------------------------------
class Conv1x1(nn.Module):
    def __init__(self, in_ch, out_ch, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ---------------------------------------------------------------------
# Cross-Frequency Residual Self-Attention (CFR-SA)
# ---------------------------------------------------------------------
class CrossFrequencyAttention(nn.Module):
    """
    Frequency-domain attention between target and adjacent features.
    """
    def __init__(self, ch, head_dim=16):
        super().__init__()
        self.q_proj = Conv1x1(ch, ch)
        self.k_proj = Conv1x1(ch, ch)
        self.v_proj = Conv1x1(ch, ch)
        self.scale = math.sqrt(head_dim)

    def forward(self, F_tar, F_adj):
        # FFT transform
        Q = torch.fft.fft2(self.q_proj(F_tar), norm='ortho')
        K = torch.fft.fft2(self.k_proj(F_adj), norm='ortho')
        V = torch.fft.fft2(self.v_proj(F_tar), norm='ortho')

        # Frequency-domain correlation
        attn = (Q * K.conj()).real / (self.scale + 1e-6)
        attn = F.softmax(attn.view(attn.size(0), -1), dim=-1).view_as(attn)

        # Weighted fusion
        O = torch.fft.ifft2(V * attn, norm='ortho').real
        return O, attn

# ---------------------------------------------------------------------
# Differential Frequency Residual Self-Attention (DFR-SA)
# ---------------------------------------------------------------------
class DifferentialFrequencyAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = Conv1x1(ch, ch)
        self.rho = nn.Parameter(torch.tensor(0.5))

    def forward(self, V, O_cfr):
        V_freq = torch.fft.fft2(V, norm='ortho')
        O_freq = torch.fft.fft2(O_cfr, norm='ortho')
        diff = V_freq - self.rho * (V_freq * O_freq)
        O = torch.fft.ifft2(diff, norm='ortho').real
        return self.conv(O)

# ---------------------------------------------------------------------
# Holistic Feature Aggregation (HFA)
# ---------------------------------------------------------------------
class HFA_Module(nn.Module):
    """
    Implements holistic feature aggregation combining CFR-SA and DFR-SA.
    """
    def __init__(self, ch):
        super().__init__()
        self.qkv_conv = Conv1x1(ch, ch)
        self.cfr = CrossFrequencyAttention(ch)
        self.dfr = DifferentialFrequencyAttention(ch)
        self.reproj = Conv1x1(ch, ch)

    def forward(self, O_tar, O_adj):
        Q = self.qkv_conv(O_tar)
        K = self.qkv_conv(O_adj)
        V = self.qkv_conv(O_tar)

        # CFR self-attention
        O_cfr, attn = self.cfr(Q, K)

        # DFR residual attention
        O_dfr = self.dfr(V, O_cfr)

        # Holistic fusion
        O_hfa = self.reproj(O_dfr + Q * O_cfr)
        return O_hfa

# ---------------------------------------------------------------------
# Test and Complexity
# ---------------------------------------------------------------------
if __name__ == "__main__":
    ch = 256
    h = w = 32
    model = HFA_Module(ch)
    x_tar = torch.randn(1, ch, h, w)
    x_adj = torch.randn(1, ch, h, w)
    y = model(x_tar, x_adj)
    print("Output shape:", y.shape)

    flops, params = thop.profile(model, inputs=(x_tar, x_adj), verbose=False)
    print(f"FLOPs: {flops/1e9:.3f}G, Params: {params/1e6:.3f}M")
