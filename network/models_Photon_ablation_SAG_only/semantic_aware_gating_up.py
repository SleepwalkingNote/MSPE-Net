"""
SemanticAwareGatingUp：跨层语义引导 + 局部细节补充的门控上采样。

改进点：
1) 门控输入由单一 feats1 改为融合 feats1 与 feats2，缓解插值同质化导致的局部盲区。
2) 显式残差：X_out = X_in + gamma * Delta，且 gamma 零初始化，训练初期等价于基线。
3) 对门控输入做 LayerNorm(C)，降低局部密度差异带来的尺度不稳定。
"""
import torch
import torch.nn as nn
from utils.pointnet_util import PointNetFeaturePropagation


def _ln_bcn(x_bcn: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
    # (B,C,N) -> (B,N,C) LN(C) -> (B,C,N)
    return ln(x_bcn.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()


class _ChannelGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_bcn):
        b, c, _ = x_bcn.shape
        avg_out = self.fc(self.avg_pool(x_bcn).view(b, c))
        max_out = self.fc(self.max_pool(x_bcn).view(b, c))
        return self.sigmoid(avg_out + max_out).view(b, c, 1)


class _SpatialGateBottleneck(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        c2 = max(dim_out // 2, 8)
        self.net = nn.Sequential(
            nn.Conv1d(dim_out, c2, kernel_size=1, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Conv1d(c2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_bcn):
        return self.net(x_bcn)


class SemanticAwareGating(nn.Module):
    """双源门控：gate_input = Fuse(LN(f1), LN(f2))，mask 由 gate_input 估计。"""

    def __init__(self, dim_out, reduction=8, init_gamma=0.0):
        super().__init__()
        self.ln_f1 = nn.LayerNorm(dim_out)
        self.ln_f2 = nn.LayerNorm(dim_out)
        self.fuse = nn.Sequential(
            nn.Conv1d(dim_out * 2, dim_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim_out),
            nn.ReLU(inplace=True),
        )
        self.channel_gate = _ChannelGate(dim_out, reduction=reduction)
        self.spatial_gate = _SpatialGateBottleneck(dim_out)
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, feats1_bcn, feats2_bcn):
        f1_n = _ln_bcn(feats1_bcn, self.ln_f1)
        f2_n = _ln_bcn(feats2_bcn, self.ln_f2)

        gate_input = self.fuse(torch.cat([f1_n, f2_n], dim=1))
        ch_w = self.channel_gate(gate_input)
        sp_w = self.spatial_gate(gate_input)
        mask = ch_w * sp_w

        delta = f2_n * mask - f2_n
        return feats2_bcn + self.gamma * delta


class SemanticAwareGatingUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
        self.sag = SemanticAwareGating(dim_out, reduction=8, init_gamma=0.0)

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(
            xyz2.transpose(1, 2),
            xyz1.transpose(1, 2),
            None,
            feats1.transpose(1, 2),
        ).transpose(1, 2)

        f1 = feats1.permute(0, 2, 1)
        f2 = feats2.permute(0, 2, 1)
        feats2_filtered = self.sag(f1, f2).permute(0, 2, 1)
        return feats1 + feats2_filtered
