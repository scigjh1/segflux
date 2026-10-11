from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualVolumetricBlock(nn.Module):


    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.gelu(self.norm1(x)))
        y = self.conv2(F.gelu(self.norm2(y)))
        return x + y


class ResidualStage(nn.Module):
    def __init__(self, channels: int, depth: int, use_checkpoint: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResidualVolumetricBlock(channels) for _ in range(depth))
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


class Downsample(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: Sequence[int] = (2, 2, 2)
    ) -> None:
        super().__init__()
        stride = tuple(int(value) for value in stride)
        if len(stride) != 3 or any(value < 1 for value in stride):
            raise ValueError("Stride must have three positive values.")
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(in_channels), in_channels),
            nn.Conv3d(in_channels, out_channels, stride, stride=stride, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecodeStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int,
        use_checkpoint: bool,
        stride: Sequence[int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        stride = tuple(int(value) for value in stride)
        if len(stride) != 3 or any(value < 1 for value in stride):
            raise ValueError("Stride must have three positive values.")
        self.up = nn.ConvTranspose3d(in_channels, out_channels, stride, stride=stride, bias=False)
        self.fuse = nn.Conv3d(out_channels + skip_channels, out_channels, 1, bias=False)
        self.stage = ResidualStage(out_channels, depth, use_checkpoint)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.stage(self.fuse(torch.cat((x, skip), dim=1)))


@dataclass
class DirectShapeOutput:
    local_logits: torch.Tensor
    sdf: torch.Tensor
    deep_logits: list[torch.Tensor]
    multiscale_features: list[torch.Tensor]


class SegFluxDirect(nn.Module):


    def __init__(
        self,
        stem_channels: int = 32,
        channels: Sequence[int] = (64, 128, 256, 512),
        depths: Sequence[int] = (2, 3, 6, 3),
        decoder_depths: Sequence[int] = (2, 2, 2, 1),
        use_checkpoint: bool = True,
        num_classes: int = 1,
        downsample_strides: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        if len(channels) != 4 or len(depths) != 4 or len(decoder_depths) != 4:
            raise ValueError("Expected four encoder and decoder stages.")
        if downsample_strides is None:
            downsample_strides = ((2, 2, 2),) * 4
        if len(downsample_strides) != 4:
            raise ValueError("Expected four downsampling strides.")
        self.downsample_strides = tuple(tuple(int(value) for value in stride) for stride in downsample_strides)
        if any(len(stride) != 3 or any(value < 1 for value in stride) for stride in self.downsample_strides):
            raise ValueError("Invalid downsampling stride.")
        c0, c1, c2, c3 = (int(v) for v in channels)
        if int(num_classes) < 1:
            raise ValueError("num_classes must be positive.")
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.GELU(),
            ResidualVolumetricBlock(stem_channels),
        )
        self.down0 = Downsample(stem_channels, c0, self.downsample_strides[0])
        self.enc0 = ResidualStage(c0, depths[0], use_checkpoint)
        self.down1 = Downsample(c0, c1, self.downsample_strides[1])
        self.enc1 = ResidualStage(c1, depths[1], use_checkpoint)
        self.down2 = Downsample(c1, c2, self.downsample_strides[2])
        self.enc2 = ResidualStage(c2, depths[2], use_checkpoint)
        self.down3 = Downsample(c2, c3, self.downsample_strides[3])
        self.enc3 = ResidualStage(c3, depths[3], use_checkpoint)

        self.dec2 = DecodeStage(c3, c2, c2, decoder_depths[0], use_checkpoint, self.downsample_strides[3])
        self.dec1 = DecodeStage(c2, c1, c1, decoder_depths[1], use_checkpoint, self.downsample_strides[2])
        self.dec0 = DecodeStage(c1, c0, c0, decoder_depths[2], use_checkpoint, self.downsample_strides[1])
        self.dec_full = DecodeStage(c0, stem_channels, stem_channels, decoder_depths[3], use_checkpoint, self.downsample_strides[0])

        self.deep_heads = nn.ModuleList(
            (
                nn.Conv3d(c2, self.num_classes, 1),
                nn.Conv3d(c1, self.num_classes, 1),
                nn.Conv3d(c0, self.num_classes, 1),
            )
        )
        self.seg_head = nn.Conv3d(stem_channels, self.num_classes, 1)
        self.sdf_head = nn.Conv3d(stem_channels, 1, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        for module in self.modules():
            if isinstance(module, ResidualVolumetricBlock):
                nn.init.zeros_(module.conv2.weight)
        for head in self.deep_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.seg_head.weight)
        nn.init.zeros_(self.seg_head.bias)

        nn.init.zeros_(self.sdf_head.weight)
        nn.init.zeros_(self.sdf_head.bias)

    def forward(self, x: torch.Tensor) -> DirectShapeOutput:
        s = self.stem(x)
        e0 = self.enc0(self.down0(s))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        d2 = self.dec2(e3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)
        full = self.dec_full(d0, s)
        local = self.seg_head(full)
        sdf = torch.tanh(self.sdf_head(full))
        deep = [head(feature) for head, feature in zip(self.deep_heads, (d2, d1, d0))]
        deep.append(local)
        return DirectShapeOutput(local_logits=local, sdf=sdf, deep_logits=deep, multiscale_features=[d2, d1, d0, full])


def warp_sdf_xy_mm(
    source_sdf: torch.Tensor,
    forward_flow_mm: torch.Tensor,
    spacing_xy_mm: torch.Tensor | Sequence[float],
    *,
    align_corners: bool = False,
) -> torch.Tensor:


    if source_sdf.ndim != 5 or forward_flow_mm.ndim != 5:
        raise ValueError("Expected 5D distance and flow tensors.")
    if source_sdf.shape[0] != forward_flow_mm.shape[0] or source_sdf.shape[2:] != forward_flow_mm.shape[2:]:
        raise ValueError("Distance and flow shapes do not match.")
    if forward_flow_mm.shape[1] != 2:
        raise ValueError("Flow must have two in-plane channels.")
    b, _, d, h, w = source_sdf.shape
    dtype, device = source_sdf.dtype, source_sdf.device
    if align_corners:
        z = torch.linspace(-1, 1, d, dtype=dtype, device=device)
        y = torch.linspace(-1, 1, h, dtype=dtype, device=device)
        x = torch.linspace(-1, 1, w, dtype=dtype, device=device)
    else:


        z = (torch.arange(d, dtype=dtype, device=device) + 0.5) * (2.0 / d) - 1.0
        y = (torch.arange(h, dtype=dtype, device=device) + 0.5) * (2.0 / h) - 1.0
        x = (torch.arange(w, dtype=dtype, device=device) + 0.5) * (2.0 / w) - 1.0
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    base = torch.stack((xx, yy, zz), dim=-1)[None].expand(b, -1, -1, -1, -1).clone()

    spacing = torch.as_tensor(spacing_xy_mm, dtype=dtype, device=device)
    if spacing.ndim == 1:
        spacing = spacing[None].expand(b, -1)
    if spacing.shape != (b, 2):
        raise ValueError("Spacing must have shape (batch, 2).")
    flow_px_x = forward_flow_mm[:, 0] / spacing[:, 0, None, None, None].clamp_min(1e-6)
    flow_px_y = forward_flow_mm[:, 1] / spacing[:, 1, None, None, None].clamp_min(1e-6)
    if align_corners:
        norm_x = 2.0 * flow_px_x / max(1, w - 1)
        norm_y = 2.0 * flow_px_y / max(1, h - 1)
    else:
        norm_x = 2.0 * flow_px_x / max(1, w)
        norm_y = 2.0 * flow_px_y / max(1, h)
    base[..., 0] -= norm_x
    base[..., 1] -= norm_y
    return F.grid_sample(source_sdf, base, mode="bilinear", padding_mode="border", align_corners=align_corners)


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "total_millions": total / 1e6}


if __name__ == "__main__":
    model = SegFluxDirect(use_checkpoint=False)
    print(parameter_report(model))
