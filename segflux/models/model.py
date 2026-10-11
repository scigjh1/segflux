from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from segflux.models.backbone import (
    SegFluxDirect,
    DirectShapeOutput,
    _groups,
    warp_sdf_xy_mm,
)


MODEL_SIZE_CONFIGS = {
    "light": {
        "stem_channels": 16,
        "channels": (32, 64, 128, 256),
        "depths": (1, 2, 3, 2),
        "decoder_depths": (1, 1, 1, 1),
    },
    "medium": {
        "stem_channels": 24,
        "channels": (48, 96, 192, 384),
        "depths": (2, 2, 4, 2),
        "decoder_depths": (1, 2, 2, 1),
    },
    "heavy": {
        "stem_channels": 32,
        "channels": (64, 128, 256, 512),
        "depths": (2, 3, 6, 3),
        "decoder_depths": (2, 2, 2, 1),
    },
}


def direct_size_kwargs(model_size: str) -> dict:
    key = str(model_size).lower()
    if key not in MODEL_SIZE_CONFIGS:
        raise ValueError("Unsupported model size.")
    return dict(MODEL_SIZE_CONFIGS[key])


@dataclass
class TAMRScaleOutput:
    shape_sdf: torch.Tensor
    forward_transport: torch.Tensor
    backward_transport: torch.Tensor
    forward_candidate: torch.Tensor
    backward_candidate: torch.Tensor
    forward_flow_mm: torch.Tensor
    backward_flow_mm: torch.Tensor
    forward_growth_per_mm: torch.Tensor
    backward_growth_per_mm: torch.Tensor
    transportability: torch.Tensor
    transport_contribution: torch.Tensor
    event_contribution: torch.Tensor
    transport_consensus: torch.Tensor
    full_consensus: torch.Tensor
    routed_consensus: torch.Tensor
    cycle_error: torch.Tensor

    @property
    def gate(self) -> torch.Tensor:

        return self.transportability


@dataclass
class TAMROutput:
    local_logits: torch.Tensor
    direct_logits: torch.Tensor
    direct: DirectShapeOutput
    scales: list[TAMRScaleOutput]
    correction: torch.Tensor


def _shift_previous(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :, :1], value[:, :, :-1]), dim=2)


def _shift_next(value: torch.Tensor) -> torch.Tensor:
    return torch.cat((value[:, :, 1:], value[:, :, -1:]), dim=2)


class TransportabilityAwareMorphologyScale(nn.Module):


    def __init__(
        self,
        channels: int,
        *,
        max_flow_mm: float = 10.0,
        max_growth_per_mm: float = 0.10,
        routing_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.max_flow_mm = float(max_flow_mm)
        self.max_growth_per_mm = float(max_growth_per_mm)
        self.routing_epsilon = float(routing_epsilon)
        self.shape_head = nn.Conv3d(channels, 1, 1)


        self.context = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.Conv3d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv3d(channels, 6, 1),
        )
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(
        self,
        feature: torch.Tensor,
        spacing_zyx_mm: torch.Tensor,
        input_shape: Sequence[int],
    ) -> tuple[TAMRScaleOutput, torch.Tensor]:
        shape = torch.tanh(self.shape_head(feature))
        raw = self.context(feature)
        forward_flow = self.max_flow_mm * torch.tanh(raw[:, 0:2])
        forward_growth = self.max_growth_per_mm * torch.tanh(raw[:, 2:3])
        backward_flow = self.max_flow_mm * torch.tanh(raw[:, 3:5])
        backward_growth = self.max_growth_per_mm * torch.tanh(raw[:, 5:6])

        input_depth, input_height, input_width = (int(value) for value in input_shape)
        _, _, depth, height, width = feature.shape
        scale_z = input_depth / depth
        scale_y = input_height / height
        scale_x = input_width / width
        delta_z = spacing_zyx_mm[:, 0] * scale_z
        feature_spacing_xy = torch.stack(
            (spacing_zyx_mm[:, 2] * scale_x, spacing_zyx_mm[:, 1] * scale_y), dim=1
        )

        previous_shape = _shift_previous(shape)
        next_shape = _shift_next(shape)
        forward_transport = warp_sdf_xy_mm(previous_shape, forward_flow, feature_spacing_xy)
        backward_transport = warp_sdf_xy_mm(next_shape, backward_flow, feature_spacing_xy)
        dz = delta_z[:, None, None, None, None].to(dtype=shape.dtype)
        forward_event = dz * forward_growth
        backward_event = dz * backward_growth
        forward_full = forward_transport + forward_event
        backward_full = backward_transport + backward_event


        deformation_change = 0.5 * (
            (forward_transport - previous_shape).abs()
            + (backward_transport - next_shape).abs()
        )
        event_change = 0.5 * (forward_event.abs() + backward_event.abs())
        rho = deformation_change / (deformation_change + event_change + self.routing_epsilon)
        rho = rho.clamp_(0.0, 1.0)

        transport_consensus = 0.5 * (forward_transport + backward_transport)
        full_consensus = 0.5 * (forward_full + backward_full)


        routing_weight = rho.detach()
        routed_consensus = routing_weight * transport_consensus + (1.0 - routing_weight) * full_consensus


        flow_to_next = _shift_next(forward_flow)
        flow_to_previous = _shift_previous(backward_flow)
        cycle_from_next = warp_sdf_xy_mm(
            warp_sdf_xy_mm(shape, flow_to_next, feature_spacing_xy),
            backward_flow,
            feature_spacing_xy,
        )
        cycle_from_previous = warp_sdf_xy_mm(
            warp_sdf_xy_mm(shape, flow_to_previous, feature_spacing_xy),
            forward_flow,
            feature_spacing_xy,
        )
        cycle_error = 0.5 * ((cycle_from_next - shape).abs() + (cycle_from_previous - shape).abs())

        output = TAMRScaleOutput(
            shape_sdf=shape,
            forward_transport=forward_transport,
            backward_transport=backward_transport,
            forward_candidate=forward_full,
            backward_candidate=backward_full,
            forward_flow_mm=forward_flow,
            backward_flow_mm=backward_flow,
            forward_growth_per_mm=forward_growth,
            backward_growth_per_mm=backward_growth,
            transportability=rho,
            transport_contribution=deformation_change,
            event_contribution=event_change,
            transport_consensus=transport_consensus,
            full_consensus=full_consensus,
            routed_consensus=routed_consensus,
            cycle_error=cycle_error,
        )
        return output, routed_consensus - shape


class SegFlux(nn.Module):


    def __init__(
        self,
        *,
        num_classes: int = 1,
        correction_bound: float = 3.0,
        use_checkpoint: bool = True,
        deformation_enabled: bool = True,
        growth_enabled: bool = True,


        model_size: str = "light",
        downsample_strides: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        if not deformation_enabled or not growth_enabled:
            raise ValueError("Deformation and morphology must both be enabled.")
        direct_kwargs = direct_size_kwargs(model_size)
        self.model_size = str(model_size).lower()
        self.direct = SegFluxDirect(
            num_classes=int(num_classes),
            use_checkpoint=use_checkpoint,
            downsample_strides=downsample_strides,
            **direct_kwargs,
        )
        self.num_classes = int(num_classes)
        self.registry_name = f"SegFlux-{self.model_size.capitalize()}-PMF-TAMR"
        self.routing_name = "analytical_A_over_A_plus_B"
        self.learned_gate = False
        self.selective_reversibility = True
        self.protocol_version = "segflux_v1"
        channels = direct_kwargs["channels"]
        self.flux_scales = nn.ModuleList(
            (
                TransportabilityAwareMorphologyScale(channels[2]),
                TransportabilityAwareMorphologyScale(channels[1]),
            )
        )

        self.scale_alpha = nn.Parameter(torch.zeros(2))
        self.correction_bound = float(correction_bound)

    def forward(self, x: torch.Tensor, spacing_zyx_mm: torch.Tensor) -> TAMROutput:
        direct = self.direct(x)
        residuals: list[torch.Tensor] = []
        scale_outputs: list[TAMRScaleOutput] = []
        for module, feature, alpha in zip(self.flux_scales, direct.multiscale_features[:2], self.scale_alpha):
            scale_output, residual = module(feature, spacing_zyx_mm, direct.local_logits.shape[2:])
            scale_outputs.append(scale_output)
            residual = F.interpolate(
                residual, size=direct.local_logits.shape[2:], mode="trilinear", align_corners=False
            )
            residuals.append(alpha * residual)
        combined = torch.stack(residuals, dim=0).sum(dim=0)
        correction = self.correction_bound * torch.tanh(combined)
        if self.num_classes == 1:
            final = direct.local_logits + correction
        else:


            final = direct.local_logits.clone()
            final[:, 1:] = final[:, 1:] + correction
        return TAMROutput(
            local_logits=final,
            direct_logits=direct.local_logits,
            direct=direct,
            scales=scale_outputs,
            correction=correction,
        )


def parameter_report(model: nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "total_millions": total / 1e6}
