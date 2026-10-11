from __future__ import annotations

import torch
import torch.nn.functional as F


TAMR_OBJECTIVE_VERSION = "tamr_physical_aux_v2"


def _mean(values: list[torch.Tensor], *, device: torch.device) -> torch.Tensor:
    return torch.stack(values).mean() if values else torch.zeros((), device=device)


def _weighted_mean(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:


    weight = weight.detach()
    return (value * weight).sum() / weight.sum().clamp_min(epsilon)


def compose_tamr_objective(
    *,
    output,
    sdf_target: torch.Tensor,
    final_seg: torch.Tensor,
    direct_seg: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    shape_terms: list[torch.Tensor] = []
    transport_terms: list[torch.Tensor] = []
    morphology_terms: list[torch.Tensor] = []
    reversibility_terms: list[torch.Tensor] = []
    flow_terms: list[torch.Tensor] = []
    growth_terms: list[torch.Tensor] = []
    rho_terms: list[torch.Tensor] = []
    deformation_contributions: list[torch.Tensor] = []
    event_contributions: list[torch.Tensor] = []

    for scale in output.scales:
        target_scale = F.interpolate(
            sdf_target,
            size=scale.shape_sdf.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        shape_terms.append(
            F.smooth_l1_loss(scale.shape_sdf.float(), target_scale.float())
        )
        if target_scale.shape[2] > 1:
            transport_terms.extend(
                (
                    F.smooth_l1_loss(
                        scale.forward_transport[:, :, 1:].float(),
                        target_scale[:, :, 1:].float(),
                    ),
                    F.smooth_l1_loss(
                        scale.backward_transport[:, :, :-1].float(),
                        target_scale[:, :, :-1].float(),
                    ),
                )
            )
            morphology_terms.extend(
                (
                    F.smooth_l1_loss(
                        scale.forward_candidate[:, :, 1:].float(),
                        target_scale[:, :, 1:].float(),
                    ),
                    F.smooth_l1_loss(
                        scale.backward_candidate[:, :, :-1].float(),
                        target_scale[:, :, :-1].float(),
                    ),
                )
            )
        if target_scale.shape[2] > 2:
            reversibility_terms.append(
                _weighted_mean(
                    scale.cycle_error[:, :, 1:-1].float(),
                    scale.transportability[:, :, 1:-1].float(),
                )
            )
        flow_terms.extend(
            (
                scale.forward_flow_mm.abs().mean(),
                scale.backward_flow_mm.abs().mean(),
            )
        )
        growth_terms.extend(
            (
                scale.forward_growth_per_mm.abs().mean(),
                scale.backward_growth_per_mm.abs().mean(),
            )
        )
        rho_terms.append(scale.transportability.mean())
        deformation_contributions.append(scale.transport_contribution.mean())
        event_contributions.append(scale.event_contribution.mean())

    device = final_seg.device
    shape = _mean(shape_terms, device=device)
    transport = _mean(transport_terms, device=device)
    morphology = _mean(morphology_terms, device=device)
    selective_reversibility = _mean(reversibility_terms, device=device)
    flow_magnitude = _mean(flow_terms, device=device)
    growth_magnitude = _mean(growth_terms, device=device)

    total = (
        final_seg
        + 0.5 * direct_seg
        + 0.10 * shape
        + 0.05 * transport
        + 0.10 * morphology
        + 0.02 * selective_reversibility
        + 0.005 * flow_magnitude
        + 0.005 * growth_magnitude
    )
    return total, {
        "final_seg": final_seg,
        "direct_seg": direct_seg,
        "shape": shape,
        "transport_only": transport,
        "full_morphology": morphology,
        "selective_reversibility": selective_reversibility,
        "flow_magnitude_mm": flow_magnitude,
        "growth_magnitude_per_mm": growth_magnitude,
        "transportability_mean": _mean(rho_terms, device=device),
        "transport_contribution_mean": _mean(
            deformation_contributions, device=device
        ),
        "event_contribution_mean": _mean(event_contributions, device=device),
        "correction_abs_mean": output.correction.abs().mean(),
    }
