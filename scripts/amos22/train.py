from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for root in (PROJECT_ROOT,):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from segflux.models.model import SegFlux, parameter_report
from segflux.losses.tamr import compose_tamr_objective, TAMR_OBJECTIVE_VERSION
from segflux.data.amos22 import (
    _starts,
    crop_patch,
    largest_component,
    load_native_xy_cpu,
    make_truncated_sdf,
    segmentation_loss,
)
from segflux.data.splits import load_split_records
from segflux.metrics.segmentation import binary_volume_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def deep_direct_loss(output, target: torch.Tensor) -> torch.Tensor:
    weights = (0.125, 0.25, 0.5, 1.0)
    parts = [segmentation_loss(logit, target, 1.0) for logit in output.direct.deep_logits]
    return sum(weight * value for weight, value in zip(weights, parts)) / sum(weights)


def flux_objective(output, target: torch.Tensor, sdf_target: torch.Tensor):

    return compose_tamr_objective(
        output=output,
        sdf_target=sdf_target,
        final_seg=segmentation_loss(output.local_logits, target, 1.0),
        direct_seg=deep_direct_loss(output, target),
    )

flux_objective.protocol_name = TAMR_OBJECTIVE_VERSION

@torch.no_grad()
def _flux_spacing(spacing_zyx, use_physical_spacing: bool):
    if use_physical_spacing:
        return spacing_zyx


    return (1.0, 1.0, 1.0)


def sliding_probabilities(model, x_cpu, spacing_zyx, patch, device, overlap, direct_only: bool, use_physical_spacing: bool = True):
    _, _, original_depth, original_height, original_width = x_cpu.shape
    pd, ph, pw = patch
    padded = F.pad(x_cpu, (0, max(0, pw-original_width), 0, max(0, ph-original_height), 0, max(0, pd-original_depth)))
    _, _, depth, height, width = padded.shape
    accumulator = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)
    counter = torch.zeros_like(accumulator)
    spacing_tensor = torch.as_tensor(
        _flux_spacing(spacing_zyx, use_physical_spacing), dtype=torch.float32, device=device
    )[None]
    for z in _starts(depth, pd, overlap):
        for yy in _starts(height, ph, overlap):
            for xx in _starts(width, pw, overlap):
                window = padded[:, :, z:z+pd, yy:yy+ph, xx:xx+pw].to(device, non_blocking=True)
                if direct_only:
                    logits = model.direct(window).local_logits
                else:
                    logits = model(window, spacing_tensor).local_logits
                probability = torch.sigmoid(logits).float().cpu()
                accumulator[:, :, z:z+pd, yy:yy+ph, xx:xx+pw] += probability
                counter[:, :, z:z+pd, yy:yy+ph, xx:xx+pw] += 1.0
    return (accumulator / counter.clamp_min(1.0))[:, :, :original_depth, :original_height, :original_width]


@torch.no_grad()
def evaluate(
    model,
    records,
    xy_size,
    patch,
    overlap,
    device,
    direct_only=False,
    foreground_label_value=0,
    use_physical_spacing=True,
):
    model.eval()
    rows = []
    for record in records:
        x, _, native_target, spacing_xyz, spacing_zyx, native_zyx = load_native_xy_cpu(
            record, xy_size, foreground_label_value
        )
        probability = sliding_probabilities(
            model, x, spacing_zyx, patch, device, overlap, direct_only, use_physical_spacing
        )
        original_xyz = tuple(int(value) for value in nib.load(record.label_path).shape)
        original_zyx = (original_xyz[2], original_xyz[1], original_xyz[0])
        probability = F.interpolate(probability, size=original_zyx, mode="trilinear", align_corners=False)[0, 0].numpy()
        prediction = probability >= 0.5
        target = native_target.astype(bool)
        spacing = (float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0]))
        raw = binary_volume_metrics(prediction, target, spacing)
        lcc_prediction = largest_component(prediction)
        lcc = binary_volume_metrics(lcc_prediction, target, spacing)
        for metrics, candidate in ((raw, prediction), (lcc, lcc_prediction)):
            intersection = int(np.logical_and(candidate, target).sum())
            metrics["precision"] = float(intersection / max(1, int(candidate.sum())))
            metrics["recall"] = float(intersection / max(1, int(target.sum())))
        rows.append({
            "case_id": record.volume_id,
            "raw": raw,
            "lcc": lcc,
            "pred_fg_ratio": float(prediction.mean()),
            "target_fg_ratio": float(target.mean()),
            "pred_target_volume_ratio": float(prediction.sum() / max(1, target.sum())),
            "native_zyx": native_zyx,
        })

    def average(section, key):
        values = [float(row[section][key]) for row in rows if np.isfinite(row[section][key])]
        return float(np.mean(values)) if values else float("inf")
    return {
        "raw_dice": average("raw", "dice"),
        "lcc_dice": average("lcc", "dice"),
        "raw_hd95": average("raw", "hd95_mm"),
        "lcc_hd95": average("lcc", "hd95_mm"),
        "raw_nsd": average("raw", "nsd"),
        "lcc_nsd": average("lcc", "nsd"),
        "raw_precision": average("raw", "precision"),
        "raw_recall": average("raw", "recall"),
        "per_case": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--foreground-label-value", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", default="96,256,256")
    parser.add_argument("--foreground-oversampling", type=float, default=0.66)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=60000)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--val-interval", type=int, default=3000)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--sliding-overlap", type=float, default=0.25)
    parser.add_argument("--joint-direct-lr", type=float, default=2e-4)
    parser.add_argument("--flux-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sdf-truncation-mm", type=float, default=20.0)
    parser.add_argument("--index-spacing-control", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Resume from this experiment's last.pt")
    args = parser.parse_args()
    args.xy_size = 0
    args.scratch_joint = True
    args.deformation_disabled = False
    args.growth_disabled = False
    if args.batch_size != 2:
        raise ValueError("AMOS22 batch size must be 2.")

    seed_everything(args.seed)
    patch = tuple(int(value) for value in args.patch_size.split(","))
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = Path(args.split_file)
    train_records = load_split_records(str(split_path), "train")
    val_records = load_split_records(str(split_path), "val")
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_rows = list(split_payload.get("splits", {}).get("test", []))
    test_ids = {str(row.get("volume_id", row.get("case_id", ""))) for row in test_rows}
    if {r.volume_id for r in train_records} & test_ids:
        raise RuntimeError("Training cases overlap the test split.")
    if {r.volume_id for r in val_records} & test_ids:
        raise RuntimeError("Validation cases overlap the test split.")

    model = SegFlux(model_size="light").to(device)
    provenance = {
        "source_initialization": "random_joint_scratch",
        "external_model_weights": None,
        "stunet_dependency": False,
    }
    decoder_parameters = list(model.direct.parameters())
    direct_lr = args.joint_direct_lr
    flux_parameters = list(model.flux_scales.parameters()) + [model.scale_alpha]
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_parameters, "lr": direct_lr, "base_lr": direct_lr, "name": "direct_joint"},
            {"params": flux_parameters, "lr": args.flux_lr, "base_lr": args.flux_lr, "name": "flux"},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    resume_state = None
    resume_rng_restored = False
    if args.resume:
        resume_path = Path(args.resume)
        resume_state = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        resume_rng_restored = resume_state.get("rng_state") is not None


    if resume_state is None:
        model.eval()
        sample_x, _, _, _, sample_spacing, _ = load_native_xy_cpu(
            train_records[0], args.xy_size, args.foreground_label_value
        )
        sample_x, _ = crop_patch(sample_x, torch.zeros_like(sample_x), patch, 0.0)
        with torch.no_grad():
            identity = model(sample_x.to(device), torch.as_tensor(sample_spacing, device=device)[None])
        identity_error = float((identity.local_logits - identity.direct_logits).abs().max().cpu())
        if identity_error >= 1e-5:
            raise RuntimeError("Initial correction is not zero.")
    else:
        identity_error = float(resume_state.get("config", {}).get("identity_max_abs_error", 0.0))

    objective_name = getattr(flux_objective, "protocol_name", "legacy_pmf_objective")
    model_name = getattr(model, "registry_name", "SegFlux-v1-Physical-Morphological-Flux")
    config = {
        **vars(args),
        "patch_size": patch,
        "model": model_name,
        "model_size": getattr(model, "model_size", None),
        "training_objective": objective_name,
        "routing": getattr(model, "routing_name", None),
        "learned_gate": getattr(model, "learned_gate", None),
        "selective_reversibility": getattr(model, "selective_reversibility", False),
        "protocol_version": getattr(model, "protocol_version", "legacy_pmf_v1"),
        "training_mode": "joint_random_scratch",
        "ablation": "pmf_plus_analytical_tamr_routing",
        "identity_max_abs_error": identity_error,
        "provenance": provenance,
        "external_model_weights": None,
        "teacher_model": None,
        "stunet_dependency": False,
        "lifecycle_enabled": False,
        "cmp_enabled": False,
        "spacing_mode": "index" if args.index_spacing_control else "physical_mm",
        "test_accessed": False,
        "parameter_report": parameter_report(model),
        "checkpoint_metric": "original_space_raw_foreground_volume_dice",
        "batch_semantics": "true batch of 2 independent 3-D patches",
        "test_record_count_metadata_only": len(test_rows),
        "resume_checkpoint": str(Path(args.resume).resolve()) if args.resume else None,
        "resume_rng_restored": resume_rng_restored,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


    initial_path = output_dir / "initial_identity_validation.json"
    if resume_state is not None and initial_path.exists():
        initial_audit = json.loads(initial_path.read_text())
        initial_direct, initial_final = initial_audit["direct"], initial_audit["final"]
    else:
        initial_direct = evaluate(
            model, val_records, args.xy_size, patch, args.sliding_overlap, device,
            direct_only=True, foreground_label_value=args.foreground_label_value,
            use_physical_spacing=not args.index_spacing_control,
        )
        initial_final = evaluate(
            model, val_records, args.xy_size, patch, args.sliding_overlap, device,
            direct_only=False, foreground_label_value=args.foreground_label_value,
            use_physical_spacing=not args.index_spacing_control,
        )
        initial_audit = {"direct": initial_direct, "final": initial_final, "raw_dice_difference": initial_final["raw_dice"] - initial_direct["raw_dice"]}
        initial_path.write_text(json.dumps(initial_audit, indent=2), encoding="utf-8")
    baseline_raw = float(resume_state.get("baseline_raw", initial_direct["raw_dice"])) if resume_state else initial_direct["raw_dice"]

    history = list(resume_state.get("history", [])) if resume_state else []
    updates = int(resume_state.get("updates", 0)) if resume_state else 0
    best_raw = max([baseline_raw] + [float(row["final_validation"]["raw_dice"]) for row in history])
    best_lcc = max([float(initial_direct["lcc_dice"])] + [float(row["final_validation"]["lcc_dice"]) for row in history])
    best_hd95 = min([float(initial_direct["raw_hd95"])] + [float(row["final_validation"]["raw_hd95"]) for row in history])
    optimizer.zero_grad(set_to_none=True)
    loss_window = []
    started = time.time()
    stop = False


    if resume_state is not None and resume_state.get("rng_state") is not None:
        rng = resume_state["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if device.type == "cuda" and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    else:
        seed_everything(args.seed)
    while updates < args.max_updates and not stop:
        order = list(range(len(train_records)))
        random.shuffle(order)
        for batch_start in range(0, len(order), args.batch_size):
            indices = order[batch_start:batch_start + args.batch_size]
            while len(indices) < args.batch_size:
                indices.append(order[random.randrange(len(order))])
            batch_x, batch_y, batch_sdf, batch_spacing = [], [], [], []
            for index in indices:
                x_cpu, y_cpu, _, _, spacing_zyx, _ = load_native_xy_cpu(
                    train_records[index], args.xy_size, args.foreground_label_value
                )
                x_cpu, y_cpu = crop_patch(x_cpu, y_cpu, patch, args.foreground_oversampling)
                sdf_cpu = make_truncated_sdf(y_cpu, spacing_zyx, args.sdf_truncation_mm)
                batch_x.append(x_cpu); batch_y.append(y_cpu); batch_sdf.append(sdf_cpu)
                batch_spacing.append(torch.as_tensor(
                    _flux_spacing(spacing_zyx, not args.index_spacing_control),
                    dtype=torch.float32,
                )[None])
            x = torch.cat(batch_x, dim=0).to(device)
            y = torch.cat(batch_y, dim=0).to(device)
            sdf = torch.cat(batch_sdf, dim=0).to(device)
            spacing = torch.cat(batch_spacing, dim=0).to(device)
            if updates < args.warmup_updates:
                multiplier = max(1e-3, (updates + 1) / max(1, args.warmup_updates))
            else:
                progress = min(1.0, (updates-args.warmup_updates) / max(1, args.max_updates-args.warmup_updates))
                multiplier = 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                output = model(x, spacing)
                loss, parts = flux_objective(output, y, sdf)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            scaler.scale(loss).backward()
            loss_window.append(float(loss.detach().cpu()))
            scaler.unscale_(optimizer)
            gradient_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 12.0))
            scaler.step(optimizer)
            scaler.update()
            updates += 1

            if args.log_interval and updates % args.log_interval == 0 and updates % args.val_interval:
                print(json.dumps({"type": "train_progress", "updates": updates, "loss": float(np.mean(loss_window[-args.log_interval:])), "gradient_norm": gradient_norm}), flush=True)
            if updates % args.val_interval == 0 or updates == args.max_updates:
                final_validation = evaluate(
                    model,
                    val_records,
                    args.xy_size,
                    patch,
                    args.sliding_overlap,
                    device,
                    direct_only=False,
                    foreground_label_value=args.foreground_label_value,
                    use_physical_spacing=not args.index_spacing_control,
                )
                direct_validation = evaluate(
                    model,
                    val_records,
                    args.xy_size,
                    patch,
                    args.sliding_overlap,
                    device,
                    direct_only=True,
                    foreground_label_value=args.foreground_label_value,
                    use_physical_spacing=not args.index_spacing_control,
                )
                row = {
                    "updates": updates,
                    "train_loss_window": float(np.mean(loss_window[-args.val_interval:])),
                    "last_loss_parts": {key: float(value.detach().cpu()) for key, value in parts.items()},
                    "gradient_norm": gradient_norm,
                    "scale_alpha": model.scale_alpha.detach().cpu().tolist(),
                    "peak_memory_mb": int(torch.cuda.max_memory_allocated(device)/2**20),
                    "elapsed_seconds": time.time()-started,
                    "final_validation": final_validation,
                    "direct_validation": direct_validation,
                    "delta_raw_vs_initial_direct": final_validation["raw_dice"] - baseline_raw,
                }
                history.append(row)
                state = {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "updates": updates, "history": history,
                    "config": config, "baseline_raw": baseline_raw,
                    "rng_state": {
                        "python": random.getstate(), "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    },
                }
                torch.save(state, output_dir / "last.pt")
                if final_validation["raw_dice"] > best_raw:
                    best_raw = final_validation["raw_dice"]
                    torch.save(state, output_dir / "best_raw_dice.pt")
                if final_validation["lcc_dice"] > best_lcc:
                    best_lcc = final_validation["lcc_dice"]
                    torch.save(state, output_dir / "best_lcc_dice.pt")
                if final_validation["raw_hd95"] < best_hd95:
                    best_hd95 = final_validation["raw_hd95"]
                    torch.save(state, output_dir / "best_hd95.pt")
                print(json.dumps(row), flush=True)
                torch.cuda.reset_peak_memory_stats(device)
                model.train()
                if updates >= 2000 and best_raw < baseline_raw - 0.003:
                    stop = True
                    print(json.dumps({"type": "flux_no_go_degradation_stop", "baseline": baseline_raw, "best": best_raw}), flush=True)
                    break

    summary = {
        "status": "complete",
        "updates": updates,
        "baseline_raw_dice": baseline_raw,
        "baseline_raw_hd95": initial_direct["raw_hd95"],
        "best_raw_dice": best_raw,
        "best_lcc_dice": best_lcc,
        "best_raw_hd95": best_hd95,
        "delta_best_raw": best_raw - baseline_raw,
        "deformation_enabled": not args.deformation_disabled,
        "growth_enabled": not args.growth_disabled,
        "spacing_mode": "index" if args.index_spacing_control else "physical_mm",
        "history": history,
        "config": config,
        "test_accessed": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
