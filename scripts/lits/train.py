from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segflux.models.model import SegFlux, parameter_report
from segflux.losses.tamr import TAMR_OBJECTIVE_VERSION, compose_tamr_objective
from segflux.data.amos22 import make_truncated_sdf
from segflux.data.lits.common import crop_patch_multiclass, starts
from segflux.data.lits.cache import CACHE_VERSION, LiTSCaseCache
from segflux.metrics.lits import (
    multiclass_lits_dice_metrics,
    multiclass_lits_metrics,
    multiclass_segmentation_loss,
)
from segflux.data.lits.protocol import resolve_lits_splits, write_resolved_split


def seed_everything(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = True


def load_case(rec):
    image = nib.load(rec.image_path)
    label = nib.load(rec.label_path)
    image_xyz = np.asarray(image.dataobj, dtype=np.float32)
    label_xyz = np.asarray(label.dataobj, dtype=np.int16)
    labels = set(np.unique(label_xyz).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError("LiTS labels must be in {0, 1, 2}.")
    image_zyx = ((np.clip(image_xyz, -125.0, 275.0) + 125.0) / 400.0).transpose(2, 1, 0)
    target = np.eye(3, dtype=np.float32)[label_xyz].transpose(3, 2, 1, 0)
    zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    spacing = torch.tensor([[zooms[2], zooms[1], zooms[0]]], dtype=torch.float32)
    return (
        torch.from_numpy(image_zyx.copy())[None, None],
        torch.from_numpy(target.copy())[None],
        spacing,
    )


@torch.no_grad()
def evaluate(
    model,
    records,
    patch,
    device,
    *,
    overlap: float = 0.25,
    tolerance_mm: float = 3.0,
    data_cache=None,
    metrics_mode: str = "dice",
):

    model.eval()
    rows = []
    for rec in records:
        if data_cache is None:
            x_cpu, target_cpu, spacing = load_case(rec)
            spacing_zyx = tuple(float(value) for value in spacing[0].tolist())
        else:
            x_cpu, target_cpu, spacing_zyx = data_cache.full_case(rec)
            spacing = torch.tensor([spacing_zyx], dtype=torch.float32)
        _, _, depth, height, width = x_cpu.shape
        pd, ph, pw = patch
        x_cpu = F.pad(
            x_cpu,
            (0, max(0, pw - width), 0, max(0, ph - height), 0, max(0, pd - depth)),
        )
        _, _, dp, hp, wp = x_cpu.shape
        acc = torch.zeros((1, 3, dp, hp, wp), dtype=torch.float32)
        cnt = torch.zeros((1, 1, dp, hp, wp), dtype=torch.float32)
        for z in starts(dp, pd, overlap):
            for yy in starts(hp, ph, overlap):
                for xx in starts(wp, pw, overlap):
                    win = x_cpu[:, :, z:z + pd, yy:yy + ph, xx:xx + pw].to(device)
                    logits = model(win, spacing.to(device)).local_logits
                    acc[:, :, z:z + pd, yy:yy + ph, xx:xx + pw] += (
                        torch.softmax(logits, 1).float().cpu()
                    )
                    cnt[:, :, z:z + pd, yy:yy + ph, xx:xx + pw] += 1.0
        pred = np.argmax(
            (acc / cnt.clamp_min(1.0))[:, :, :depth, :height, :width][0].numpy(),
            axis=0,
        ).astype(np.int16)
        target = np.argmax(target_cpu[0].numpy(), axis=0).astype(np.int16)
        case_metrics = (
            multiclass_lits_metrics(pred, target, spacing_zyx, tolerance_mm)
            if metrics_mode == "full"
            else multiclass_lits_dice_metrics(pred, target)
        )
        rows.append({"case_id": rec.volume_id, **case_metrics})

    def mean(section: str, key: str) -> float:
        values = [
            float(row[section][key])
            for row in rows
            if key in row[section] and np.isfinite(row[section][key])
        ]
        return float(np.mean(values)) if values else float("nan")

    keys = ("dice", "hd95_mm", "nsd", "assd_mm", "precision", "recall")
    liver = {key: mean("liver", key) for key in keys}
    tumor = {key: mean("tumor", key) for key in keys}


    gt_lesions = sum(float(row["lesion_detection"]["gt_lesions"]) for row in rows)
    pred_lesions = sum(float(row["lesion_detection"]["pred_lesions"]) for row in rows)
    detected = sum(float(row["lesion_detection"]["detected_gt_lesions"]) for row in rows)
    matched = sum(float(row["lesion_detection"]["matched_pred_lesions"]) for row in rows)
    lesion_recall = detected / gt_lesions if gt_lesions else float("nan")
    lesion_precision = matched / pred_lesions if pred_lesions else float("nan")
    lesion_f1 = (
        2.0 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall)
        if np.isfinite(lesion_precision)
        and np.isfinite(lesion_recall)
        and lesion_precision + lesion_recall
        else float("nan")
    )
    global_lesion = {
        "lesion_recall": float(lesion_recall),
        "lesion_precision": float(lesion_precision),
        "lesion_f1": float(lesion_f1),
        "false_positive_lesions_per_case": float((pred_lesions - matched) / max(1, len(rows))),
        "gt_lesions": float(gt_lesions),
        "pred_lesions": float(pred_lesions),
    }
    case_mean_lesion = {
        key: float(
            np.nanmean([row["lesion_detection"][key] for row in rows])
        )
        for key in ("lesion_recall", "lesion_precision", "lesion_f1", "false_positive_lesions")
    }
    return {
        "liver": liver,
        "tumor": tumor,
        "macro_dice": float(np.nanmean([liver["dice"], tumor["dice"]])),
        "lesion_detection_global": global_lesion,
        "lesion_detection_case_mean": case_mean_lesion,
        "per_case": rows,
        "surface_tolerance_mm": float(tolerance_mm),
    }


def deep_multiclass_loss(output, target_onehot: torch.Tensor) -> torch.Tensor:

    weights = (0.125, 0.25, 0.5, 1.0)
    parts = []
    for logit in output.direct.deep_logits:
        target_scale = F.interpolate(
            target_onehot.float(), size=logit.shape[2:], mode="nearest"
        )
        parts.append(multiclass_segmentation_loss(logit, target_scale))
    return sum(weight * value for weight, value in zip(weights, parts)) / sum(weights)


def lits_tamr_objective(output, target_onehot: torch.Tensor, whole_liver_sdf: torch.Tensor):

    final_seg = multiclass_segmentation_loss(output.local_logits, target_onehot)
    direct_seg = deep_multiclass_loss(output, target_onehot)
    return compose_tamr_objective(
        output=output,
        sdf_target=whole_liver_sdf,
        final_seg=final_seg,
        direct_seg=direct_seg,
    )


lits_tamr_objective.protocol_name = f"lits_whole_liver_{TAMR_OBJECTIVE_VERSION}"


def _gradient_max(parameters) -> float:
    values = [
        float(parameter.grad.detach().abs().max().cpu())
        for parameter in parameters
        if parameter.grad is not None
    ]
    return max(values, default=0.0)


def apply_linear_warmup(optimizer, base_lr: float, completed_updates: int, warmup_updates: int) -> float:
    factor = (
        1.0
        if warmup_updates <= 0
        else min(1.0, float(completed_updates + 1) / float(warmup_updates))
    )
    current_lr = float(base_lr) * factor
    for group in optimizer.param_groups:
        group["lr"] = current_lr
    return current_lr


def _sample_one(rec, patch, args, data_cache):
    if data_cache is None:
        x, y, spacing = load_case(rec)
        x, y = crop_patch_multiclass(
            x,
            y,
            patch,
            args.foreground_oversampling,
            args.tumor_sampling_probability,
        )
    else:


        x, y, spacing = data_cache.sample_patch(
            rec,
            patch,
            args.foreground_oversampling,
            args.tumor_sampling_probability,
            "corrected_v1",
        )
    whole_liver = y[:, 1:].sum(dim=1, keepdim=True).clamp_(0.0, 1.0)
    sdf = make_truncated_sdf(
        whole_liver,
        spacing[0].numpy(),
        args.sdf_truncation_mm,
    )
    return x, y, spacing, sdf


def _sample_batch(records, patch, args, data_cache):
    samples = [
        _sample_one(records[random.randrange(len(records))], patch, args, data_cache)
        for _ in range(args.batch_size)
    ]
    return tuple(torch.cat([sample[index] for sample in samples], dim=0) for index in range(4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--data-root", help='LiTS data directory')
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inner-val-count", type=int)
    parser.add_argument("--inner-val-fraction", type=float)
    parser.add_argument("--model-size", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--patch-size", default="96,160,160")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-updates", type=int, default=60000)
    parser.add_argument("--val-interval", type=int, default=3000)
    parser.add_argument("--validation-metrics", choices=["dice", "full"], default="dice")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-max-cases", type=int, default=12)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--foreground-oversampling", type=float, default=0.66)
    parser.add_argument(
        "--tumor-sampling-probability",
        type=float,
        default=0.50,
        help=(
            'Tumor sample ratio'
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--sdf-truncation-mm", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    args = parser.parse_args()

    if args.max_updates != 60000 or args.val_interval != 3000:
        raise ValueError("Expected 60000 updates and 3000-step validation.")
    if args.batch_size != 2:
        raise ValueError("LiTS batch size must be 2.")
    if not 0.0 <= args.tumor_sampling_probability <= 1.0:
        raise ValueError("Tumor sampling probability must be in [0, 1].")

    seed_everything(args.seed)
    device = torch.device(args.device)
    patch = tuple(map(int, args.patch_size.split(",")))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_lits_splits(
        args.split_file,
        seed=args.seed,
        inner_val_count=args.inner_val_count,
        inner_val_fraction=args.inner_val_fraction,
        data_root=args.data_root,
    )
    train_records = resolved.internal_train
    val_records = resolved.internal_val
    if not train_records or not val_records:
        raise RuntimeError("Training and validation splits must be nonempty.")
    write_resolved_split(output_dir / "resolved_splits.json", resolved)

    data_cache = LiTSCaseCache(args.cache_dir, args.cache_max_cases) if args.cache_dir else None
    model = SegFlux(num_classes=3, model_size=args.model_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and device.type == "cuda"
        else torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    )

    config = {
        **vars(args),
        "patch_size": patch,
        "protocol_version": "lits_nested_v2",
        "objective": lits_tamr_objective.protocol_name,
        "cache_version": CACHE_VERSION if data_cache else None,
        "lr_schedule": "linear_warmup_then_constant",
        "tensor_axis_order": "zyx",
        "spacing_order": "sz_sy_sx",
        "num_classes": 3,
        "class_names": ["background", "liver", "tumor"],
        "geometric_foreground": "label 1 union label 2",
        "segmentation_loss": "cross-entropy + foreground soft Dice (unweighted)",
        "deep_supervision_weights": [0.125, 0.25, 0.5, 1.0],
        "checkpoint_metric": "internal-validation macro Dice",
        "outer_test_accessed_during_training": False,
        "split_audit": resolved.audit,
        "parameter_report": parameter_report(model),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    history = []
    best = -1.0
    updates = 0
    losses: list[float] = []
    start = time.time()
    grad_audit_max = {"scale_alpha": 0.0, "shape_head": 0.0, "context_head": 0.0}
    latest_parts: dict[str, float] = {}

    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scaler_state_dict") is not None:
            scaler.load_state_dict(state["scaler_state_dict"])
        updates = int(state.get("updates", 0))
        history = list(state.get("history", []))
        grad_audit_max.update(state.get("pmf_gradient_audit_max", {}))
        best = max(
            [float(row.get("validation", {}).get("macro_dice", -1.0)) for row in history]
            + [-1.0]
        )

    validated_updates = {
        int(row["updates"])
        for row in history
        if row.get("validation") is not None and "updates" in row
    }
    if updates > 0 and updates % args.val_interval == 0 and updates not in validated_updates:
        metrics = evaluate(
            model,
            val_records,
            patch,
            device,
            overlap=0.25,
            data_cache=data_cache,
            metrics_mode=args.validation_metrics,
        )
        row = {
            "updates": updates,
            "train_loss": None,
            "loss_parts": {},
            "gradient_norm": None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation": metrics,
            "elapsed_seconds": time.time() - start,
            "resumed_validation": True,
        }
        history.append(row)
        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "updates": updates,
            "history": history,
            "config": config,
            "pmf_gradient_audit_max": grad_audit_max,
        }
        torch.save(state, output_dir / "checkpoint_last.pt")
        if metrics["macro_dice"] > best:
            best = metrics["macro_dice"]
            torch.save(state, output_dir / "best_internal_macro_dice.pt")

    while updates < args.max_updates:
        x, target, spacing, sdf = _sample_batch(
            train_records, patch, args, data_cache
        )
        tumor_voxels = int(target[:, 2].sum().item())
        tumor_cases = int((target[:, 2].flatten(1).sum(1) > 0).sum().item())
        x, target, spacing, sdf = (
            x.to(device),
            target.to(device),
            spacing.to(device),
            sdf.to(device),
        )
        current_lr = apply_linear_warmup(
            optimizer, args.learning_rate, updates, args.warmup_updates
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(x, spacing)
            loss, parts = lits_tamr_objective(output, target, sdf)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        alpha_grad = _gradient_max([model.scale_alpha])
        shape_grad = _gradient_max(
            parameter
            for scale in model.flux_scales
            for parameter in scale.shape_head.parameters()
        )
        context_grad = _gradient_max(
            parameter
            for scale in model.flux_scales
            for parameter in scale.context[-1].parameters()
        )
        grad_audit_max["scale_alpha"] = max(grad_audit_max["scale_alpha"], alpha_grad)
        grad_audit_max["shape_head"] = max(grad_audit_max["shape_head"], shape_grad)
        grad_audit_max["context_head"] = max(grad_audit_max["context_head"], context_grad)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0))
        scaler.step(optimizer)
        scaler.update()
        updates += 1
        losses.append(float(loss.detach()))
        latest_parts = {
            key: float(value.detach().float().cpu()) for key, value in parts.items()
        }

        if updates == 10:
            inactive = [
                name
                for name, value in grad_audit_max.items()
                if not np.isfinite(value) or value <= 0.0
            ]
            if float(model.scale_alpha.detach().abs().max().cpu()) <= 0.0:
                inactive.append("scale_alpha_value")
            if inactive:
                raise RuntimeError("Inactive model gradients detected.")

        if updates % args.progress_interval == 0:
            (output_dir / "progress.json").write_text(
                json.dumps(
                    {
                        "updates": updates,
                        "train_loss": float(np.mean(losses[-args.progress_interval:])),
                        "learning_rate": current_lr,
                        "elapsed_seconds": time.time() - start,
                        "loss_parts": latest_parts,
                        "tumor_voxels_in_last_batch": tumor_voxels,
                        "tumor_positive_samples_in_last_batch": tumor_cases,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

        state = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "updates": updates,
            "history": history,
            "config": config,
            "pmf_gradient_audit_max": grad_audit_max,
        }
        if updates % args.checkpoint_interval == 0:
            torch.save(state, output_dir / "checkpoint_last.pt")

        if updates % args.val_interval == 0 or updates == args.max_updates:
            metrics = evaluate(
                model,
                val_records,
                patch,
                device,
                overlap=0.25,
                data_cache=data_cache,
                metrics_mode=args.validation_metrics,
            )
            row = {
                "updates": updates,
                "train_loss": float(np.mean(losses[-args.val_interval:])),
                "loss_parts": latest_parts,
                "gradient_norm": grad_norm,
                "learning_rate": current_lr,
                "pmf_gradient_audit_max": dict(grad_audit_max),
                "scale_alpha": model.scale_alpha.detach().cpu().tolist(),
                "validation": metrics,
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            state["history"] = history
            torch.save(state, output_dir / "checkpoint_last.pt")
            if metrics["macro_dice"] > best:
                best = metrics["macro_dice"]
                state["best_internal_macro_dice"] = best
                torch.save(state, output_dir / "best_internal_macro_dice.pt")
            model.train()
            print(
                json.dumps(
                    {
                        "model": "segflux_light",
                        "updates": updates,
                        "internal_val_macro_dice": metrics["macro_dice"],
                        "internal_val_liver_dice": metrics["liver"]["dice"],
                        "internal_val_tumor_dice": metrics["tumor"]["dice"],
                        "best_internal_macro_dice": best,
                    }
                ),
                flush=True,
            )

    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "training_complete",
                "best_internal_macro_dice": best,
                "history": history,
                "config": config,
                "outer_test_accessed": False,
                "next_step": (
                    "evaluate_lits_outer.py"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
