from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT,):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from segflux.models.model import SegFlux, parameter_report
from segflux.data.lits.cache import LiTSCaseCache
from segflux.data.lits.protocol import load_resolved_split
from scripts.lits.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolved-splits", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-max-cases", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-repeat", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    access_marker = output / "outer_test_access.json"
    if access_marker.exists() and not args.allow_repeat:
        raise RuntimeError("Outer-test evaluation already completed.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if config.get("outer_test_accessed_during_training", False):
        raise RuntimeError("Checkpoint selection used outer-test data.")
    model_size = config.get("model_size", "light")
    patch = tuple(config.get("patch_size", (96, 160, 160)))
    if len(patch) != 3:
        raise ValueError("Patch size must have three dimensions.")

    resolved = load_resolved_split(args.resolved_splits)
    if not resolved.outer_test:
        raise RuntimeError("Outer-test split is empty.")
    train_ids = {record.volume_id for record in resolved.internal_train}
    val_ids = {record.volume_id for record in resolved.internal_val}
    outer_ids = {record.volume_id for record in resolved.outer_test}
    if train_ids & outer_ids or val_ids & outer_ids:
        raise RuntimeError("Outer-test cases overlap development data.")

    device = torch.device(args.device)
    model = SegFlux(num_classes=3, model_size=model_size).to(device)
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state is None:
        raise KeyError("Checkpoint does not contain model weights.")
    model.load_state_dict(state, strict=True)
    data_cache = LiTSCaseCache(args.cache_dir, args.cache_max_cases) if args.cache_dir else None

    metrics = evaluate(
        model,
        resolved.outer_test,
        patch,
        device,
        overlap=0.25,
        tolerance_mm=3.0,
        data_cache=data_cache,
        metrics_mode="full",
    )
    payload = {
        "status": "outer_test_complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_updates": int(checkpoint.get("updates", -1)),
        "selection_metric": "internal-validation macro Dice",
        "outer_test_case_count": len(resolved.outer_test),
        "outer_test_ids": sorted(outer_ids),
        "outer_test_accessed_for_checkpoint_selection": False,
        "model_size": model_size,
        "parameter_report": parameter_report(model),
        "metrics": metrics,
        "split_audit": resolved.audit,
    }
    (output / "outer_test_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    access_marker.write_text(
        json.dumps(
            {
                "accessed": True,
                "purpose": "outer_test",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "case_count": len(resolved.outer_test),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "whole_liver_dice": metrics["liver"]["dice"],
                "tumor_dice": metrics["tumor"]["dice"],
                "macro_dice": metrics["macro_dice"],
                "tumor_hd95_mm": metrics["tumor"]["hd95_mm"],
                "tumor_nsd": metrics["tumor"]["nsd"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
