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
from segflux.data.splits import load_split_records
from scripts.amos22.train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "calibration", "test"])
    parser.add_argument("--role", required=True, choices=["outer_test", "untouched_holdout"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-repeat", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / f"{args.role}_access.json"
    if marker.exists() and not args.allow_repeat:
        raise RuntimeError("Evaluation for this split already completed.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if config.get("test_accessed", False):
        raise RuntimeError("Checkpoint selection used test data.")
    records = load_split_records(args.split_file, args.split_name)
    if not records:
        raise RuntimeError("Evaluation split is empty.")

    model = SegFlux(
        num_classes=1,
        model_size=config.get("model_size", "light") or "light",
        deformation_enabled=not bool(config.get("deformation_disabled", False)),
        growth_enabled=not bool(config.get("growth_disabled", False)),
    )
    state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    if state is None:
        raise KeyError("Checkpoint does not contain model weights.")
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.to(device)
    patch = tuple(config.get("patch_size", (96, 256, 256)))
    metrics = evaluate(
        model,
        records,
        int(config.get("xy_size", 0)),
        patch,
        float(config.get("sliding_overlap", 0.25)),
        device,
        direct_only=False,
        foreground_label_value=int(config.get("foreground_label_value", 6)),
        use_physical_spacing=not bool(config.get("index_spacing_control", False)),
    )
    payload = {
        "status": "complete",
        "role": args.role,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_updates": int(checkpoint.get("updates", -1)),
        "evaluation_case_count": len(records),
        "case_ids": [record.volume_id for record in records],
        "checkpoint_selection_used_this_split": False,
        "parameter_report": parameter_report(model),
        "metrics": metrics,
    }
    (output / f"{args.role}_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "accessed": True,
                "role": args.role,
                "purpose": "evaluation",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "case_count": len(records),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "dice": metrics["raw_dice"],
        "hd95_mm": metrics["raw_hd95"],
        "nsd": metrics["raw_nsd"],
        "precision": metrics["raw_precision"],
        "recall": metrics["raw_recall"],
    }), flush=True)


if __name__ == "__main__":
    main()
