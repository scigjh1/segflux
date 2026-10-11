from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT,):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.acdc.train import (
    ACDCVolumeCache,
    build_controlled_model,
    parameter_report,
    parse_floats,
    validate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--disk-cache-dir")
    parser.add_argument("--cache-cases", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-repeat", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "heldout_test_access.json"
    if marker.exists() and not args.allow_repeat:
        raise RuntimeError("Held-out evaluation already completed.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint.get("config", {}))
    if config.get("heldout_test_used_for_checkpoint_selection", False):
        raise RuntimeError("Checkpoint selection used held-out test data.")
    model_name = config.get("model", "segflux_light")
    if model_name != "segflux_light":
        raise ValueError("Unsupported ACDC model.")
    patch = tuple(config.get("patch_size", (32, 256, 224)))
    target_spacing = tuple(
        config.get("target_spacing_zyx", parse_floats("5.0,1.5625,1.5625"))
    )
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    patient_ids = list(protocol["test"])
    if len(patient_ids) != 20:
        raise RuntimeError("ACDC test split must contain 20 patients.")

    disk_cache = Path(args.disk_cache_dir) if args.disk_cache_dir else None
    cache = ACDCVolumeCache(
        Path(args.data_root),
        patient_ids,
        target_spacing,
        args.cache_cases,
        float(config.get("sdf_truncation_mm", 20.0)),
        disk_cache,
    )
    if len(cache.records) != 40:
        raise RuntimeError("ACDC test split must contain 40 phases.")

    device = torch.device(args.device)
    model = build_controlled_model(model_name, patch).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    spacing = torch.tensor(target_spacing, dtype=torch.float32, device=device)[None]
    metrics = validate(model, model_name, cache, patch, spacing, device)

    by_patient: dict[str, list[float]] = {}
    for row in metrics["per_case"]:
        values = [float(row[str(label)]) for label in (1, 2, 3)]
        by_patient.setdefault(row["patient_id"], []).extend(values)
    patient_mean_dice = {
        patient: float(np.mean(values)) for patient, values in sorted(by_patient.items())
    }

    payload = {
        "status": "heldout_test_complete",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_updates": int(checkpoint.get("updates", -1)),
        "checkpoint_selection": "validation cohort only",
        "heldout_test_patient_count": 20,
        "heldout_test_phase_count": 40,
        "heldout_test_used_for_checkpoint_selection": False,
        "model": model_name,
        "parameter_report": parameter_report(model),
        "metrics": metrics,
        "patient_mean_dice": patient_mean_dice,
    }
    (output / "heldout_test_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    marker.write_text(
        json.dumps(
            {
                "evaluated": True,
                "purpose": "heldout",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "patient_count": 20,
                "phase_count": 40,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "rv": metrics["rv"]["mean"],
                "myo": metrics["myo"]["mean"],
                "lv": metrics["lv"]["mean"],
                "mean": metrics["macro_dice"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
