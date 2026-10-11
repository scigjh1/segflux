from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", help="outer_test_metrics.json files")
    parser.add_argument("--output")
    args = parser.parse_args()
    if len(args.metrics) != 5:
        raise ValueError("Expected metrics from five folds.")
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.metrics]

    def values(path):
        out = []
        for payload in payloads:
            value = payload
            for key in path:
                value = value[key]
            out.append(float(value))
        return np.asarray(out, dtype=np.float64)

    fields = {
        "whole_liver_dice": ("metrics", "liver", "dice"),
        "tumor_dice": ("metrics", "tumor", "dice"),
        "macro_dice": ("metrics", "macro_dice"),
        "tumor_hd95_mm": ("metrics", "tumor", "hd95_mm"),
        "tumor_nsd": ("metrics", "tumor", "nsd"),
    }
    summary = {}
    for name, path in fields.items():
        data = values(path)
        summary[name] = {
            "fold_values": data.tolist(),
            "mean": float(np.mean(data)),
            "std_population_across_folds": float(np.std(data, ddof=0)),
            "std_sample_across_folds": float(np.std(data, ddof=1)),
        }
    result = {
        "fold_count": 5,
        "summary": summary,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
