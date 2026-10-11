from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from segflux.data.records import VolumeRecord, discover_records
from segflux.utils import atomic_json_dump

SPLIT_NAMES = ("train", "val", "calibration", "test")


def make_volume_splits(
    records: list[VolumeRecord],
    seed: int,
    ratios: tuple[float, float, float, float] = (0.60, 0.15, 0.10, 0.15),
) -> dict[str, list[VolumeRecord]]:
    if not records:
        raise ValueError("Cannot split an empty dataset.")
    if len(ratios) != len(SPLIT_NAMES):
        raise ValueError("Unexpected number of split ratios.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be nonnegative.")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")

    ordered = sorted(records, key=lambda record: record.volume_id)
    random.Random(seed).shuffle(ordered)
    counts = _largest_remainder_counts(len(ordered), ratios)
    splits: dict[str, list[VolumeRecord]] = {}
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        splits[name] = sorted(
            ordered[cursor : cursor + count], key=lambda record: record.volume_id
        )
        cursor += count
    assert cursor == len(ordered)
    assert_no_volume_leakage(splits)
    return splits


def save_splits(
    splits: dict[str, list[VolumeRecord]],
    output: str | Path,
    seed: int,
    dataset_root: str | Path,
) -> None:
    assert_no_volume_leakage(splits)
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "dataset_root": str(Path(dataset_root).resolve()),
        "counts": {name: len(splits.get(name, [])) for name in SPLIT_NAMES},
        "splits": {
            name: [record.to_dict() for record in splits.get(name, [])]
            for name in SPLIT_NAMES
        },
    }
    atomic_json_dump(payload, output)


def load_split_records(path: str | Path, split: str) -> list[VolumeRecord]:
    if split not in SPLIT_NAMES:
        raise ValueError("Unsupported split name.")
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    raw_records = payload.get("splits", {}).get(split)
    if raw_records is None:
        raise ValueError("Split not found in manifest.")
    return [VolumeRecord(**record) for record in raw_records]


def assert_no_volume_leakage(splits: dict[str, Iterable[VolumeRecord]]) -> None:
    owner: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            previous = owner.get(record.volume_id)
            if previous is not None:
                raise ValueError("Volume appears in multiple splits.")
            owner[record.volume_id] = split_name


def _largest_remainder_counts(
    total: int, ratios: tuple[float, ...]
) -> tuple[int, ...]:
    exact = [total * ratio for ratio in ratios]
    counts = [int(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (exact[index] - counts[index], ratios[index]),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return tuple(counts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create patient-level data splits.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--layout",
        choices=("msd", "flat", "kits23", "amos22_ct", "amos22_mri"),
        default="msd",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=4,
        metavar=("TRAIN", "VAL", "CALIBRATION", "TEST"),
        default=(0.60, 0.15, 0.10, 0.15),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = discover_records(args.dataset_root, args.layout)
    ratios = tuple(float(value) for value in args.ratios)
    splits = make_volume_splits(records, seed=args.seed, ratios=ratios)
    save_splits(splits, args.output, args.seed, args.dataset_root)
    counts = ", ".join(f"{name}={len(splits[name])}" for name in SPLIT_NAMES)
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
