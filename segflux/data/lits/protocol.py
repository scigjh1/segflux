from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class ResolvedLiTSSplits:
    internal_train: list[Any]
    internal_val: list[Any]
    outer_test: list[Any]
    audit: dict[str, Any]


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("volume_id", row.get("case_id", row.get("sample_id"))))


def _record(row: dict[str, Any]) -> Any:
    return SimpleNamespace(
        volume_id=row["volume_id"],
        sample_id=row.get("sample_id", f"lits:{row['volume_id']}"),
        dataset=row.get("dataset", "lits_official_labeled_train"),
        image_path=row["image_path"],
        label_path=row["label_path"],
        raw=dict(row),
    )


def _records(rows: list[dict[str, Any]]) -> list[Any]:
    return [_record(row) for row in rows]


def _resolve_case_file(root: Path, subdir: str, stem: str) -> str:

    candidates = [root / subdir / f"{stem}.nii", root / subdir / f"{stem}.nii.gz"]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("NIfTI image or label file not found.")


def _rows_from_case_ids(case_ids: list[Any], data_root: str | Path | None) -> list[dict[str, Any]]:
    if data_root is None:
        raise ValueError("data_root is required for case IDs.")
    root = Path(data_root)
    rows = []
    for value in case_ids:
        case_id = int(value)
        rows.append({
            "sample_id": f"lits:volume-{case_id}",
            "dataset": "lits_official_labeled_train",
            "volume_id": f"volume-{case_id}",
            "case_id": case_id,
            "image_path": _resolve_case_file(root, "images", f"volume-{case_id}"),
            "label_path": _resolve_case_file(root, "labels", f"segmentation-{case_id}"),
        })
    return rows


def _assert_disjoint(parts: dict[str, list[dict[str, Any]]]) -> None:
    owner: dict[str, str] = {}
    for name, rows in parts.items():
        for row in rows:
            identifier = _row_id(row)
            previous = owner.get(identifier)
            if previous is not None:
                raise ValueError("Case ID occurs in multiple splits.")
            owner[identifier] = name


def _resolve_roles(payload: dict[str, Any], data_root: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None, list[dict[str, Any]] | None, str]:
    splits = payload.get("splits", {})
    if all(key in splits for key in ("development_case_ids", "outer_test_case_ids")):
        development = _rows_from_case_ids(list(splits["development_case_ids"]), data_root)
        outer = _rows_from_case_ids(list(splits["outer_test_case_ids"]), data_root)
        return development, outer, None, None, "id_only_manifest"
    if all(key in splits for key in ("internal_train", "internal_val", "outer_test")):
        return [], list(splits["outer_test"]), list(splits["internal_train"]), list(splits["internal_val"]), "explicit_nested_manifest"
    if all(key in splits for key in ("development", "outer_test")):
        return list(splits["development"]), list(splits["outer_test"]), None, None, "development_outer_manifest"
    if "train" in splits and "val" in splits:
        development = list(splits["train"])
        outer = list(splits["val"])
        fold_case_ids = {str(value) for value in payload.get("fold_case_ids", [])}
        outer_case_ids = {
            str(row.get("case_id", str(row.get("volume_id", "")).replace("volume-", "")))
            for row in outer
        }
        if fold_case_ids and fold_case_ids != outer_case_ids:
            raise ValueError("Outer-test cases do not match fold IDs.")
        return development, outer, None, None, "legacy_train_is_development_val_is_outer"
    raise ValueError("Unsupported split manifest format.")


def _remap_rows_to_data_root(rows: list[dict[str, Any]], data_root: str | Path | None) -> list[dict[str, Any]]:
    if data_root is None:
        return [dict(row) for row in rows]
    root = Path(data_root)
    result = []
    for row in rows:
        updated = dict(row)
        image_name = Path(str(row["image_path"])).name
        label_name = Path(str(row["label_path"])).name
        updated["image_path"] = str(root / "images" / image_name)
        updated["label_path"] = str(root / "labels" / label_name)
        result.append(updated)
    return result

def resolve_lits_splits(
    path: str | Path,
    *,
    seed: int,
    inner_val_count: int | None = None,
    inner_val_fraction: float | None = None,
    data_root: str | Path | None = None,
) -> ResolvedLiTSSplits:


    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    development, outer, explicit_train, explicit_val, role_mode = _resolve_roles(payload, data_root=data_root)
    if role_mode != "id_only_manifest":
        development = _remap_rows_to_data_root(development, data_root)
        outer = _remap_rows_to_data_root(outer, data_root)
    if explicit_train is not None:
        explicit_train = _remap_rows_to_data_root(explicit_train, data_root)
    if explicit_val is not None:
        explicit_val = _remap_rows_to_data_root(explicit_val, data_root)

    if explicit_train is not None and explicit_val is not None:
        internal_train, internal_val = explicit_train, explicit_val
        split_choice = "manifest_explicit"
    else:
        if inner_val_count is not None and inner_val_fraction is not None:
            raise ValueError("Specify a validation count or fraction, not both.")
        if inner_val_count is None and inner_val_fraction is None:
            raise ValueError("Internal validation size is required.")
        if len(development) < 2:
            raise ValueError("Development split is too small.")
        if inner_val_fraction is not None:
            fraction = float(inner_val_fraction)
            if not 0.0 < fraction < 1.0:
                raise ValueError("Validation fraction must be between 0 and 1.")
            count = int(round(len(development) * fraction))
            count = max(1, min(len(development) - 1, count))
            split_choice = f"generated_fraction_{fraction:g}"
        else:
            count = int(inner_val_count)
            if not 1 <= count < len(development):
                raise ValueError("Invalid validation case count.")
            split_choice = f"generated_count_{count}"

        ordered = sorted(development, key=_row_id)
        fold = int(payload.get("fold", 0))
        rng = random.Random(int(seed) + fold)
        rng.shuffle(ordered)
        internal_val = sorted(ordered[:count], key=_row_id)
        internal_train = sorted(ordered[count:], key=_row_id)

    parts = {
        "internal_train": internal_train,
        "internal_val": internal_val,
        "outer_test": outer,
    }
    _assert_disjoint(parts)
    union_count = sum(len(rows) for rows in parts.values())
    all_unique = len({_row_id(row) for rows in parts.values() for row in rows})
    if union_count != all_unique:
        raise AssertionError("Duplicate case IDs across splits.")

    audit = {
        "source_manifest": str(source.resolve()),
        "source_protocol": payload.get("protocol"),
        "data_root_remap": str(Path(data_root).resolve()) if data_root is not None else None,
        "fold": payload.get("fold"),
        "role_mode": role_mode,
        "inner_split_choice": split_choice,
        "inner_split_seed": int(seed) + int(payload.get("fold", 0)),
        "counts": {name: len(rows) for name, rows in parts.items()},
        "internal_train_ids": [_row_id(row) for row in internal_train],
        "internal_val_ids": [_row_id(row) for row in internal_val],
        "outer_test_ids": [_row_id(row) for row in outer],
        "outer_test_used_for_checkpoint_selection": False,
    }
    return ResolvedLiTSSplits(
        internal_train=_records(internal_train),
        internal_val=_records(internal_val),
        outer_test=_records(outer),
        audit=audit,
    )


def write_resolved_split(path: str | Path, resolved: ResolvedLiTSSplits) -> None:

    def rows(records: list[Any]) -> list[dict[str, Any]]:
        return [dict(getattr(record, "raw")) for record in records]

    payload = {
        "schema_version": 1,
        "protocol": "lits_nested_outer",
        "audit": resolved.audit,
        "splits": {
            "internal_train": rows(resolved.internal_train),
            "internal_val": rows(resolved.internal_val),
            "outer_test": rows(resolved.outer_test),
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_resolved_split(path: str | Path) -> ResolvedLiTSSplits:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload["splits"]
    parts = {
        "internal_train": list(splits["internal_train"]),
        "internal_val": list(splits["internal_val"]),
        "outer_test": list(splits["outer_test"]),
    }
    _assert_disjoint(parts)
    return ResolvedLiTSSplits(
        internal_train=_records(parts["internal_train"]),
        internal_val=_records(parts["internal_val"]),
        outer_test=_records(parts["outer_test"]),
        audit=dict(payload.get("audit", {})),
    )
