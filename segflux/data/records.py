from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VolumeRecord:
    volume_id: str
    image_path: str
    label_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def discover_msd_records(dataset_root: str | Path) -> list[VolumeRecord]:
    root = Path(dataset_root)
    dataset_json = root / "dataset.json"
    records: list[VolumeRecord] = []

    if dataset_json.exists():
        with dataset_json.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        for item in payload.get("training", []):
            image_path = _resolve_msd_path(root, item["image"])
            label_path = _resolve_msd_path(root, item["label"])
            records.append(
                VolumeRecord(
                    volume_id=_volume_id(image_path),
                    image_path=str(image_path),
                    label_path=str(label_path),
                )
            )
    else:
        image_dir = root / "imagesTr"
        label_dir = root / "labelsTr"
        for image_path in sorted(image_dir.glob("*.nii*")):
            label_path = label_dir / image_path.name
            if label_path.exists():
                records.append(
                    VolumeRecord(
                        volume_id=_volume_id(image_path),
                        image_path=str(image_path),
                        label_path=str(label_path),
                    )
                )

    if not records:
        raise RuntimeError("No records found.")
    duplicate_ids = _duplicates(record.volume_id for record in records)
    if duplicate_ids:
        raise RuntimeError("Duplicate volume IDs.")
    return records


def discover_kits23_records(dataset_root: str | Path) -> list[VolumeRecord]:


    root = Path(dataset_root)
    records = []
    missing_images = 0
    missing_labels = 0
    for case_dir in sorted(root.glob("case_*")):
        if not case_dir.is_dir():
            continue
        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"
        if not image_path.exists():
            missing_images += 1
            continue
        if not label_path.exists():
            missing_labels += 1
            continue
        records.append(
            VolumeRecord(
                volume_id=case_dir.name,
                image_path=str(image_path.resolve()),
                label_path=str(label_path.resolve()),
            )
        )
    if not records:
        raise RuntimeError("No records found.")
    return records


def discover_amos22_records(
    dataset_root: str | Path,
    modality: str,
) -> list[VolumeRecord]:


    root = Path(dataset_root)
    modality_name = str(modality).lower()
    if modality_name not in {"ct", "mri"}:
        raise ValueError("Modality must be CT or MRI.")
    records = []
    for partition in ("Tr", "Va"):
        image_dir = root / f"images{partition}"
        label_dir = root / f"labels{partition}"
        for image_path in sorted(image_dir.glob("amos_*.nii.gz")):
            raw_id = image_path.name[len("amos_") : -len(".nii.gz")]
            try:
                numeric_id = int(raw_id)
            except ValueError:
                continue
            is_mri = numeric_id >= 500
            if (modality_name == "mri") != is_mri:
                continue
            label_path = label_dir / image_path.name
            if not label_path.exists():
                continue
            records.append(
                VolumeRecord(
                    volume_id=f"amos_{numeric_id:04d}",
                    image_path=str(image_path.resolve()),
                    label_path=str(label_path.resolve()),
                )
            )
    if not records:
        raise RuntimeError("No records found.")
    duplicate_ids = _duplicates(record.volume_id for record in records)
    if duplicate_ids:
        raise RuntimeError("Duplicate volume IDs.")
    return records


def discover_records(
    dataset_root: str | Path,
    layout: str = "msd",
) -> list[VolumeRecord]:
    if layout in {"msd", "flat"}:
        return discover_msd_records(dataset_root)
    if layout == "kits23":
        return discover_kits23_records(dataset_root)
    if layout == "amos22_ct":
        return discover_amos22_records(dataset_root, "ct")
    if layout == "amos22_mri":
        return discover_amos22_records(dataset_root, "mri")
    raise ValueError("Unsupported dataset layout.")


def _resolve_msd_path(root: Path, raw: str) -> Path:
    normalized = raw.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return (root / normalized).resolve()


def _volume_id(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _duplicates(values) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
