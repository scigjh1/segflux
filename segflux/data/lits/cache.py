from __future__ import annotations

import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch


CACHE_VERSION = "lits_zyx_float16_uint8_v1"


def _case_key(rec: Any) -> str:
    value = str(getattr(rec, "volume_id"))
    return "case_" + "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)


def _atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prepare_case(rec: Any, cache_dir: str | Path, force: bool = False) -> dict[str, Any]:

    root = Path(cache_dir)
    case_dir = root / _case_key(rec)
    image_path = case_dir / "image_zyx_f16.npy"
    label_path = case_dir / "label_zyx_u8.npy"
    metadata_path = case_dir / "metadata.json"
    if not force and image_path.exists() and label_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_version") == CACHE_VERSION:
            return {"case_id": str(getattr(rec, "volume_id")), "status": "cached", **metadata}

    case_dir.mkdir(parents=True, exist_ok=True)
    image = nib.load(str(rec.image_path))
    label = nib.load(str(rec.label_path))
    image_xyz = np.asarray(image.dataobj, dtype=np.float32)
    label_xyz = np.asarray(label.dataobj, dtype=np.int16)
    labels = set(np.unique(label_xyz).tolist())
    if not labels.issubset({0, 1, 2}):
        raise ValueError("LiTS labels must be in {0, 1, 2}.")


    image_zyx = np.ascontiguousarray(
        ((np.clip(image_xyz, -125.0, 275.0) + 125.0) / 400.0).transpose(2, 1, 0),
        dtype=np.float16,
    )
    label_zyx = np.ascontiguousarray(label_xyz.transpose(2, 1, 0), dtype=np.uint8)
    zooms_xyz = tuple(float(value) for value in image.header.get_zooms()[:3])
    metadata = {
        "cache_version": CACHE_VERSION,
        "case_id": str(getattr(rec, "volume_id")),
        "shape_zyx": list(image_zyx.shape),
        "spacing_zyx_mm": [zooms_xyz[2], zooms_xyz[1], zooms_xyz[0]],
        "labels": sorted(int(value) for value in labels),
        "source_image_size": int(Path(rec.image_path).stat().st_size),
        "source_label_size": int(Path(rec.label_path).stat().st_size),
    }
    _atomic_save(image_path, image_zyx)
    _atomic_save(label_path, label_zyx)
    _atomic_json(metadata_path, metadata)
    return {"case_id": metadata["case_id"], "status": "created", **metadata}


def prepare_records(records: Iterable[Any], cache_dir: str | Path, force: bool = False) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for record in records:
        key = _case_key(record)
        if key not in seen:
            seen.add(key)
            result.append(prepare_case(record, cache_dir, force=force))
    return result


@dataclass
class _Entry:
    image: np.ndarray
    label: np.ndarray
    spacing_zyx_mm: tuple[float, float, float]
    foreground_coords: np.ndarray | None = None
    tumor_coords: np.ndarray | None = None


class LiTSCaseCache:


    def __init__(self, cache_dir: str | Path, max_cases: int = 12):
        self.root = Path(cache_dir)
        self.max_cases = max(1, int(max_cases))
        self.entries: OrderedDict[str, _Entry] = OrderedDict()

    def _case_dir(self, rec: Any) -> Path:
        return self.root / _case_key(rec)

    def _entry(self, rec: Any) -> _Entry:
        key = _case_key(rec)
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.entries[key] = entry
            return entry
        case_dir = self._case_dir(rec)
        metadata_path = case_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError("Cache metadata not found.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_version") != CACHE_VERSION:
            raise RuntimeError("Cache version mismatch.")
        entry = _Entry(
            image=np.load(case_dir / "image_zyx_f16.npy", mmap_mode="r"),
            label=np.load(case_dir / "label_zyx_u8.npy", mmap_mode="r"),
            spacing_zyx_mm=tuple(float(v) for v in metadata["spacing_zyx_mm"]),
        )
        self.entries[key] = entry
        while len(self.entries) > self.max_cases:
            self.entries.popitem(last=False)
        return entry

    @staticmethod
    def _coordinates(entry: _Entry, tumor: bool) -> np.ndarray:
        attribute = "tumor_coords" if tumor else "foreground_coords"
        value = getattr(entry, attribute)
        if value is None:
            if tumor:
                mask = entry.label == 2
            else:
                mask = entry.label > 0
            value = np.argwhere(mask)
            setattr(entry, attribute, value)
        return value

    @staticmethod
    def _extract_patch(array: np.ndarray, starts: tuple[int, int, int], patch: tuple[int, int, int]) -> np.ndarray:
        output = np.zeros(patch, dtype=array.dtype)
        source_slices = []
        destination_slices = []
        for start, size, limit in zip(starts, patch, array.shape):
            source_start = max(0, start)
            source_end = min(limit, start + size)
            destination_start = max(0, -start)
            destination_end = destination_start + max(0, source_end - source_start)
            source_slices.append(slice(source_start, source_end))
            destination_slices.append(slice(destination_start, destination_end))
        output[tuple(destination_slices)] = array[tuple(source_slices)]
        return output

    def sample_patch(
        self,
        rec: Any,
        patch: tuple[int, int, int],
        foreground_probability: float,
        tumor_probability: float,
        sampling_mode: str = "legacy",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:


        if sampling_mode not in {"legacy", "corrected_v1"}:
            raise ValueError("Unsupported sampling mode.")
        entry = self._entry(rec)
        shape = tuple(int(v) for v in entry.label.shape)
        tumor_coords = self._coordinates(entry, tumor=True)
        foreground_coords = self._coordinates(entry, tumor=False)
        if sampling_mode == "corrected_v1":
            draw = random.random()
            tumor_cut = foreground_probability * tumor_probability
            if len(tumor_coords) and draw < tumor_cut:
                centre = tumor_coords[random.randrange(len(tumor_coords))]
            elif len(foreground_coords) and draw < foreground_probability:
                centre = foreground_coords[random.randrange(len(foreground_coords))]
            else:
                centre = np.asarray([random.randrange(max(1, length)) for length in shape], dtype=np.int64)
        else:
            if len(tumor_coords) and random.random() < tumor_probability:
                centre = tumor_coords[random.randrange(len(tumor_coords))]
            elif len(foreground_coords) and random.random() < foreground_probability:
                centre = foreground_coords[random.randrange(len(foreground_coords))]
            else:
                centre = np.asarray([random.randrange(max(1, length)) for length in shape], dtype=np.int64)
        starts = tuple(
            min(max(int(centre[index]) - patch[index] // 2, 0), max(0, shape[index] - patch[index]))
            for index in range(3)
        )
        image_patch = self._extract_patch(entry.image, starts, patch).astype(np.float32, copy=False)
        label_patch = self._extract_patch(entry.label, starts, patch)


        target = np.eye(3, dtype=np.float32)[label_patch].transpose(3, 0, 1, 2)
        x = torch.from_numpy(np.ascontiguousarray(image_patch))[None, None]
        y = torch.from_numpy(np.ascontiguousarray(target))[None]
        spacing = torch.tensor([entry.spacing_zyx_mm], dtype=torch.float32)
        return x, y, spacing

    def full_case(self, rec: Any) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        entry = self._entry(rec)
        image = np.ascontiguousarray(np.asarray(entry.image, dtype=np.float32))
        target = np.eye(3, dtype=np.float32)[np.asarray(entry.label, dtype=np.uint8)].transpose(3, 0, 1, 2)
        return (
            torch.from_numpy(image)[None, None],
            torch.from_numpy(np.ascontiguousarray(target))[None],
            entry.spacing_zyx_mm,
        )
