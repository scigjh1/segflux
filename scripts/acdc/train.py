from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from segflux.models.model import SegFlux, parameter_report


CLASSES = (1, 2, 3)
ACDC_SDF_STRIDES = ((1, 2, 2), (1, 2, 2), (2, 2, 2), (1, 2, 2))


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_ints(value: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3 or any(item < 1 for item in result):
        raise ValueError("Patch size must contain three positive integers.")
    return result


def parse_floats(value: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != 3 or any(item <= 0 for item in result):
        raise ValueError("Spacing must contain three positive values.")
    return result


def phase_paths(data_root: Path, patient_id: str) -> list[tuple[str, str, Path, Path]]:
    patient_dir = data_root / patient_id
    values: dict[str, str] = {}
    info = patient_dir / "Info.cfg"
    for line in info.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, item = line.split(":", 1)
            values[key.strip().lower()] = item.strip()
    records = []
    for phase_name, key in (("ED", "ed"), ("ES", "es")):
        frame = int(values[key])
        image = patient_dir / f"{patient_id}_frame{frame:02d}.nii.gz"
        target = patient_dir / f"{patient_id}_frame{frame:02d}_gt.nii.gz"
        if not image.exists() or not target.exists():
            raise FileNotFoundError("ACDC image or label file not found.")
        records.append((patient_id, phase_name, image, target))
    return records


def _resample(tensor: torch.Tensor, shape: tuple[int, int, int], mode: str) -> torch.Tensor:
    kwargs = {"size": shape, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = False
    return F.interpolate(tensor[None, None], **kwargs)[0, 0]


def normalize_image(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, (1, 99))
    clipped = np.clip(image, lo, hi).astype(np.float32, copy=False)
    return (clipped - clipped.mean()) / (clipped.std() + 1e-6)


def signed_foreground_distance(label: np.ndarray, spacing: tuple[float, float, float], truncation_mm: float) -> np.ndarray:
    foreground = label > 0
    inside = distance_transform_edt(foreground, sampling=spacing)
    outside = distance_transform_edt(~foreground, sampling=spacing)
    return np.clip((inside - outside) / truncation_mm, -1.0, 1.0).astype(np.float32)


class ACDCVolumeCache:


    def __init__(
        self,
        data_root: Path,
        patient_ids: list[str],
        target_spacing: tuple[float, float, float],
        cache_cases: int,
        sdf_truncation_mm: float,
        disk_cache_dir: Path | None = None,
        training_patch: tuple[int, int, int] | None = None,
    ) -> None:
        self.data_root = data_root
        self.target_spacing = target_spacing
        self.cache_cases = int(cache_cases)
        self.sdf_truncation_mm = float(sdf_truncation_mm)
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.training_patch = training_patch
        if self.disk_cache_dir:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.records = [item for patient_id in patient_ids for item in phase_paths(data_root, patient_id)]
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, index: int) -> dict:
        patient_id, phase, image_path, label_path = self.records[index]
        key = f"{patient_id}:{phase}"
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        cache_path = self.disk_cache_dir / f"{patient_id}_{phase}.pt" if self.disk_cache_dir else None
        if cache_path and cache_path.exists():
            item = torch.load(cache_path, map_location="cpu")
            self._attach_training_view(item)
            self.cache[key] = item
            self.cache.move_to_end(key)
            while len(self.cache) > self.cache_cases:
                self.cache.popitem(last=False)
            return item
        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))
        image_xyz = np.asarray(image_nii.dataobj, dtype=np.float32)
        label_xyz = np.asarray(label_nii.dataobj, dtype=np.int16)
        if image_xyz.shape != label_xyz.shape:
            raise RuntimeError("Image and label shapes differ.")
        values = set(np.unique(label_xyz).tolist())
        if not values.issubset({0, 1, 2, 3}):
            raise ValueError("Unexpected ACDC label value.")
        spacing_xyz = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        original_shape = tuple(int(value) for value in image_xyz.shape[::-1])
        target_shape = tuple(max(1, int(round(size * source / target))) for size, source, target in zip(original_shape, spacing_zyx, self.target_spacing))
        image = _resample(torch.from_numpy(normalize_image(image_xyz).transpose(2, 1, 0).copy()), target_shape, "trilinear")
        label = _resample(torch.from_numpy(label_xyz.transpose(2, 1, 0).copy().astype(np.float32)), target_shape, "nearest").long()
        item = {
            "case_id": key,
            "patient_id": patient_id,
            "phase": phase,
            "image": image,
            "label": label,
            "sdf": torch.from_numpy(signed_foreground_distance(label.numpy(), self.target_spacing, self.sdf_truncation_mm)),
            "original_label": label_xyz.transpose(2, 1, 0).copy(),
            "original_shape": original_shape,
            "original_spacing_zyx": spacing_zyx,
        }
        if cache_path:
            temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
            torch.save(item, temporary)
            os.replace(temporary, cache_path)
        self._attach_training_view(item)
        self.cache[key] = item
        self.cache.move_to_end(key)
        while len(self.cache) > self.cache_cases:
            self.cache.popitem(last=False)
        return item

    def _attach_training_view(self, item: dict) -> None:

        if self.training_patch is None or "training_foreground" in item:
            return
        for name in ("image", "label", "sdf"):
            item[f"training_{name}"] = pad_to(item[name], self.training_patch)
        item["training_foreground"] = torch.nonzero(item["training_label"] > 0, as_tuple=False).to(torch.int32)

    def prepare_all(self) -> None:
        for index in range(len(self.records)):
            self.get(index)
            if (index + 1) % 20 == 0 or index + 1 == len(self.records):
                print(json.dumps({"prepared": index + 1, "total": len(self.records)}), flush=True)


def pad_to(tensor: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    depth, height, width = tensor.shape[-3:]
    return F.pad(tensor, (0, max(0, shape[2] - width), 0, max(0, shape[1] - height), 0, max(0, shape[0] - depth)))


def random_crop(item: dict, patch: tuple[int, int, int], foreground_probability: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    image, label, sdf = (item[f"training_{name}"] for name in ("image", "label", "sdf"))
    foreground = item["training_foreground"]
    if len(foreground) and random.random() < foreground_probability:
        centre = foreground[random.randrange(len(foreground))].tolist()
    else:
        centre = [random.randrange(length) for length in image.shape]
    starts = [max(0, min(int(centre[i]) - patch[i] // 2, image.shape[i] - patch[i])) for i in range(3)]
    z, y, x = starts
    image, label, sdf = (value[z:z + patch[0], y:y + patch[1], x:x + patch[2]] for value in (image, label, sdf))
    return image[None, None], label[None], sdf[None, None]


def random_training_batch(
    cache: ACDCVolumeCache,
    patch: tuple[int, int, int],
    foreground_probability: float,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    images, labels, sdfs = [], [], []
    for _ in range(int(batch_size)):
        item = cache.get(random.randrange(len(cache.records)))
        image, label, sdf = random_crop(item, patch, foreground_probability)
        image, label, sdf = image.to(device), label.to(device), sdf.to(device)
        image, label, sdf = augment_patch(image, label, sdf)
        images.append(image); labels.append(label); sdfs.append(sdf)
    return torch.cat(images, 0), torch.cat(labels, 0), torch.cat(sdfs, 0)

def augment_patch(image: torch.Tensor, label: torch.Tensor, sdf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if random.random() < 0.5:
        image, label, sdf = (torch.flip(value, (-1,)) for value in (image, label, sdf))
    if random.random() < 0.5:
        image, label, sdf = (torch.flip(value, (-2,)) for value in (image, label, sdf))
    if random.random() < 0.20:
        image = image + 0.03 * torch.randn_like(image)
    if random.random() < 0.35:
        image = image * random.uniform(0.9, 1.1) + random.uniform(-0.1, 0.1)
    return image, label, sdf


def build_controlled_model(name: str, patch: tuple[int, int, int]) -> nn.Module:
    if name != "segflux_light":
        raise ValueError("Unsupported ACDC model.")
    return SegFlux(num_classes=4, model_size="light", downsample_strides=ACDC_SDF_STRIDES)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, 1)
    one_hot = F.one_hot(target, 4).movedim(-1, 1).float()
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum((0, 2, 3, 4))
    total = (probabilities[:, 1:] + one_hot[:, 1:]).sum((0, 2, 3, 4))
    return 1.0 - ((2.0 * intersection + 1e-5) / (total + 1e-5)).mean()


def forward_logits(model: nn.Module, name: str, image: torch.Tensor, spacing: torch.Tensor):
    output = model(image, spacing) if name == "segflux_light" else model(image)
    return output, output.local_logits


def loss_for(model: nn.Module, name: str, image: torch.Tensor, target: torch.Tensor, sdf: torch.Tensor, spacing: torch.Tensor):
    output, logits = forward_logits(model, name, image, spacing)
    loss = F.cross_entropy(logits, target) + soft_dice_loss(logits, target)
    details = {"segmentation": float(loss.detach())}
    if name == "segflux_light":
        direct_sdf = F.smooth_l1_loss(output.direct.sdf, sdf)
        flux_loss = torch.zeros((), device=image.device)
        for scale in output.scales:
            target_scale = F.interpolate(sdf, size=scale.shape_sdf.shape[2:], mode="trilinear", align_corners=False)
            flux_loss = flux_loss + F.smooth_l1_loss(scale.shape_sdf, target_scale)
            flux_loss = flux_loss + F.smooth_l1_loss(scale.routed_consensus, target_scale)
        loss = loss + 0.10 * direct_sdf + 0.05 * flux_loss
        details.update({"direct_sdf": float(direct_sdf.detach()), "pmf": float(flux_loss.detach())})
    return loss, details


def starts(length: int, patch: int) -> list[int]:
    if length <= patch:
        return [0]
    step = max(1, patch // 2)
    result = list(range(0, length - patch + 1, step))
    if result[-1] != length - patch:
        result.append(length - patch)
    return result


@torch.no_grad()
def predict_volume(model: nn.Module, name: str, image: torch.Tensor, patch: tuple[int, int, int], spacing: torch.Tensor, device: torch.device) -> torch.Tensor:
    source_shape = tuple(image.shape)
    image = pad_to(image, patch)
    shape = tuple(image.shape)
    accumulator = torch.zeros((4, *shape), dtype=torch.float32)
    counter = torch.zeros((1, *shape), dtype=torch.float32)
    model.eval()
    for z in starts(shape[0], patch[0]):
        for y in starts(shape[1], patch[1]):
            for x in starts(shape[2], patch[2]):
                window = image[z:z + patch[0], y:y + patch[1], x:x + patch[2]][None, None].to(device)
                _, logits = forward_logits(model, name, window, spacing)
                accumulator[:, z:z + patch[0], y:y + patch[1], x:x + patch[2]] += torch.softmax(logits, 1)[0].float().cpu()
                counter[:, z:z + patch[0], y:y + patch[1], x:x + patch[2]] += 1.0
    return (accumulator / counter.clamp_min(1.0))[:, :source_shape[0], :source_shape[1], :source_shape[2]].argmax(0)


def dice(prediction: np.ndarray, target: np.ndarray, label: int) -> float:
    p, t = prediction == label, target == label
    if not p.any() and not t.any():
        return 1.0
    return float((2.0 * (p & t).sum() + 1e-8) / (p.sum() + t.sum() + 1e-8))


@torch.no_grad()
def validate(model: nn.Module, name: str, cache: ACDCVolumeCache, patch: tuple[int, int, int], spacing: torch.Tensor, device: torch.device) -> dict:
    rows = []
    for index in range(len(cache.records)):
        item = cache.get(index)
        prediction = predict_volume(model, name, item["image"], patch, spacing, device).float()
        native = F.interpolate(prediction[None, None], size=item["original_shape"], mode="nearest")[0, 0].numpy().astype(np.int16)
        row = {"case_id": item["case_id"], "patient_id": item["patient_id"], "phase": item["phase"]}
        row.update({str(label): dice(native, item["original_label"], label) for label in CLASSES})
        rows.append(row)
    result = {"per_case": rows}
    for label, structure in ((1, "rv"), (2, "myo"), (3, "lv")):
        result[structure] = {phase.lower(): float(np.mean([row[str(label)] for row in rows if row["phase"] == phase])) for phase in ("ED", "ES")}
        result[structure]["mean"] = float(np.mean([row[str(label)] for row in rows]))
    result["macro_dice"] = float(np.mean([row[str(label)] for row in rows for label in CLASSES]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["segflux_light"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-size", default="32,256,224")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--target-spacing", default="5.0,1.5625,1.5625")
    parser.add_argument("--max-updates", type=int, default=60000)
    parser.add_argument("--val-interval", type=int, default=3000)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--warmup-updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--foreground-oversampling", type=float, default=0.66)
    parser.add_argument("--sdf-truncation-mm", type=float, default=20.0)
    parser.add_argument("--cache-cases", type=int, default=160)
    parser.add_argument("--disk-cache-dir")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--preload-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    if args.max_updates != 60000 or args.val_interval != 3000:
        raise ValueError("Expected 60000 updates and 3000-step validation.")
    if args.batch_size != 2:
        raise ValueError("ACDC batch size must be 2.")
    seed_all(args.seed)
    torch.set_num_threads(max(1, args.cpu_threads))
    torch.set_num_interop_threads(1)
    patch, target_spacing = parse_ints(args.patch_size), parse_floats(args.target_spacing)
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    disk_cache = Path(args.disk_cache_dir) if args.disk_cache_dir else None
    train_cache = ACDCVolumeCache(Path(args.data_root), protocol["train"], target_spacing, args.cache_cases, args.sdf_truncation_mm, disk_cache, patch)
    val_cache = ACDCVolumeCache(Path(args.data_root), protocol["val"], target_spacing, args.cache_cases, args.sdf_truncation_mm, disk_cache)
    if len(train_cache.records) != 140 or len(val_cache.records) != 20:
        raise RuntimeError("ACDC phase counts do not match the protocol.")
    if args.prepare_only:
        if disk_cache is None:
            raise ValueError("--prepare-only requires --disk-cache-dir.")
        train_cache.prepare_all()
        val_cache.prepare_all()
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "cache_complete.json").write_text(
            json.dumps({"train_phases": 140, "validation_phases": 20, "disk_cache_dir": str(disk_cache)}, indent=2) + "\n",
            encoding="utf-8",
        )
        return
    if args.preload_memory:
        train_cache.prepare_all()
        val_cache.prepare_all()
    device = torch.device(args.device)
    model = build_controlled_model(args.model, patch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler") and device.type == "cuda"
        else torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "patch_size": patch, "target_spacing_zyx": target_spacing, "train_phases": len(train_cache.records), "validation_phases": len(val_cache.records), "heldout_test_used_for_checkpoint_selection": False, "heldout_test_used_for_hyperparameter_tuning": False, "selection_metric": "native-space RV/MYO/LV macro Dice across ED and ES", "parameter_report": parameter_report(model), "segflux_downsample_strides": ACDC_SDF_STRIDES if args.model == "segflux_light" else None, "batch_semantics": "true batch of 2 independent 3-D patches"}
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    update, best, history = 0, -math.inf, []
    if args.resume_from:
        state = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state.get("scaler", {}))
        update, best, history = int(state["updates"]), float(state.get("best", -math.inf)), state.get("history", [])
    spacing = torch.tensor(target_spacing, dtype=torch.float32, device=device)[None]
    model.train()
    started, losses = time.time(), []
    validated_updates = {int(row["updates"]) for row in history if "updates" in row}
    if update > 0 and update % args.val_interval == 0 and update not in validated_updates:
        metrics = validate(model, args.model, val_cache, patch, spacing, device)
        row = {
            "updates": update,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": None,
            "loss_parts": {},
            "metrics": metrics,
            "elapsed_seconds": time.time() - started,
            "resumed_validation": True,
        }
        history.append(row)
        if metrics["macro_dice"] > best:
            best = metrics["macro_dice"]
            state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "updates": update, "best": best, "history": history, "config": config}
            torch.save(state, output / "best_validation.pt")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "updates": update, "best": best, "history": history, "config": config}
        torch.save(state, output / "checkpoint_last.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        model.train()
        print(json.dumps({"update": update, "macro_dice": metrics["macro_dice"], "best": best, "resumed_validation": True}), flush=True)
    while update < args.max_updates:
        image, target, sdf = random_training_batch(
            train_cache, patch, args.foreground_oversampling, args.batch_size, device
        )
        progress = min(1.0, float(update + 1) / float(max(1, args.warmup_updates)))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * progress
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, parts = loss_for(model, args.model, image, target, sdf, spacing.expand(image.shape[0], -1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(optimizer)
        scaler.update()
        update += 1
        losses.append(float(loss.detach()))
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "updates": update, "best": best, "history": history, "config": config}
        if update % args.checkpoint_interval == 0:
            torch.save(state, output / "checkpoint_last.pt")
        if update % args.val_interval == 0 or update == args.max_updates:
            metrics = validate(model, args.model, val_cache, patch, spacing, device)
            row = {"updates": update, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": float(np.mean(losses[-args.val_interval:])), "loss_parts": parts, "metrics": metrics, "elapsed_seconds": time.time() - started}
            history.append(row)
            state["history"] = history
            if metrics["macro_dice"] > best:
                best = metrics["macro_dice"]
                state["best"] = best
                torch.save(state, output / "best_validation.pt")
            torch.save(state, output / "checkpoint_last.pt")
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
            model.train()
            print(json.dumps({"update": update, "macro_dice": metrics["macro_dice"], "best": best}), flush=True)
    (output / "complete.json").write_text(json.dumps({"updates": update, "best_validation_macro_dice": best}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
