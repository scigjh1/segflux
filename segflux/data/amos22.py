from __future__ import annotations
import random
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

_VOLUME_CACHE_ENABLED = False
_VOLUME_CACHE = {}

def set_volume_cache(enabled: bool) -> None:

    global _VOLUME_CACHE_ENABLED
    _VOLUME_CACHE_ENABLED = bool(enabled)

def _binary_foreground(array: np.ndarray, foreground_label_value: int) -> np.ndarray:
    if foreground_label_value > 0:
        return (array == foreground_label_value).astype(np.float32)
    return (array > 0).astype(np.float32)

def load_native_xy_cpu(rec, xy_size: int, foreground_label_value: int = 0):
    cache_key = (
        str(rec.image_path),
        str(rec.label_path),
        int(xy_size),
        int(foreground_label_value),
    )
    if _VOLUME_CACHE_ENABLED and cache_key in _VOLUME_CACHE:
        return _VOLUME_CACHE[cache_key]
    image = nib.load(rec.image_path)
    label = nib.load(rec.label_path)
    arr = np.asarray(image.dataobj, np.float32)
    lab = _binary_foreground(np.asarray(label.dataobj), foreground_label_value)
    spacing_xyz = np.asarray(image.header.get_zooms()[:3], np.float32)


    x = torch.from_numpy((np.clip(arr, -125, 275) + 125.0) / 400.0).permute(2, 1, 0)[None, None]
    y = torch.from_numpy(lab).permute(2, 1, 0)[None, None]
    native_zyx = tuple(int(v) for v in x.shape[2:])
    spacing_zyx = np.asarray((spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]), np.float32)

    if xy_size > 0 and native_zyx[1:] != (xy_size, xy_size):
        spacing_zyx[1] *= native_zyx[1] / xy_size
        spacing_zyx[2] *= native_zyx[2] / xy_size
        x = F.interpolate(x, size=(native_zyx[0], xy_size, xy_size), mode="trilinear", align_corners=False)
        y = F.interpolate(y, size=(native_zyx[0], xy_size, xy_size), mode="nearest")
    result = (
        x,
        y,
        lab.transpose(2, 1, 0).astype(bool),
        spacing_xyz,
        spacing_zyx,
        native_zyx,
    )
    if _VOLUME_CACHE_ENABLED:
        _VOLUME_CACHE[cache_key] = result
    return result

def crop_patch(x: torch.Tensor, y: torch.Tensor, patch: tuple[int, int, int], foreground_probability: float):
    _, _, d, h, w = x.shape
    pd, ph, pw = patch
    pad = (0, max(0, pw - w), 0, max(0, ph - h), 0, max(0, pd - d))
    if any(pad):
        x = F.pad(x, pad)
        y = F.pad(y, pad)
    _, _, d, h, w = x.shape
    foreground = torch.nonzero(y[0, 0] > 0.5, as_tuple=False)
    if foreground.numel() and random.random() < foreground_probability:
        centre = foreground[random.randrange(foreground.shape[0])].tolist()
    else:
        centre = [random.randrange(d), random.randrange(h), random.randrange(w)]
    start = [max(0, min(int(centre[i]) - patch[i] // 2, (d, h, w)[i] - patch[i])) for i in range(3)]
    z, yy, xx = start
    return x[:, :, z:z + pd, yy:yy + ph, xx:xx + pw], y[:, :, z:z + pd, yy:yy + ph, xx:xx + pw]

def make_truncated_sdf(target: torch.Tensor, spacing_zyx: np.ndarray, truncation_mm: float) -> torch.Tensor:

    mask = target[0, 0].numpy() > 0.5
    if not mask.any():
        sdf = np.full(mask.shape, -1.0, np.float32)
    else:
        inside = ndimage.distance_transform_edt(mask, sampling=spacing_zyx)
        outside = ndimage.distance_transform_edt(~mask, sampling=spacing_zyx)
        sdf = np.clip((inside - outside) / truncation_mm, -1.0, 1.0).astype(np.float32)
    return torch.from_numpy(sdf)[None, None]

def segmentation_loss(logit: torch.Tensor, target: torch.Tensor, bce_weight: float) -> torch.Tensor:
    target = F.interpolate(target, size=logit.shape[2:], mode="nearest")
    prob = torch.sigmoid(logit)
    intersection = (prob * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (prob.sum() + target.sum() + 1.0)
    return dice + bce_weight * F.binary_cross_entropy_with_logits(logit.float(), target.float())

def _starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    step = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, length - patch + 1, step))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts

def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())
