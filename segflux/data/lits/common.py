from __future__ import annotations

import random
import torch
import torch.nn.functional as F


def crop_patch_multiclass(
    x: torch.Tensor,
    y: torch.Tensor,
    patch: tuple[int, int, int],
    foreground_probability: float,
    tumor_probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:


    _, _, depth, height, width = x.shape
    pd, ph, pw = patch
    pad = (0, max(0, pw - width), 0, max(0, ph - height), 0, max(0, pd - depth))
    if any(pad):
        x, y = F.pad(x, pad), F.pad(y, pad)
    _, _, depth, height, width = x.shape
    foreground = torch.nonzero(y[0, 1:].amax(dim=0) > 0, as_tuple=False)
    tumor = torch.nonzero(y[0, 2] > 0, as_tuple=False)

    if foreground.numel() and random.random() < float(foreground_probability):
        if tumor.numel() and random.random() < float(tumor_probability):
            centre = tumor[random.randrange(len(tumor))].tolist()
        else:
            centre = foreground[random.randrange(len(foreground))].tolist()
    else:
        centre = [random.randrange(length) for length in (depth, height, width)]
    starts = [
        max(0, min(int(centre[i]) - patch[i] // 2, length - patch[i]))
        for i, length in enumerate((depth, height, width))
    ]
    z, yy, xx = starts
    return (
        x[:, :, z:z + pd, yy:yy + ph, xx:xx + pw],
        y[:, :, z:z + pd, yy:yy + ph, xx:xx + pw],
    )


def starts(length: int, patch: int, overlap: float) -> list[int]:
    if length <= patch:
        return [0]
    step = max(1, int(round(patch * (1.0 - overlap))))
    values = list(range(0, length - patch + 1, step))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values
