from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage


def binary_volume_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: Iterable[float] = (1.0, 1.0, 1.0),
    surface_tolerance_mm: float = 1.0,
) -> dict[str, float]:


    predicted = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if predicted.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ.")
    if predicted.ndim != 3:
        raise ValueError("Expected a 3D volume.")

    spacing = tuple(float(value) for value in spacing_mm)
    if len(spacing) != predicted.ndim or any(value <= 0 for value in spacing):
        raise ValueError("Invalid voxel spacing.")
    if surface_tolerance_mm < 0:
        raise ValueError("Surface tolerance must be nonnegative.")

    predicted_count = int(predicted.sum())
    truth_count = int(truth.sum())
    intersection = int(np.logical_and(predicted, truth).sum())
    denominator = predicted_count + truth_count
    union = int(np.logical_or(predicted, truth).sum())

    if denominator == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "hd95_mm": 0.0,
            "nsd": 1.0,
            "predicted_voxels": 0.0,
            "target_voxels": 0.0,
        }
    dice = 2.0 * intersection / denominator
    iou = intersection / union if union else 1.0
    if predicted_count == 0 or truth_count == 0:
        return {
            "dice": float(dice),
            "iou": float(iou),
            "hd95_mm": float("inf"),
            "nsd": 0.0,
            "predicted_voxels": float(predicted_count),
            "target_voxels": float(truth_count),
        }

    predicted_surface = _surface(predicted)
    truth_surface = _surface(truth)
    distance_to_truth = ndimage.distance_transform_edt(
        ~truth_surface,
        sampling=spacing,
    )[predicted_surface]
    distance_to_prediction = ndimage.distance_transform_edt(
        ~predicted_surface,
        sampling=spacing,
    )[truth_surface]
    symmetric_distances = np.concatenate(
        (distance_to_truth, distance_to_prediction)
    )
    hd95 = float(np.percentile(symmetric_distances, 95))
    within_tolerance = (
        np.count_nonzero(distance_to_truth <= surface_tolerance_mm)
        + np.count_nonzero(distance_to_prediction <= surface_tolerance_mm)
    )
    nsd = within_tolerance / max(symmetric_distances.size, 1)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "hd95_mm": hd95,
        "nsd": float(nsd),
        "predicted_voxels": float(predicted_count),
        "target_voxels": float(truth_count),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_and(mask, ~eroded)
