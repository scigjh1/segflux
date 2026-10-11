from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
import torch
import torch.nn.functional as F


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool))


def _surface_distances(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]):
    pred_surface = _surface(pred)
    truth_surface = _surface(truth)
    if not pred_surface.any() or not truth_surface.any():
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    pred_to_truth = ndimage.distance_transform_edt(~truth_surface, sampling=spacing)[pred_surface]
    truth_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[truth_surface]
    return pred_to_truth.astype(np.float64), truth_to_pred.astype(np.float64)


def binary_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    spacing: tuple[float, float, float],
    surface_tolerance_mm: float = 3.0,
) -> dict[str, float]:

    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = float(np.logical_and(prediction, truth).sum())
    pred_count = float(prediction.sum())
    truth_count = float(truth.sum())
    dice = _safe_div(2.0 * intersection, pred_count + truth_count)
    precision = _safe_div(intersection, pred_count)
    recall = _safe_div(intersection, truth_count)
    pred_to_truth, truth_to_pred = _surface_distances(prediction, truth, spacing)
    distances = np.concatenate([pred_to_truth, truth_to_pred])
    if distances.size:
        hd95 = float(np.percentile(distances, 95.0))
        nsd = float(np.mean(distances <= float(surface_tolerance_mm)))
        assd = float(np.mean(distances))
    else:
        hd95 = float("nan")
        nsd = float("nan")
        assd = float("nan")
    return {
        "dice": dice,
        "hd95_mm": hd95,
        "nsd": nsd,
        "assd_mm": assd,
        "precision": precision,
        "recall": recall,
        "predicted_voxels": pred_count,
        "target_voxels": truth_count,
    }


def lesion_detection_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    pred_labels, pred_count = ndimage.label(np.asarray(prediction, dtype=bool), structure=structure)
    truth_labels, truth_count = ndimage.label(np.asarray(truth, dtype=bool), structure=structure)
    if truth_count:
        detected_truth = sum(
            bool(np.any(pred_labels[truth_labels == index] > 0))
            for index in range(1, truth_count + 1)
        )
        lesion_recall = float(detected_truth / truth_count)
    else:
        detected_truth = 0
        lesion_recall = float("nan")
    if pred_count:
        matched_pred = sum(
            bool(np.any(truth_labels[pred_labels == index] > 0))
            for index in range(1, pred_count + 1)
        )
        lesion_precision = float(matched_pred / pred_count)
    else:
        matched_pred = 0
        lesion_precision = float("nan")
    if np.isfinite(lesion_precision) and np.isfinite(lesion_recall) and lesion_precision + lesion_recall:
        lesion_f1 = float(2.0 * lesion_precision * lesion_recall / (lesion_precision + lesion_recall))
    else:
        lesion_f1 = float("nan")
    return {
        "gt_lesions": float(truth_count),
        "pred_lesions": float(pred_count),
        "detected_gt_lesions": float(detected_truth),
        "matched_pred_lesions": float(matched_pred),
        "lesion_recall": lesion_recall,
        "lesion_precision": lesion_precision,
        "lesion_f1": lesion_f1,
        "false_positive_lesions": float(max(0, pred_count - matched_pred)),
    }


def multiclass_lits_metrics(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    spacing: tuple[float, float, float],
    surface_tolerance_mm: float = 3.0,
) -> dict[str, Any]:

    prediction_labels = np.asarray(prediction_labels)
    target_labels = np.asarray(target_labels)
    prediction_tumor = prediction_labels == 2
    target_tumor = target_labels == 2
    prediction_liver = prediction_labels > 0
    target_liver = target_labels > 0
    return {
        "liver": binary_metrics(prediction_liver, target_liver, spacing, surface_tolerance_mm),
        "tumor": binary_metrics(prediction_tumor, target_tumor, spacing, surface_tolerance_mm),
        "lesion_detection": lesion_detection_metrics(prediction_tumor, target_tumor),
        "whole_liver_definition": "label 1 union label 2",
    }


def _binary_dice_only(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    intersection = float(np.logical_and(prediction, truth).sum())
    pred_count = float(prediction.sum())
    truth_count = float(truth.sum())
    return {
        "dice": _safe_div(2.0 * intersection, pred_count + truth_count),
        "hd95_mm": float("nan"),
        "nsd": float("nan"),
        "assd_mm": float("nan"),
        "precision": _safe_div(intersection, pred_count),
        "recall": _safe_div(intersection, truth_count),
        "predicted_voxels": pred_count,
        "target_voxels": truth_count,
    }


def multiclass_lits_dice_metrics(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, Any]:


    prediction_labels = np.asarray(prediction_labels)
    target_labels = np.asarray(target_labels)
    prediction_tumor = prediction_labels == 2
    target_tumor = target_labels == 2
    prediction_liver = prediction_labels > 0
    target_liver = target_labels > 0
    return {
        "liver": _binary_dice_only(prediction_liver, target_liver),
        "tumor": _binary_dice_only(prediction_tumor, target_tumor),
        "lesion_detection": lesion_detection_metrics(prediction_tumor, target_tumor),
        "whole_liver_definition": "label 1 union label 2",
    }

def multiclass_segmentation_loss(logits: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:

    if logits.shape[1] != 3 or target_onehot.shape[1] != 3:
        raise ValueError("LiTS segmentation requires three classes.")
    target_index = target_onehot.argmax(dim=1).long()
    ce = F.cross_entropy(logits.float(), target_index)
    probability = torch.softmax(logits.float(), dim=1)
    dims = tuple(range(2, probability.ndim))
    fg_target = target_onehot[:, 1:].float()
    intersection = (probability[:, 1:] * fg_target).sum(dim=dims)
    denominator = probability[:, 1:].sum(dim=dims) + fg_target.sum(dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return ce + dice_loss
