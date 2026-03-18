import sys
from pathlib import Path
import json

import cv2
import torch
import numpy as np
import pandas as pd

from torchvision.ops import box_convert

# =========================
# HELPERS
# =========================
def yolo_seg_to_mask(label_file, img_h=480, img_w=640, normalize=True):
    """
    Convert YOLOv8 format label file to a multi class mask (H x W).

    label_file: path to YOLO segmentation label .txt file
    img_h, img_w: output mask size
    normalize: True if coordinates are normalized [0,1]

    Returns:
        mask: np.ndarray (H, W), dtype=np.uint8
        labels: list of (class_id, polygon)
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    labels = []

    with open(label_file, "r") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue  # skip invalid line

        cls = int(parts[0])
        coords = np.array(list(map(float, parts[5:])), dtype=np.float32)
        pts = coords.reshape(-1, 2)

        if normalize:
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h

        pts = np.round(pts).astype(np.int32)

        # Fill polygon into mask
        cv2.fillPoly(mask, [pts], cls+1)  # classes start from 1 in mask, 0 is background

        labels.append((cls, pts))

    return mask, labels


def compute_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float((inter + eps) / (union + eps))


def compute_dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    inter = np.logical_and(pred, gt).sum()
    return float((2 * inter + eps) / (pred.sum() + gt.sum() + eps))


def compute_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float((pred == gt).sum() / gt.size)


def compute_precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    return float((tp + eps) / (tp + fp + eps))


def compute_recall(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred, gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    return float((tp + eps) / (tp + fn + eps))


def get_model_performance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred_mask.shape}, gt={gt_mask.shape}")

    return {
        "IoU": compute_iou(pred_mask, gt_mask),
        "Dice": compute_dice(pred_mask, gt_mask),
        "Accuracy": compute_accuracy(pred_mask, gt_mask),
        "Precision": compute_precision(pred_mask, gt_mask),
        "Recall": compute_recall(pred_mask, gt_mask),
    }


def merge_predicted_masks(masks: np.ndarray) -> np.ndarray:
    """
    Merge multiple predicted masks into one binary mask.
    """
    if masks is None or len(masks) == 0:
        return None

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return np.any(masks.astype(bool), axis=0)


def run_inference_on_image(image_path: Path) -> tuple[np.ndarray, dict]:
    """
    Run GroundingDINO + SAM2 on one image and return:
    - merged predicted mask (bool HxW)
    - metadata dict
    """
    image_source, image = load_image(str(image_path))
    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    h, w, _ = image_source.shape

    if boxes.shape[0] == 0:
        pred_mask = np.zeros((h, w), dtype=bool)
        meta = {
            "num_boxes": 0,
            "avg_confidence": 0.0,
            "labels": [],
        }
        return pred_mask, meta

    boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=MULTIMASK_OUTPUT,
    )

    if MULTIMASK_OUTPUT:
        best = np.argmax(scores, axis=1)
        masks = masks[np.arange(masks.shape[0]), best]

    pred_mask = merge_predicted_masks(masks)
    if pred_mask is None:
        pred_mask = np.zeros((h, w), dtype=bool)

    conf_np = confidences.cpu().numpy() if hasattr(confidences, "cpu") else np.asarray(confidences)

    meta = {
        "num_boxes": int(len(input_boxes)),
        "avg_confidence": float(np.mean(conf_np)) if len(conf_np) else 0.0,
        "labels": list(labels),
        "input_boxes": input_boxes.tolist(),
    }

    return pred_mask, meta


def make_overlay(image_bgr: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """
    Green = GT only
    Red = Pred only
    Yellow = overlap
    """
    overlay = image_bgr.copy()

    pred_only = np.logical_and(pred_mask, ~gt_mask)
    gt_only = np.logical_and(gt_mask, ~pred_mask)
    overlap = np.logical_and(pred_mask, gt_mask)

    overlay[gt_only] = [0, 255, 0]
    overlay[pred_only] = [0, 0, 255]
    overlay[overlap] = [0, 255, 255]

    return cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)


def test_model_performance(image_folder: Path, label_folder: Path, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    """
    Compute average model performance on all matching image/label pairs.
    Assumes labels are YOLO segmentation txt files with same stem as image.
    """
    image_files = {
        p.stem: p for p in image_folder.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    }
    label_files = {
        p.stem: p for p in label_folder.iterdir()
        if p.suffix.lower() == ".txt"
    }

    common_stems = sorted(image_files.keys() & label_files.keys())

    if not common_stems:
        raise ValueError("No matching image/label pairs found.")

    skipped_images = sorted(image_files.keys() - label_files.keys())
    skipped_labels = sorted(label_files.keys() - image_files.keys())

    if skipped_images:
        print("Skipping images without labels:", skipped_images)
    if skipped_labels:
        print("Skipping labels without images:", skipped_labels)

    rows = []

    for stem in common_stems:
        image_path = image_files[stem]
        label_path = label_files[stem]

        print(f"Processing {stem}...")

        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            print(f"[WARN] Failed to read image {image_path}")
            continue

        img_h, img_w = rgb.shape[:2]

        pred_mask, meta = run_inference_on_image(image_path)
        gt_mask = yolo_seg_to_mask(label_path, img_h, img_w)

        metrics = get_model_performance(pred_mask, gt_mask)

        # save predicted mask
        pred_u8 = (pred_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_dir / "pred_masks" / f"{stem}_pred.png"), pred_u8)

        # save overlay
        overlay = make_overlay(rgb, pred_mask, gt_mask)
        cv2.imwrite(str(output_dir / "viz" / f"{stem}_overlay.png"), overlay)

        row = {
            "stem": stem,
            "image_path": str(image_path),
            "label_path": str(label_path),
            "num_boxes": meta["num_boxes"],
            "avg_confidence": meta["avg_confidence"],
            **metrics,
            "pred_pixels": int(pred_mask.sum()),
            "gt_pixels": int(gt_mask.sum()),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    avg_metrics = {}
    for col in ["IoU", "Dice", "Accuracy", "Precision", "Recall"]:
        avg_metrics[col] = float(df[col].mean()) if len(df) else None

    return avg_metrics, df