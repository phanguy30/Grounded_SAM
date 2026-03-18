import sys
from pathlib import Path
import json

import cv2
import torch
import numpy as np
import pandas as pd

from torchvision.ops import box_convert

# ---- paths to submodule ----
LABELING_DIR = Path(__file__).resolve().parent
REPO_ROOT = LABELING_DIR.parent / "Grounded-SAM-2"
sys.path.insert(0, str(REPO_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict


# =========================
# CONFIG
# =========================
TEXT_PROMPT = "gate."

IMAGE_DIR = LABELING_DIR / "dataset" / "gate_dataset" / "images" / "train"
LABEL_DIR = LABELING_DIR / "dataset" / "gate_dataset" / "labels" / "train"
OUTPUT_DIR = LABELING_DIR / "outputs" / "batch_compare"

SAM2_CHECKPOINT = REPO_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = REPO_ROOT / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = REPO_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MULTIMASK_OUTPUT = False

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "viz").mkdir(exist_ok=True)
(OUTPUT_DIR / "pred_masks").mkdir(exist_ok=True)


# =========================
# LOAD MODELS ONCE
# =========================
sam2_model = build_sam2(
    SAM2_MODEL_CONFIG,
    str(SAM2_CHECKPOINT),
    device=DEVICE
)
sam2_predictor = SAM2ImagePredictor(sam2_model)

grounding_model = load_model(
    model_config_path=str(GROUNDING_DINO_CONFIG),
    model_checkpoint_path=str(GROUNDING_DINO_CHECKPOINT),
    device=DEVICE
)


# =========================
# HELPERS
# =========================

def yolo_bbox_line_to_xyxy(label_line: str, img_w: int, img_h: int):
    """
    Parse one YOLO bbox line:
    class_id x_center y_center width height
    all normalized to [0, 1]
    """
    parts = label_line.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid YOLO bbox line: {label_line}")

    class_id = int(parts[0])
    xc, yc, bw, bh = map(float, parts[1:5])

    xc *= img_w
    yc *= img_h
    bw *= img_w
    bh *= img_h

    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2

    return class_id, [x1, y1, x2, y2]


def yolo_bbox_file_to_boxes(label_file: Path, img_w: int, img_h: int):
    """
    Parse a YOLO bbox txt file into a list of xyxy boxes.
    """
    boxes = []
    class_ids = []

    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            class_id, box = yolo_bbox_line_to_xyxy(line, img_w, img_h)
            class_ids.append(class_id)
            boxes.append(box)

    return boxes, class_ids


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
        cv2.fillPoly(mask, [pts], cls+1)  # classes start from 1 in mask, 0 is background # we mostly only have one class anyways
        # so the mask is 0,1

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
    

def compute_box_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def get_bbox_performance(
    pred_boxes: list,
    gt_boxes: list,
    iou_threshold: float = 0.5
) -> dict:
    """
    Evaluate bounding box predictions using greedy matching.

    Parameters
    ----------
    pred_boxes : list of [x1, y1, x2, y2]
    gt_boxes : list of [x1, y1, x2, y2]
    iou_threshold : float

    Returns
    -------
    dict with IoU, Precision, Recall, F1
    """

    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"IoU": 1.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0}

    if len(pred_boxes) == 0:
        return {"IoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    if len(gt_boxes) == 0:
        return {"IoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0}

    matched_gt = set()
    matched_pred = set()
    ious = []

    # -------------------------
    # Greedy matching
    # -------------------------
    for i, pbox in enumerate(pred_boxes):
        best_j = -1
        best_iou = 0.0

        for j, gtbox in enumerate(gt_boxes):
            if j in matched_gt:
                continue

            iou = compute_box_iou(pbox, gtbox)

            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_iou >= iou_threshold and best_j != -1:
            matched_pred.add(i)
            matched_gt.add(best_j)
            ious.append(best_iou)

    # -------------------------
    # Metrics
    # -------------------------
    tp = len(matched_pred)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    mean_iou = float(np.mean(ious)) if ious else 0.0

    return {
        "IoU": mean_iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
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




def run_inference_on_image(image_path: Path, mode: str = "mask"):
    """
    Run GroundingDINO (and optionally SAM2) on one image.

    Parameters
    ----------
    image_path : Path
        Path to image.
    mode : str
        "bbox" -> return bounding boxes only
        "mask" -> return segmentation masks

    Returns
    -------
    output :
        If mode == "bbox":
            list of dicts, one per detection
        If mode == "mask":
            list of dicts, one per detection, each with mask + box + score
    meta : dict
        Summary metadata
    """
    if mode not in {"bbox", "mask"}:
        raise ValueError("mode must be either 'bbox' or 'mask'")

    image_source, image = load_image(str(image_path))
    h, w, _ = image_source.shape

    # Only needed for SAM2 mode
    if mode == "mask":
        sam2_predictor.set_image(image_source)
        
        
    # Calling GroundDino to get boxes
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # No detections
    if boxes.shape[0] == 0:
        return [], {
            "num_boxes": 0,
            "avg_confidence": 0.0,
            "labels": [],
            "image_size": [h, w],
            "mode": mode,
        }

    # Convert cxcywh -> xyxy in pixel coordinates
    boxes = boxes * torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    conf_np = confidences.cpu().numpy() if hasattr(confidences, "cpu") else np.asarray(confidences)

    # -------------------------
    # BBOX MODE
    # -------------------------
    if mode == "bbox":
        outputs = []

        for box, conf, label in zip(input_boxes, conf_np, labels):
            x1, y1, x2, y2 = box.tolist()

            outputs.append({
                "label": label,
                "confidence": float(conf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

        meta = {
            "num_boxes": len(outputs),
            "avg_confidence": float(np.mean(conf_np)) if len(conf_np) else 0.0,
            "labels": list(labels),
            "image_size": [h, w],
            "mode": mode,
        }

        return outputs, meta

    # -------------------------
    # MASK MODE
    # -------------------------
    outputs = []

    for box, conf, label in zip(input_boxes, conf_np, labels):
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=MULTIMASK_OUTPUT,
        )

        if MULTIMASK_OUTPUT:
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])
        else:
            best_mask = masks[0]
            best_score = float(scores[0]) if np.ndim(scores) > 0 else float(scores)

        outputs.append({
            "label": label,
            "confidence": float(conf),          # DINO confidence
            "sam_score": best_score,            # SAM2 mask score
            "bbox_xyxy": box.tolist(),
            "mask": best_mask.astype(bool),     # H x W boolean mask
        })

    meta = {
        "num_boxes": len(outputs),
        "avg_confidence": float(np.mean(conf_np)) if len(conf_np) else 0.0,
        "labels": list(labels),
        "image_size": [h, w],
        "mode": mode,
    }

    return outputs, meta



def make_overlay(
    image_bgr: np.ndarray,
    mode: str = "mask",
    pred_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
    pred_boxes: list | None = None,
    gt_boxes: list | None = None,
) -> np.ndarray:
    """
    Create overlay for either segmentation masks or bounding boxes.

    Parameters
    ----------
    image_bgr : np.ndarray
        Original image in BGR format.
    mode : str
        "mask" or "bbox"
    pred_mask : np.ndarray | None
        Predicted boolean mask (H, W), used in mask mode.
    gt_mask : np.ndarray | None
        Ground truth boolean mask (H, W), used in mask mode.
    pred_boxes : list | None
        Predicted boxes in xyxy format, used in bbox mode.
    gt_boxes : list | None
        Ground truth boxes in xyxy format, used in bbox mode.

    Returns
    -------
    np.ndarray
        Overlay image.
    """
    if mode not in {"mask", "bbox"}:
        raise ValueError("mode must be either 'mask' or 'bbox'")

    overlay = image_bgr.copy()

    # -------------------------
    # MASK MODE
    # -------------------------
    if mode == "mask":
        if pred_mask is None or gt_mask is None:
            raise ValueError("pred_mask and gt_mask are required for mode='mask'")

        pred_mask = pred_mask.astype(bool)
        gt_mask = gt_mask.astype(bool)

        pred_only = np.logical_and(pred_mask, ~gt_mask)
        gt_only = np.logical_and(gt_mask, ~pred_mask)
        overlap = np.logical_and(pred_mask, gt_mask)

        # Green = GT only
        # Red = Pred only
        # Yellow = overlap
        overlay[gt_only] = [0, 255, 0]
        overlay[pred_only] = [0, 0, 255]
        overlay[overlap] = [0, 255, 255]

        return cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0)

    # -------------------------
    # BBOX MODE
    # -------------------------
    pred_boxes = pred_boxes or []
    gt_boxes = gt_boxes or []

    # Draw GT boxes in green
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return cv2.addWeighted(image_bgr, 0.7, overlay, 0.3, 0)

def test_model_performance(
    image_folder: Path,
    label_folder: Path,
    mode: str = "mask"
) -> tuple[dict, pd.DataFrame]:
    """
    Compute average model performance on all matching image/label pairs.

    Parameters
    ----------
    image_folder : Path
        Folder containing images.
    label_folder : Path
        Folder containing YOLO .txt labels.
        Use YOLO bbox labels for mode="bbox" and YOLOv8 segmentation labels for mode="mask".
    mode : str
        "mask" -> evaluate segmentation masks
        "bbox" -> evaluate bounding boxes

    Returns
    -------
    avg_metrics : dict
        Average metrics over all samples.
    df : pd.DataFrame
        Per-image metrics.
    """
    if mode not in {"mask", "bbox"}:
        raise ValueError("mode must be either 'mask' or 'bbox'")

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

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[WARN] Failed to read image {image_path}")
            continue

        img_h, img_w = image_bgr.shape[:2]

        # -------------------------
        # MASK MODE
        # -------------------------
        if mode == "mask":
            pred_outputs, meta = run_inference_on_image(image_path, mode="mask")

            pred_mask = np.zeros((img_h, img_w), dtype=bool)
            for obj in pred_outputs:
                pred_mask |= obj["mask"].astype(bool)

            gt_mask, _ = yolo_seg_to_mask(label_path, img_h, img_w)
            gt_mask = gt_mask.astype(bool)

            metrics = get_model_performance(pred_mask, gt_mask)

            # save predicted mask
            pred_u8 = (pred_mask.astype(np.uint8) * 255)
            cv2.imwrite(str(OUTPUT_DIR / "pred_masks" / f"{stem}_pred.png"), pred_u8)

            # save overlay
            overlay = make_overlay(
                image_bgr=image_bgr,
                mode="mask",
                pred_mask=pred_mask,
                gt_mask=gt_mask,
            )
            cv2.imwrite(str(OUTPUT_DIR / "viz" / f"{stem}_overlay.png"), overlay)

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

        # -------------------------
        # BBOX MODE
        # -------------------------
        elif mode == "bbox":
            pred_outputs, meta = run_inference_on_image(image_path, mode="bbox")

            pred_boxes = [obj["bbox_xyxy"] for obj in pred_outputs]
            gt_boxes, gt_class_ids = yolo_bbox_file_to_boxes(label_path, img_w, img_h)
            

            metrics = get_bbox_performance(pred_boxes, gt_boxes)

            overlay = make_overlay(
                image_bgr=image_bgr,
                mode="bbox",
                pred_boxes=pred_boxes,
                gt_boxes=gt_boxes,
            )
            cv2.imwrite(str(OUTPUT_DIR / "viz" / f"{stem}_overlay.png"), overlay)

            row = {
                "stem": stem,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "num_boxes": meta["num_boxes"],
                "avg_confidence": meta["avg_confidence"],
                **metrics,
                "num_pred_boxes": len(pred_boxes),
                "num_gt_boxes": len(gt_boxes),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    if mode == "mask":
        metric_cols = ["IoU", "Dice", "Accuracy", "Precision", "Recall"]
    else:
        metric_cols = ["IoU", "Precision", "Recall", "F1"]

    avg_metrics = {}
    for col in metric_cols:
        avg_metrics[col] = float(df[col].mean()) if (len(df) and col in df.columns) else None

    return avg_metrics, df





if __name__ == "__main__":
    curr_mode = "bbox"  # "mask" or "bbox"
    avg_metrics, df = test_model_performance(IMAGE_DIR, LABEL_DIR, mode = curr_mode)

    df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    print("\nPer-image metrics:")
    print(df)

    print("\nAverage metrics:")
    print(avg_metrics)