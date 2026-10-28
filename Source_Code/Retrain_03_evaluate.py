import os
import time
import glob
import cv2
import numpy as np
import gc
import torch
from ultralytics import YOLO, SAM
from tqdm import tqdm

def calculate_iou_f1_count(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 1.0
    f1 = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 1.0
    return iou, f1

def get_true_metrics():
    print("=====================================================")
    print("🔥 EVALUATING NEW INDEFINITE (MAXED-OUT) WEIGHTS 🔥")
    print("Testing against True Boolean Pixel Mapping...")
    print("=====================================================\n")

    val_images = glob.glob(r"..\3_Native_YOLO11_Instance_Pipeline\yolo_seg_dataset\images\val\*.jpg", recursive=True)
    val_labels = glob.glob(r"..\3_Native_YOLO11_Instance_Pipeline\yolo_seg_dataset\labels\val\*.txt", recursive=True)

    # Automatically identify the absolute highest epoch weights dynamically
    yolo_weights = glob.glob(r"runs\segment\**\Indefinite_YOLO_Train*\weights\best.pt", recursive=True)
    rtdetr_weights = glob.glob(r"runs\detect\**\Indefinite_RTDETR_Train*\weights\best.pt", recursive=True)

    yolo_weights.sort(key=os.path.getmtime, reverse=True)
    rtdetr_weights.sort(key=os.path.getmtime, reverse=True)

    print(f"✅ Loading New YOLO11s: {yolo_weights[0]}")
    print(f"✅ Loading New RT-DETR: {rtdetr_weights[0]}")
    
    yolo_model = YOLO(yolo_weights[0])
    hybrid_prompter = YOLO(rtdetr_weights[0])
    sam = SAM("sam_b.pt")

    results = {
        "Hybrid": {"iou": 0, "f1": 0, "count": 0, "time": 0},
        "YOLO11": {"iou": 0, "f1": 0, "count": 0, "time": 0}
    }

    for img_path in tqdm(val_images, desc="Flattening True Boolean Pixels"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 1. GROUND TRUTH
        gt_mask = np.zeros((h, w), dtype=bool)
        label_path = [l for l in val_labels if os.path.basename(img_path).replace('.jpg', '.txt') in l]
        if label_path:
            with open(label_path[0], 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    polygon = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    polygon[:, 0] *= w
                    polygon[:, 1] *= h
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 1)
                    gt_mask = np.logical_or(gt_mask, poly_mask.astype(bool))

        # 2. YOLO NATIVE
        start = time.time()
        yolo_res = yolo_model(img, conf=0.25, verbose=False)[0]
        results["YOLO11"]["time"] += (time.time() - start)
        y_pred = np.zeros((h, w), dtype=bool)
        if yolo_res.masks is not None:
            results["YOLO11"]["count"] += len(yolo_res.boxes)
            for mask in yolo_res.masks.data.cpu().numpy():
                y_pred = np.logical_or(y_pred, cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))
        
        y_iou, y_f1 = calculate_iou_f1_count(y_pred, gt_mask)
        results["YOLO11"]["iou"] += y_iou
        results["YOLO11"]["f1"] += y_f1

        # 3. HYBRID TRANSFORMER
        start = time.time()
        box_res = hybrid_prompter(img, conf=0.25, verbose=False)[0]
        h_pred = np.zeros((h, w), dtype=bool)
        boxes = box_res.boxes.xyxy
        if len(boxes) > 0:
            results["Hybrid"]["count"] += len(boxes)
            sam_res = sam(img, bboxes=boxes, verbose=False)
            if sam_res[0].masks is not None:
                for mask in sam_res[0].masks.data.cpu().numpy():
                    h_pred = np.logical_or(h_pred, cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool))
        results["Hybrid"]["time"] += (time.time() - start)

        h_iou, h_f1 = calculate_iou_f1_count(h_pred, gt_mask)
        results["Hybrid"]["iou"] += h_iou
        results["Hybrid"]["f1"] += h_f1

        # 4. AGGRESSIVELY EMPTY CUDA VRAM CACHE
        del y_pred, yolo_res, h_pred, box_res
        if len(boxes) > 0:
            del sam_res
        gc.collect()
        torch.cuda.empty_cache()

    total = len(val_images)
    print("\n✅ INDEFINITE WEIGHTS COMPUTATION COMPLETE.\n")
    print("-" * 75)
    print(f"{'Metric (Based on Indefinite Models)':<33} | {'Hybrid (RT-DETR+SAM)':<20} | {'YOLO11s-Seg':<20}")
    print("-" * 75)
    print(f"{'mIoU (TRUE strict foreground)':<33} | {results['Hybrid']['iou']/total:<20.4f} | {results['YOLO11']['iou']/total:<20.4f}")
    print(f"{'F1 Score (TRUE Dice)':<33} | {results['Hybrid']['f1']/total:<20.4f} | {results['YOLO11']['f1']/total:<20.4f}")
    print(f"{'Inference Latency':<33} | {(results['Hybrid']['time']/total)*1000:<20.1f} | {(results['YOLO11']['time']/total)*1000:<20.1f}")
    print(f"{'Particle Count (GT=695)':<33} | {results['Hybrid']['count']:<20} | {results['YOLO11']['count']:<20}")
    print("-" * 75)

if __name__ == "__main__":
    get_true_metrics()
