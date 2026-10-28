import os
import time
import glob
import cv2
import numpy as np
from ultralytics import YOLO, SAM
from tqdm import tqdm

def calculate_iou_f1(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 1.0
    f1 = (2 * intersection) / (pred_mask.sum() + gt_mask.sum()) if (pred_mask.sum() + gt_mask.sum()) > 0 else 1.0
    return iou, f1

def get_true_metrics():
    print("=====================================================")
    print("🔥 INITIATING TRUE BRUTE-FORCE MATHEMATICAL EVALUATION 🔥")
    print("WARNING: Executing Boolean Pixel Flattening (Takes 35+ mins)")
    print("=====================================================\n")

    val_images = glob.glob(r"..\..\3_Native_YOLO11_Instance_Pipeline\yolo_seg_dataset\images\val\*.jpg", recursive=True)
    val_labels = glob.glob(r"..\..\3_Native_YOLO11_Instance_Pipeline\yolo_seg_dataset\labels\val\*.txt", recursive=True)

    yolo_model = YOLO(r"..\Native_YOLO11_Model\YOLO11_best.pt")
    hybrid_prompter = YOLO(r"..\Hybrid_SAM_Model\RT_DETR_best.pt")
    sam = SAM("sam_b.pt")

    results = {
        "Hybrid": {"iou": 0, "f1": 0, "count": 0, "time": 0},
        "YOLO11": {"iou": 0, "f1": 0, "count": 0, "time": 0}
    }

    # ==========================================
    # TRUE PIXEL LOOP
    # ==========================================
    for img_path in tqdm(val_images, desc="Flattening & Crunching Arrays"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # 1. GENERATE GROUND TRUTH BOOLEAN MASK
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

        # ---------------------------------------------
        # 2. EVALUATE NATIVE YOLO11
        # ---------------------------------------------
        start_yolo = time.time()
        yolo_res = yolo_model(img, conf=0.25, verbose=False)[0]
        results["YOLO11"]["time"] += (time.time() - start_yolo)
        
        y_pred_mask = np.zeros((h, w), dtype=bool)
        if yolo_res.masks is not None:
            results["YOLO11"]["count"] += len(yolo_res.boxes)
            masks_data = yolo_res.masks.data.cpu().numpy()
            for mask in masks_data:
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                y_pred_mask = np.logical_or(y_pred_mask, mask_resized)
        
        y_iou, y_f1 = calculate_iou_f1(y_pred_mask, gt_mask)
        results["YOLO11"]["iou"] += y_iou
        results["YOLO11"]["f1"] += y_f1

        # ---------------------------------------------
        # 3. EVALUATE HYBRID PIPELINE
        # ---------------------------------------------
        start_hybrid = time.time()
        box_res = hybrid_prompter(img, conf=0.25, verbose=False)[0]
        boxes = box_res.boxes.xyxy
        
        h_pred_mask = np.zeros((h, w), dtype=bool)
        if len(boxes) > 0:
            results["Hybrid"]["count"] += len(boxes)
            sam_res = sam(img, bboxes=boxes, verbose=False)
            if sam_res[0].masks is not None:
                sam_masks = sam_res[0].masks.data.cpu().numpy()
                for mask in sam_masks:
                    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                    h_pred_mask = np.logical_or(h_pred_mask, mask_resized)
        results["Hybrid"]["time"] += (time.time() - start_hybrid)

        h_iou, h_f1 = calculate_iou_f1(h_pred_mask, gt_mask)
        results["Hybrid"]["iou"] += h_iou
        results["Hybrid"]["f1"] += h_f1

    # ==========================================
    # FINALIZE AVERAGES
    # ==========================================
    total_imgs = len(val_images)
    print("\n✅ BRUTE-FORCE COMPUTATION COMPLETE.\n")
    print("-" * 75)
    print(f"{'Metric':<30} | {'Hybrid (RT-DETR+SAM)':<20} | {'YOLO11s-Seg':<20}")
    print("-" * 75)
    print(f"{'mIoU (TRUE strict foreground)':<30} | {results['Hybrid']['iou']/total_imgs:<20.4f} | {results['YOLO11']['iou']/total_imgs:<20.4f}")
    print(f"{'F1 Score (TRUE Dice)':<30} | {results['Hybrid']['f1']/total_imgs:<20.4f} | {results['YOLO11']['f1']/total_imgs:<20.4f}")
    print(f"{'Inference Time':<30} | {(results['Hybrid']['time']/total_imgs)*1000:<20.1f} | {(results['YOLO11']['time']/total_imgs)*1000:<20.1f}")
    print(f"{'Particle Count (GT=695)':<30} | {results['Hybrid']['count']:<20} | {results['YOLO11']['count']:<20}")
    print("-" * 75)

if __name__ == "__main__":
    get_true_metrics()
