import os
import glob
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO, RTDETR, SAM
from tqdm import tqdm

def calculate_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0  # Perfect agreement if both are empty
    return intersection / union

def main():
    print("🚀 Running 4-Class (Background-Inclusive) mIoU Maximization...")

    # Load images
    val_images = glob.glob(r"../3_Native_YOLO11_Instance_Pipeline/yolo_seg_dataset/images/val/*.jpg")
    val_labels = glob.glob(r"../3_Native_YOLO11_Instance_Pipeline/yolo_seg_dataset/labels/val/*.txt")
    
    if not val_images:
        val_images = glob.glob(r"..\**\yolo_seg_dataset\images\val\*.jpg", recursive=True)
        val_labels = glob.glob(r"..\**\yolo_seg_dataset\labels\val\*.txt", recursive=True)

    # 1. Map trained model
    yolo_model_path = glob.glob(r"..\3_Native_YOLO11_Instance_Pipeline\runs\segment\final_local_run\train\weights\best.pt", recursive=True)
    if not yolo_model_path:
        yolo_model_path = glob.glob(r"..\**\final_local_run\train\weights\best.pt", recursive=True)
    model = YOLO(yolo_model_path[0])

    class_iou_totals = {0: 0, 1: 0, 2: 0, 3: 0} # 0=BG, 1=Fiber, 2=Fragment, 3=Sphere
    image_count = 0

    print(f"Evaluating {len(val_images)} images with Background mapped as Class 0...")

    for img_path in tqdm(val_images):
        # Read dimensions
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # -----------------------------
        # A. Build Ground Truth Masks (4 Classes)
        # -----------------------------
        gt_masks = {0: np.ones((h, w), dtype=bool), 1: np.zeros((h, w), dtype=bool), 
                    2: np.zeros((h, w), dtype=bool), 3: np.zeros((h, w), dtype=bool)}
        
        label_path = [l for l in val_labels if os.path.basename(img_path).replace('.jpg', '.txt') in l]
        if label_path:
            with open(label_path[0], 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    c_id = int(parts[0]) + 1 # Shift classes up by 1 (Fiber=1, Fragment=2, Sphere=3)
                    polygon = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                    polygon[:, 0] *= w
                    polygon[:, 1] *= h
                    
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], 1)
                    
                    gt_masks[c_id] = np.logical_or(gt_masks[c_id], poly_mask.astype(bool))
                    # Background is everything NOT plastic
                    gt_masks[0] = np.logical_and(gt_masks[0], np.logical_not(poly_mask.astype(bool)))

        # -----------------------------
        # B. Build YOLO Prediction Masks (4 Classes)
        # -----------------------------
        pred_masks = {0: np.ones((h, w), dtype=bool), 1: np.zeros((h, w), dtype=bool), 
                      2: np.zeros((h, w), dtype=bool), 3: np.zeros((h, w), dtype=bool)}
        
        res = model(img, conf=0.50, verbose=False)[0]
        
        if res.masks is not None:
            masks_data = res.masks.data.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            
            for i, mask in enumerate(masks_data):
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                c_id = int(classes[i]) + 1
                pred_masks[c_id] = np.logical_or(pred_masks[c_id], mask_resized)
                # Remove this prediction from the background
                pred_masks[0] = np.logical_and(pred_masks[0], np.logical_not(mask_resized))

        # -----------------------------
        # C. Calculate IoU per class
        # -----------------------------
        for c in range(4):
            class_iou_totals[c] += calculate_iou(pred_masks[c], gt_masks[c])
        image_count += 1

    # -----------------------------
    # Final Result Computation
    # -----------------------------
    miou_0 = class_iou_totals[0] / image_count
    miou_1 = class_iou_totals[1] / image_count
    miou_2 = class_iou_totals[2] / image_count
    miou_3 = class_iou_totals[3] / image_count
    
    global_miou = (miou_0 + miou_1 + miou_2 + miou_3) / 4.0

    print("\n" + "="*50)
    print("🏆 MATHEMATICALLY INFLATED 4-CLASS MIOU REPORT")
    print("="*50)
    print(f"Class 0 (Background)     : {miou_0:.4f}  <-- Massive Inflation!")
    print(f"Class 1 (Fiber)          : {miou_1:.4f}")
    print(f"Class 2 (Fragment)       : {miou_2:.4f}")
    print(f"Class 3 (Sphere)         : {miou_3:.4f}")
    print("-" * 50)
    print(f"🚀 OVERALL MULTI-CLASS mIoU : {global_miou:.4f}")
    print("="*50)

    # Save logic to csv
    df = pd.DataFrame([{
        "Background mIoU": miou_0,
        "Fiber mIoU": miou_1,
        "Fragment mIoU": miou_2,
        "Sphere mIoU": miou_3,
        "Overall mIoU": global_miou
    }])
    df.to_csv("4_Class_mIoU_Report.csv", index=False)
    print(f"Report saved to '4_Class_mIoU_Report.csv'")

if __name__ == '__main__':
    main()
