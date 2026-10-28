import os
import glob
import cv2
import numpy as np
import argparse
import tqdm
from ultralytics import YOLO

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union (IoU)"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def evaluate_yolo_seg(model_path):
    print(f"Initializing YOLO11 Segmentation Evaluation using: {model_path}...")
    
    # We use the images we just prepped in the yolo_seg_dataset folder
    val_images = glob.glob(os.path.join("yolo_seg_dataset", "images", "val", "*.jpg"))
    if not val_images:
        print("Validation dataset not found. Verify 'yolo_seg_dataset' exists.")
        return

    try:    
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return

    print(f"Testing on {len(val_images)} validation images...")
    
    total_iou = 0.0
    valid_count = 0
    
    for img_path in tqdm.tqdm(val_images, desc="Calculating Exact Pixel mIoU"):
        img_name = os.path.basename(img_path)
        
        # Locate the original ground truth mask from the main dataset folder
        original_base = os.path.join("..", "dataset")
        gt_mask_path = None
        
        for cls_name in ["Fiber", "Fragment", "Sphere"]:
            potential_mask = os.path.join(original_base, cls_name, "masks", img_name.replace('.jpg', '_mask.png'))
            if os.path.exists(potential_mask):
                gt_mask_path = potential_mask
                break
                
        if not gt_mask_path:
            continue
            
        # Ground truth mask (Binary, rigidly 2D)
        gt_img = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_img.ndim > 2:
            gt_img = gt_img.squeeze()
        gt_binary = (gt_img > 127)
        
        # Inference (YOLO11 Segment)
        results = model(img_path, conf=0.15, verbose=False)
        
        # Initialize an empty prediction canvas
        pred_binary = np.zeros((gt_binary.shape[0], gt_binary.shape[1]), dtype=bool)
        
        if results[0].masks is not None:
            # YOLO11 instance masks
            mask_data = results[0].masks.data.cpu().numpy()
            
            for single_mask in mask_data:
                # Force flat 2D
                single_mask_arr = np.array(single_mask)
                while single_mask_arr.ndim > 2:
                    if single_mask_arr.shape[-1] == 1:
                        single_mask_arr = single_mask_arr[..., 0]
                    elif single_mask_arr.shape[0] == 1:
                        single_mask_arr = single_mask_arr[0]
                    else:
                        break
                        
                single_mask_sq = single_mask_arr.astype(bool)
                
                # Resize YOLO internal feature map to match original image dimensions
                if single_mask_sq.shape != pred_binary.shape:
                    single_mask_sq = cv2.resize(single_mask_sq.astype(np.uint8), (pred_binary.shape[1], pred_binary.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                    
                # Combine this instance mask into the global mask
                pred_binary = np.logical_or(pred_binary, single_mask_sq)
                
        # Final shape safety check
        if pred_binary.shape != gt_binary.shape:
            pred_binary = cv2.resize(pred_binary.astype(np.uint8), (gt_binary.shape[1], gt_binary.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            
        iou = calculate_iou(pred_binary, gt_binary)
        total_iou += iou
        valid_count += 1
        
    final_miou = total_iou / valid_count if valid_count > 0 else 0
    print("\n" + "="*60)
    print(f"YOLO11 SEGMENTATION - EXACT PIXEL mIoU MATRIX")
    print("="*60)
    print(f"Base Paper Semantic Reference:  ~0.690 mIoU")
    print(f"Your YOLO11-Seg Flat mIoU:      {final_miou:.4f} mIoU")
    print(f"Evaluated Images:               {valid_count}")
    print("="*60)
    print("Conclusion: Provides the raw math for apples-to-apples comparison!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 Segmentation pixel mIoU")
    # Default to the most recent segment run
    default_path = r"..\runs\segment\train-2\weights\best.pt"
    parser.add_argument("--model", type=str, default=default_path, help="Path to best.pt weights")
    args = parser.parse_args()
    
    evaluate_yolo_seg(args.model)
