import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO, SAM
import matplotlib.pyplot as plt

def run_hybrid_instance_segmentation(image_path, yolo_checkpoint="yolov8n.pt"):
    """
    Architecture Proposed in Novelty 1: 
    1. YOLO extracts Bounding Boxes (Object Detector)
    2. MobileSAM receives Box Prompts and extracts precise distinct instances
    """
    print(f"Loading Image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        # Load YOLO detector (assuming it has been trained on bounding boxes)
        print("Loading YOLO Detector...")
        detector = YOLO(yolo_checkpoint)
        
        # Load Mobile SAM 
        print("Loading MobileSAM (Lightweight Instance Segmenter)...")
        sam = SAM('mobile_sam.pt')
        
    except Exception as e:
        print(f"Error loading models. Have you run 'pip install ultralytics'? Error: {e}")
        return

    # Step 1: Detect bounding boxes
    print(f"Stage 1: Predicting Bounding Boxes for {os.path.basename(image_path)}")
    # We pass the path directly to avoid RGB/BGR channel swap issues, lowering conf threshold
    det_results = detector(image_path, conf=0.15)
    
    boxes = det_results[0].boxes.xyxy.cpu().numpy()
    classes = det_results[0].boxes.cls.cpu().numpy()
    
    if len(boxes) == 0:
        print("No particles detected by YOLO in this image. Trying next image...")
        return False
        
    print(f"Found {len(boxes)} particles. Passing boxes as prompts to MobileSAM...")

    # Step 2: SAM Instance Segmentation using Box Prompts
    all_masks = []
    
    # We pass the image and the bounding boxes as prompts
    sam_results = sam(img_rgb, bboxes=boxes)
    
    # Extract binary masks
    for result in sam_results:
        if result.masks is not None:
            mask_data = result.masks.data.cpu().numpy()
            all_masks.append(mask_data)
            
    # Visualize the results
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_rgb)
    plt.axis('off')
    
    # YOLO Bboxes
    plt.subplot(1, 3, 2)
    plt.title(f"YOLO Detections (Count: {len(boxes)})")
    res_plotted = det_results[0].plot()
    plt.imshow(res_plotted)
    plt.axis('off')
    
    # Mobile SAM Instance Masks
    plt.subplot(1, 3, 3)
    plt.title("MobileSAM Instance Segmentations")
    combined_mask = np.zeros_like(img_rgb)
    colors = np.random.randint(0, 255, (256, 3))
    
    for i, mask_batch in enumerate(all_masks):
        for j, single_mask in enumerate(mask_batch):
            color = colors[(i * len(mask_batch) + j) % 256]
            colored_mask = np.zeros_like(img_rgb)
            colored_mask[single_mask.astype(bool)] = color
            combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 0.6, 0)
            
    final_overlay = cv2.addWeighted(img_rgb, 0.5, combined_mask, 0.8, 0)
    plt.imshow(final_overlay)
    plt.axis('off')
    
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", "hybrid_segmentation_output.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Hybrid Instance Segmentation complete! Result saved to: {save_path}")
    return True

if __name__ == "__main__":
    import glob
    # Loop over validation images until we find one with particles
    val_images = glob.glob("yolo_dataset/images/val/*.jpg")
    if val_images:
        for img_path in val_images:
            success = run_hybrid_instance_segmentation(img_path, yolo_checkpoint=r"runs\detect\train-2\weights\best.pt")
            if success:
                break
    else:
        print("Please run 1_prepare_yolo_dataset.py first to format the data!")
