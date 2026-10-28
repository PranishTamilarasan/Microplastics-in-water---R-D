import os
import glob
import shutil
from ultralytics import RTDETR, SAM

def main():
    print("🎨 Generating HYBRID Presentation Images (RT-DETR + SAM-2.1)...")

    # 1. Safely find the RT-DETR weights (Boxes)
    box_weights_path = glob.glob(r"**\rt-detr**\best.pt", recursive=True)
    if not box_weights_path:
        # Fallback to general detect folder if it's named differently
        box_weights_path = glob.glob(r"**\detect\**\best.pt", recursive=True)
        
    rt_detr_weights = box_weights_path[0]
    print(f"✅ Found RT-DETR Prompter: {rt_detr_weights}")
    
    # 2. Init Models
    prompter = RTDETR(rt_detr_weights)
    sam_decoder = SAM("sam_b.pt") # Using the locally-cached SAM weights!

    # Target specific images to compare!
    target_stems = ['B_CW_3_above500_303', 'DD_CW_1_Stereoscope_212-500_52', 'B_CW_1_212-500_135']
    val_images = glob.glob(r"**\yolo_seg_dataset\images\val\*.jpg", recursive=True)
    
    export_dir = "Hybrid_Presentation_Images"
    os.makedirs(export_dir, exist_ok=True)

    for stem in target_stems:
        matches = [img for img in val_images if stem in img]
        if not matches:
            # If specific images aren't found, grab a random validation image
            matches = [val_images[3]]
            stem = "Random_Backup_Image"
            
        target_image = matches[0]
        print(f"✅ Drawing HYBRID Masks on: {stem}")

        # A. Prompter (RT-DETR) draws the bounding boxes
        box_results = prompter(target_image, conf=0.55, verbose=False)
        boxes = box_results[0].boxes.xyxy # Get coordinates
        
        # Save just the boxes (optional, for cool comparison!)
        box_results[0].save(os.path.join(export_dir, f"Step1_RT_DETR_Boxes_{stem}.jpg"))

        # B. Decoder (SAM-2.1) traces the exact edges using the boxes!
        if len(boxes) > 0:
            sam_results = sam_decoder(target_image, bboxes=boxes, verbose=False)
            sam_results[0].save(os.path.join(export_dir, f"Step2_SAM_Masks_{stem}.jpg"))
        else:
            print(f"No objects detected in {stem} to prompt SAM.")

        # C. Save Raw Image
        shutil.copy(target_image, os.path.join(export_dir, f"Raw_Image_{stem}.jpg"))

    print(f"\n🎉 DONE! Open the folder '{export_dir}' on your sidebar.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
