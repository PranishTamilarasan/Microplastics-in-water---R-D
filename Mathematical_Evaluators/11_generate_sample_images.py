import os
import glob
import shutil
from ultralytics import YOLO

def main():
    print("🎨 Generating Presentation Images...")

    # 1. Safely find the YOLO weights and Val Images no matter where they are
    model_path = glob.glob(r"**\best.pt", recursive=True)
    yolo_weights = [p for p in model_path if "segment" in p.lower() or "final_local_run" in p.lower()][0]
    
    val_images = glob.glob(r"**\yolo_seg_dataset\images\val\*.jpg", recursive=True)
    if not val_images:
        print("Could not find validation images!")
        return
        
    print("🔍 Hunting for an image with heavy aggregation (overlapping items)...")
    
    model = YOLO(yolo_weights)
    target_image = None
    
    for img in val_images:
        res = model(img, conf=0.55, verbose=False)
        # Look for an image that has AT LEAST 5 particles (guaranteeing overlap/crowding!)
        if res[0].boxes is not None and len(res[0].boxes) >= 5:
            target_image = img
            break
            
    if target_image is None:
        print("Couldn't find one with 5+, defaulting to the first one.")
        target_image = val_images[0]

    print(f"✅ Found weights: {yolo_weights}")
    print(f"✅ Drawing Masks on: {target_image}")

    # 2. Run Inference
    model = YOLO(yolo_weights)
    export_dir = "YOLO_Presentation_Images"
    os.makedirs(export_dir, exist_ok=True)
    
    results = model(target_image, conf=0.55, verbose=False)
    
    # Extract image and force it to save EXACTLY next to the raw image!
    results[0].save(os.path.join(export_dir, "YOLO_Predicted_Mask.jpg"))

    # 3. Copy the RAW Unmodified Image right next to it!
    raw_save_path = os.path.join(export_dir, "Raw_Original_Image.jpg")
    shutil.copy(target_image, raw_save_path)

    print(f"\n🎉 DONE! Open the folder '{export_dir}' on your sidebar.")
    print("You now have the Raw Image AND the YOLO Prediction Image side-by-side to paste into Word!")

if __name__ == "__main__":
    main()
