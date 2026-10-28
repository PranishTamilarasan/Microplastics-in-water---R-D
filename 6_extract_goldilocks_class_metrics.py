import os
from ultralytics import YOLO

def extract_class_metrics():
    print("=========================================================")
    print("🥇 EXTRACTING GOLDILOCKS CLASS METRICS 🥇")
    print("Loading Native YOLO11s-Seg Final Model...")
    print("=========================================================\n")

    weights_path = r"Native_YOLO11_Model\YOLO11_best.pt"
    
    # Using the flawless newly generated absolute path dataset YAML from our previous step!
    absolute_base = r"D:\PRANISH\R_and_D\Base Paper Reproduce - Toxics"
    dataset_yaml = os.path.join(absolute_base, "5_Ultimate_Indefinite_Retraining", "dataset.yaml")

    if not os.path.exists(weights_path):
        print(f"ERROR: Cannot find weights at {weights_path}")
        return

    model = YOLO(weights_path)

    print("Running deep PyTorch calculation over 488 images...")
    print("(This takes ~90 seconds because it plots 10 different calculus integration curves per particle!)")
    
    # Internal evaluation mapping (Verbose=True natively prints the pristine class breakdown!)
    metrics = model.val(data=dataset_yaml, imgsz=416, split="val", verbose=True)

    print("\n✅ Native PyTorch Class Breakdown Printed successfully!")
        
if __name__ == "__main__":
    extract_class_metrics()
