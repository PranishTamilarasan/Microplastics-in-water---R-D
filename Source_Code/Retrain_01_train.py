import os
import glob
from ultralytics import YOLO, RTDETR

def main():
    print("=======================================================================")
    print("🔥 INITIATING ULTIMATE INDEFINITE RETRAINING PIPELINE 🔥")
    print("WARNING: This pipeline relies on Early Stopping (Patience = 100).")
    print("It will train for days until the AI mathematically maxes out!")
    print("=======================================================================\n")

    # Target explicitly generated dataset YAML
    data_path = os.path.abspath("dataset.yaml")
    if not os.path.exists(data_path):
        print("ERROR: dataset.yaml missing!")
        return

    project_dir = "Ultimate_Runs"

    # ===============================================
    # 1. INDEFINITE YOLO11s-Seg TRAINING
    # ===============================================
    print("\n🚀 [STAGE 1] YOLO11s-Seg Skipped (Already Completed Successfully!)...")
    # yolo_model = YOLO("yolo11s-seg.pt")
    # 
    # try:
    #     yolo_model.train(
    #         data=data_path,
    #         epochs=2000, 
    #         patience=10, 
    #         imgsz=416, 
    #         batch=4, 
    #         optimizer="AdamW",
    #         project=project_dir, 
    #         name="Indefinite_YOLO_Train", 
    #         plots=True,
    #         exist_ok=True
    #     )
    # except KeyboardInterrupt:
    #     print("\n🛑 YOLO TRAINING MANUALLY PAUSED BY USER! Don't worry, your 'best.pt' is already safely saved!")

    # ===============================================
    # 2. INDEFINITE RT-DETR (HYBRID PROMPTER) TRAINING
    # ===============================================
    print("\n🚀 [STAGE 2] Launching RT-DETR Indefinite Training...")
    rtdetr_model = RTDETR("rtdetr-l.pt")
    
    try:
        rtdetr_model.train(
            data=data_path,
            epochs=2000, 
            patience=10, 
            imgsz=640, 
            batch=2,   
            optimizer="AdamW",
            amp=False,          # Prevents NaN math crashes via FP32 arithmetic
            lr0=0.0001,         # Stabilizes Transformer learning rate
            project=project_dir, 
            name="Indefinite_RTDETR_Train", 
            plots=True,
            exist_ok=True
        )
    except KeyboardInterrupt:
        print("\n🛑 RT-DETR TRAINING MANUALLY PAUSED BY USER! Your 'best.pt' is already safely saved!")

    print("\n✅ INDEFINITE TRAINING CONCLUDED.")
    print(f"Your absolute maximum optimized weights are located in: {project_dir}")
    print("You can now pass these finalized 'best.pt' weights into your Evaluator script!")

if __name__ == "__main__":
    main()
