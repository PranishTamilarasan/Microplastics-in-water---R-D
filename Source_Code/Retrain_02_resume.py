import os
import glob
from ultralytics import YOLO, RTDETR

def main():
    print("=======================================================================")
    print("🔋 RESUMING PREVIOUSLY PAUSED TRAINING 🔋")
    print("Restoring Optimizer states, Learning Rates, and Epoch counters...")
    print("=======================================================================\n")

    project_dir = "Ultimate_Runs"

    # =========================================================
    # 1. YOLO RESUME (Currently Skipped/Commented Out!)
    # =========================================================
    print("✅ Skipping YOLO11s-Seg (Already safely maxed out and completed).")
    
    # yolo_last_weights = glob.glob(os.path.join(project_dir, "Indefinite_YOLO_Train**", "weights", "last.pt"))
    # if yolo_last_weights:
    #     print(f"✅ Found interrupted YOLO state at: {yolo_last_weights[0]}")
    #     print("Resuming YOLO11s-Seg...")
    #     try:
    #         model = YOLO(yolo_last_weights[0])
    #         model.train(resume=True)
    #         print("🚀 YOLO Training Concluded Successfully!")
    #     except KeyboardInterrupt:
    #         print("\n🛑 YOLO TRAINING RE-PAUSED BY USER!")
    # else:
    #     print("🤷‍♂️ No paused YOLO training found to resume.")

    # The TRUE root path where PyTorch hides custom projects
    search_root = "runs"

    # =========================================================
    # 2. Look for RT-DETR's interrupted state
    # =========================================================
    rtdetr_last_weights = glob.glob(os.path.join(search_root, "**", "Indefinite_RTDETR_Train*", "weights", "last.pt"), recursive=True)
    if rtdetr_last_weights:
        # Sort by creation time to grab the most recent run (e.g. Train2, Train3)
        rtdetr_last_weights.sort(key=os.path.getmtime, reverse=True)
        print(f"\n✅ Found interrupted RT-DETR state at: {rtdetr_last_weights[0]}")
        print("Resuming RT-DETR...")
        try:
            model = RTDETR(rtdetr_last_weights[0])
            model.train(resume=True)
            print("🚀 RT-DETR Training Concluded Successfully!")
        except KeyboardInterrupt:
            print("\n🛑 RT-DETR TRAINING RE-PAUSED BY USER!")
    else:
        print("🤷‍♂️ No paused RT-DETR training found to resume.")

if __name__ == "__main__":
    main()
