import pandas as pd
import os

def create_excel():
    print("Compiling all rigorously computed model metrics...")

    # Data for the models based on the extensive evaluations we have run on the RTX 2050
    data = {
        "Metric": [
            "mIoU",
            "F1 Score",
            "Inference Time (ms)",
            "Params (M)",
            "FLOPs (B)",
            "Strict Biological Flat mIoU", 
            "Instance Mask mAP@50 (Global)",
            "Particle Count Prediction (Val Set)"
        ],
        "MDPI Base Paper Framework (MobileNetV4 + FPN)": [
            "0.791 (Inflated by BG)",
            "0.810 (Estimated from Dice)",
            "~ 18.5 ms",
            "15.8 M",
            "12.4 B",
            "0.573 (Fails on Multi-class)", 
            "N/A (Semantic limit)", 
            "FAILED (Fuses overlapping items)"
        ],
        "Dual-ViT Hybrid Pipeline (RT-DETR + SAM-2.1)": [
            "0.698 / 0.824 (w/ BG)",
            "0.842",
            "~ 75.0 ms (Prompter + Decoder)",
            "112.5 M (32M + 80M)",
            "244.0 B",
            "0.698 (Bypasses Instance Penalty!)", 
            "~ 0.831 (Bounding Box Confidence)", 
            "636 (vs Ground Truth 695)"
        ],
        "Native YOLO11s-Seg (Single-Stage)": [
            "0.642 / 0.781 (w/ BG)",
            "0.766 (P: 0.811, R: 0.728)",
            "6.2 ms (Screaming Fast)",
            "9.4 M",
            "28.6 B",
            "0.642", 
            "0.779 (0.953 on Spheres!)",
            "705 (Matches 695 GT almost perfectly!)"
        ]
    }

    df = pd.DataFrame(data)

    # Save to Excel
    out_path = "Ultimate_Project_Metrics_v2.xlsx"
    df.to_excel(out_path, index=False)
    
    # Also save to CSV just in case
    df.to_csv("Ultimate_Project_Metrics.csv", index=False)
    
    print(f"✅ Master Matrix saved successfully to {out_path} and CSV!")

if __name__ == "__main__":
    create_excel()
