import os
import glob
import json
import shutil
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

print("Preparing Dataset for Native YOLO11 Segmentation...")

# Local Paths
dataset_dir = "../dataset"
output_dir = "yolo_seg_dataset"

for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

class_map = {'Fiber': 0, 'Fragment': 1, 'Sphere': 2}
all_data = []

# Gather image paths
for cls_name, cls_id in class_map.items():
    json_paths = glob.glob(os.path.join(dataset_dir, cls_name, 'labels', '*.json'))
    for json_path in json_paths:
        img_path = json_path.replace('labels', 'images').replace('.json', '.jpg')
        if os.path.exists(img_path):
            all_data.append({'img': img_path, 'json': json_path, 'cls_id': cls_id})

# 80/20 train validation split
train_data, val_data = train_test_split(all_data, test_size=0.2, random_state=42)

def process_seg(data_list, split):
    for data in data_list:
        img_path = data['img']
        json_path = data['json']
        cls_id = data['cls_id']
        img_name = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        h, w = img.shape[:2]
        
        with open(json_path, 'r') as f:
            label_data = json.load(f)
            
        yolo_polygons = []
        for shape in label_data.get('shapes', []):
            pts = np.array(shape['points'])
            if len(pts) < 3: 
                continue
            
            # Normalize polygon coordinates strictly to [0 - 1]
            norm_pts = pts / np.array([w, h])
            
            # Flatten into a continuous line: x1 y1 x2 y2 ... xn yn
            flat_pts = " ".join([f"{x:.5f} {y:.5f}" for x, y in norm_pts])
            yolo_polygons.append(f"{cls_id} {flat_pts}")
            
        if yolo_polygons:
            shutil.copy(img_path, os.path.join(output_dir, 'images', split, img_name))
            
            txt_name = img_name.replace('.jpg', '.txt')
            with open(os.path.join(output_dir, 'labels', split, txt_name), 'w') as f:
                f.write('\n'.join(yolo_polygons))

process_seg(train_data, 'train')
process_seg(val_data, 'val')

# Generate the massive config for YOLO11
abs_output_dir = os.path.abspath(output_dir).replace('\\', '/')
yaml_content = f"""
path: {abs_output_dir}
train: images/train
val: images/val
names:
  0: Fiber
  1: Fragment
  2: Sphere
"""

with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content.strip())
    
print("✓ Dataset preparation complete! Look for 'yolo_seg_dataset/dataset.yaml'")
