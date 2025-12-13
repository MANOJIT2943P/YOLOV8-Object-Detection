# Script to merge multiple YOLOv8 datasets into one unified dataset
# Merges stair_dataset, people_dataset, door_dataset, vehicle_dataset, pothole_dataset
# into smart_nav_dataset with proper class ID remapping

import os
import shutil

# Source datasets with their new class IDs
datasets = {
    "stair_dataset": 0,      # class 0
    "people_dataset": 1,     # class 1
    "door_dataset": 2,       # class 2
    "pothole_dataset": 3     # class 3
}

# Destination folders
base_dir = "smart_nav_dataset"
dst_img_train = os.path.join(base_dir, "images", "train")
dst_img_val = os.path.join(base_dir, "images", "val")
dst_img_test = os.path.join(base_dir, "images", "test")
dst_lbl_train = os.path.join(base_dir, "labels", "train")
dst_lbl_val = os.path.join(base_dir, "labels", "val")
dst_lbl_test = os.path.join(base_dir, "labels", "test")

# Mapping of source splits to destination splits
dst_map = {
    "train": (dst_img_train, dst_lbl_train),
    "valid": (dst_img_val, dst_lbl_val),
    "test": (dst_img_test, dst_lbl_test)
}

# Create all destination directories
print("Creating destination directories...")
os.makedirs(dst_img_train, exist_ok=True)
os.makedirs(dst_img_val, exist_ok=True)
os.makedirs(dst_img_test, exist_ok=True)
os.makedirs(dst_lbl_train, exist_ok=True)
os.makedirs(dst_lbl_val, exist_ok=True)
os.makedirs(dst_lbl_test, exist_ok=True)
print("Destination directories created successfully!")

# Process each dataset
for dataset_name, new_class_id in datasets.items():
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_name):
        print(f"Warning: Dataset {dataset_name} not found. Skipping...")
        continue
    
    # Process each split (train/valid/test)
    for split in ["train", "valid", "test"]:
        print(f"  Processing split: {split}")
        
        # Define source paths
        src_img_dir = os.path.join(dataset_name, split, "images")
        src_lbl_dir = os.path.join(dataset_name, split, "labels")
        
        # Check if source image directory exists
        if not os.path.exists(src_img_dir):
            # Try alternative path (without /images subdirectory)
            src_img_dir = os.path.join(dataset_name, split)
            if not os.path.exists(src_img_dir):
                print(f"    Skipping missing folder: {src_img_dir}")
                continue
        
        # Check if source label directory exists
        if not os.path.exists(src_lbl_dir):
            # Try alternative path (without /labels subdirectory)
            src_lbl_dir = os.path.join(dataset_name, split)
            if not os.path.exists(src_lbl_dir):
                print(f"    Warning: Label directory not found for {split}. Skipping...")
                continue
        
        # Get destination paths
        dst_img_dir, dst_lbl_dir = dst_map[split]
        
        # Process all files in the image directory
        file_count = 0
        for fname in os.listdir(src_img_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            try:
                # Copy image file
                src_img_path = os.path.join(src_img_dir, fname)
                dst_img_path = os.path.join(dst_img_dir, fname)
                shutil.copy2(src_img_path, dst_img_path)
                
                # Find corresponding label file (same name, .txt extension)
                lbl_name = os.path.splitext(fname)[0] + ".txt"
                src_lbl_path = os.path.join(src_lbl_dir, lbl_name)
                dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)
                
                # Copy and remap label file if it exists
                if os.path.exists(src_lbl_path):
                    with open(src_lbl_path, "r") as f_in, open(dst_lbl_path, "w") as f_out:
                        for line in f_in:
                            parts = line.strip().split()
                            if len(parts) > 0:
                                # Replace class ID with new class ID
                                parts[0] = str(new_class_id)
                                f_out.write(" ".join(parts) + "\n")
                    file_count += 1
                else:
                    print(f"    Warning: Label file not found for {fname}")
                    
            except Exception as e:
                print(f"    Error processing {fname}: {str(e)}")
                continue
                
        print(f"    Copied {file_count} files from {dataset_name}/{split}")
    
    print(f"Finished processing dataset: {dataset_name}")

# Create YAML configuration file
print("\nCreating YAML configuration file...")
yaml_content = f"""train: {os.path.join(base_dir, "images", "train")}
val: {os.path.join(base_dir, "images", "val")}
test: {os.path.join(base_dir, "images", "test")}

nc: 4
names: ['stair', 'people', 'door', 'pothole']
"""

yaml_path = os.path.join(base_dir, "data.yaml")
try:
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"YAML file created successfully at: {yaml_path}")
except Exception as e:
    print(f"Error creating YAML file: {str(e)}")

print("\nDataset merging completed!")
print("Final class mapping:")
print("  0: stair")
print("  1: people")
print("  2: door")
print("  3: pothole")