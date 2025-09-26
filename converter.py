import os
import shutil

# === Paths ===
original_label_dir = r"D:\Research_Project\New folder (3)\dataset\labels\Train"
output_label_dir = r"D:\Research_Project\New folder (3)\dataset\labels\Train"  # Same folder, just adding files

# === Enhanced suffixes to replicate for ===
suffixes = ["_out", "_SwinIR", "_SwinIR_large"]

# === Go through all .txt files ===
for fname in os.listdir(original_label_dir):
    if not fname.endswith(".txt"):
        continue

    base_name = fname[:-4]  # remove .txt
    original_path = os.path.join(original_label_dir, fname)

    for suffix in suffixes:
        new_name = base_name + suffix + ".txt"
        new_path = os.path.join(output_label_dir, new_name)

        if not os.path.exists(new_path):
            shutil.copyfile(original_path, new_path)
            print(f"Copied: {new_name}")
        else:
            print(f"Already exists: {new_name}")
