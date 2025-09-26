import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from PIL import Image

# --- Configuration ---
INPUT_ROOT = r"/home/avigoyal1/CVGithub/Model/Images/Unhealthy/AMD.001"
OUTPUT_ROOT = r"/home/avigoyal1/CVGithub/ModelPipeline/Images/Unhealthy_Output"
SUBSETS = ['train', 'val']
TARGET_LAYERS = ['RPE', 'PR2']
IMAGE_WIDTH = 512


def parse_all_layers(xml_path):
    """
    Parses an XML file to extract ONLY the target layer segmentations for each image.
    Returns a dict: {filename: {layer_name: [y_values]}}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_segments = {}

    for image in root.findall('.//Image'):
        exam_url_elem = image.find('./ImageData/ExamURL')
        if exam_url_elem is None or not exam_url_elem.text:
            continue

        filename = os.path.basename(exam_url_elem.text.strip())
        segs = {}

        segmentation = image.find('./Segmentation')
        if segmentation is None:
            continue

        for segline in segmentation.findall('./SegLine'):
            name_elem = segline.find('./Name')
            array_elem = segline.find('./Array')

            if name_elem is None or array_elem is None or not array_elem.text:
                continue

            name = name_elem.text.strip()
            if name not in TARGET_LAYERS:
                continue

            points = array_elem.text.strip().split()
            y_vals = [float(p) if p != '3e+038' else -1.0 for p in points]

            if len(y_vals) != IMAGE_WIDTH:
                y_vals = [-1.0] * IMAGE_WIDTH

            segs[name] = y_vals

        # Only keep entries that contain both RPE and PR2
        if all(layer in segs for layer in TARGET_LAYERS):
            image_segments[filename] = segs

    return image_segments


def visualize_random_lines(output_root, subset='train', num_images=5):
    label_dir = os.path.join(output_root, subset, "labels")
    image_dir = os.path.join(output_root, subset, "images")

    if not os.path.exists(label_dir) or not os.path.exists(image_dir):
        print(f"⚠️ Directories for subset '{subset}' not found. Cannot visualize.")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    if len(label_files) < num_images:
        print(f"⚠️ Only found {len(label_files)} label files in '{subset}'. Visualizing all.")
        num_images = len(label_files)

    if num_images == 0:
        print(f"No label files found in {label_dir} to visualize.")
        return

    sample_files = random.sample(label_files, num_images)

    for file in sample_files:
        base = file.replace(".txt", "")
        label_path = os.path.join(label_dir, file)
        image_path = os.path.join(image_dir, base + ".tif")

        if not os.path.exists(image_path):
            print(f"⚠️ Image file not found for label '{file}': {image_path}")
            continue

        label_array = np.loadtxt(label_path, delimiter=',')
        image = Image.open(image_path).convert('L')
        image_np = np.array(image)

        plt.figure(figsize=(10, 4))
        plt.imshow(image_np, cmap='gray')
        height, width = image_np.shape

        if label_array.ndim == 1:
            label_array = label_array[np.newaxis, :]

        if label_array.shape[1] != width:
            print(f"⚠️ Skipping visualization for {file} due to shape mismatch.")
            plt.close()
            continue

        colors = ['red', 'blue']
        for i in range(label_array.shape[0]):
            y_vals = label_array[i]
            valid_mask = y_vals != -1.0
            x_vals = np.arange(width)
            plt.plot(x_vals[valid_mask], y_vals[valid_mask], color=colors[i], label=TARGET_LAYERS[i])

        plt.title(f"{base} - {', '.join(TARGET_LAYERS)}")
        plt.legend()
        plt.ylim(height, 0)
        plt.tight_layout()
        plt.show()


def main():
    for subset in SUBSETS:
        os.makedirs(os.path.join(OUTPUT_ROOT, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, subset, "labels"), exist_ok=True)

    image_label_pairs = []
    skipped_due_to_layers = 0
    total_images_found = 0

    if not os.path.isdir(INPUT_ROOT):
        print(f"🚨 Error: Input directory does not exist:\n  -> {INPUT_ROOT}")
        return

    print(f"🔍 Searching for XML files in: {INPUT_ROOT}")
    xml_files_found = False
    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files_found = True
                xml_path = os.path.join(root, file)
                print(f"\nProcessing XML file: {os.path.basename(xml_path)}")
                segmentations = parse_all_layers(xml_path)

                tif_map = {f: f for f in os.listdir(root) if f.lower().endswith(".tif")}

                for fname, layers in tqdm(segmentations.items(), desc="  Matching data"):
                    total_images_found += 1

                    if fname not in tif_map:
                        continue

                    tif_path = os.path.join(root, tif_map[fname])
                    save_name = os.path.splitext(tif_map[fname])[0]

                    try:
                        layer_array = [layers[layer_name] for layer_name in TARGET_LAYERS]
                        layer_array = np.array(layer_array, dtype=np.float32)
                        image_label_pairs.append((tif_path, layer_array, save_name))
                    except KeyError:
                        skipped_due_to_layers += 1
                        continue

    if not xml_files_found:
        print("\n❌ No XML files were found in the input directory.")
        return

    if not image_label_pairs:
        print("\n❌ No valid image-label pairs found with both RPE and PR2.")
        return

    # Split
    random.seed(42)
    random.shuffle(image_label_pairs)
    split_idx = int(0.7 * len(image_label_pairs))
    splits = {
        'train': image_label_pairs[:split_idx],
        'val': image_label_pairs[split_idx:]
    }

    total_processed = 0
    for subset, data in splits.items():
        if not data:
            continue
        print(f"\nProcessing {subset} set...")
        for img_path, label_array, save_name in tqdm(data, desc=f"  Saving {subset} files"):
            try:
                shutil.copyfile(img_path, os.path.join(OUTPUT_ROOT, subset, "images", f"{save_name}.tif"))
                txt_path = os.path.join(OUTPUT_ROOT, subset, "labels", f"{save_name}.txt")
                np.savetxt(txt_path, label_array, fmt='%.6f', delimiter=',')
                total_processed += 1
            except Exception as e:
                print(f"🚨 Error while processing {save_name}: {e}")

    print(f"\n✅ Done. Processed {total_processed} images.")
    print(f"   Train: {len(splits['train'])}, Val: {len(splits['val'])}")
    print(f"📉 Skipped images due to missing RPE or PR2: {skipped_due_to_layers}")
    print(f"📊 Total valid image-label pairs: {len(image_label_pairs)}")

    print("\n👁️ Visualizing 5 random training samples...")
    visualize_random_lines(OUTPUT_ROOT, subset='train', num_images=5)


if __name__ == "__main__":
    main()
