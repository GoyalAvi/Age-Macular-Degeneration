
import os
import numpy as np
import cv2
from PIL import Image

# === CONFIGURATION ===
image_folder = r"C:\Users\91975\Desktop\train\images"       # e.g., "images/"
annotation_folder = r"C:\Users\91975\Desktop\train\labels"    # e.g., "annotations/"
mask_output_folder = r"C:\Users\91975\Desktop\train\masks"    # e.g., "output/masks/"
overlay_output_folder = r"C:\Users\91975\Desktop\train\overlays"

os.makedirs(mask_output_folder, exist_ok=True)
os.makedirs(overlay_output_folder, exist_ok=True)

def draw_line_on_mask(y_coords, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)
    width = len(y_coords)

    for x in range(width - 1):
        try:
            y1 = int(round(float(y_coords[x])))
            y2 = int(round(float(y_coords[x + 1])))
        except:
            continue

        y1 = np.clip(y1, 0, image_shape[0] - 1)
        y2 = np.clip(y2, 0, image_shape[0] - 1)

        pt1 = (x, y1)
        pt2 = (x + 1, y2)
        cv2.line(mask, pt1, pt2, color=255, thickness=1)

    return mask

# === MAIN LOOP ===
for filename in os.listdir(image_folder):
    if filename.endswith(".tif"):
        image_path = os.path.join(image_folder, filename)
        base_name = os.path.splitext(filename)[0]
        annotation_path = os.path.join(annotation_folder, base_name + ".txt")

        if not os.path.exists(annotation_path):
            print(f"[Missing] Annotation for {filename}")
            continue

        # Load image and get shape
        image = Image.open(image_path).convert("L")
        image_np = np.array(image)
        h, w = image_np.shape

        # Read annotation
        with open(annotation_path, "r") as file:
            lines = file.readlines()
            if len(lines) < 2:
                print(f"[Invalid] Annotation format in {annotation_path}")
                continue

            rpe_y = list(map(float, lines[0].strip().split(",")))
            pr2_y = list(map(float, lines[1].strip().split(",")))

            # Handle coordinate length mismatch
            if len(rpe_y) != w or len(pr2_y) != w:
                print(f"[Mismatch] {base_name}: rpe={len(rpe_y)}, pr2={len(pr2_y)}, image width={w}")
                min_len = min(len(rpe_y), len(pr2_y), w)
                rpe_y = rpe_y[:min_len]
                pr2_y = pr2_y[:min_len]
                image_np = image_np[:, :min_len]
                h, w = image_np.shape

            # Scale if needed
            if max(rpe_y) > h:
                scale = h / max(rpe_y)
                rpe_y = [int(y * scale) for y in rpe_y]
            else:
                rpe_y = [int(y) for y in rpe_y]

            if max(pr2_y) > h:
                scale = h / max(pr2_y)
                pr2_y = [int(y * scale) for y in pr2_y]
            else:
                pr2_y = [int(y) for y in pr2_y]

            # Create binary line masks
            mask_rpe = draw_line_on_mask(rpe_y, (h, w))
            mask_pr2 = draw_line_on_mask(pr2_y, (h, w))

            # Combine both lines into a labeled mask
            combined_mask = np.zeros((h, w), dtype=np.uint8)
            combined_mask[mask_rpe == 255] = 1  # RPE line = 1
            combined_mask[mask_pr2 == 255] = 2  # PR2 line = 2

            # === Save true mask (.npy) for training ===
            npy_mask_path = os.path.join(mask_output_folder, base_name + "_mask.npy")
            np.save(npy_mask_path, combined_mask)

            # === Save visible mask (.png) ===
            visible_mask = np.zeros_like(combined_mask, dtype=np.uint8)
            visible_mask[combined_mask == 1] = 127
            visible_mask[combined_mask == 2] = 255
            mask_png_path = os.path.join(mask_output_folder, base_name + "_mask_vis.png")
            cv2.imwrite(mask_png_path, visible_mask)

            # === Create and save overlay image ===
            overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            overlay[mask_rpe == 255] = (0, 255, 0)   # Green = RPE
            overlay[mask_pr2 == 255] = (0, 0, 255)   # Red = PR2
            overlay_path = os.path.join(overlay_output_folder, base_name + "_overlay.png")
            cv2.imwrite(overlay_path, overlay)

            print(f"[✓] Saved: {base_name}_mask.npy, _mask_vis.png, _overlay.png")
