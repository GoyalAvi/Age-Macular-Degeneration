import torch
import numpy as np
from model import UNet
from PIL import Image
import cv2
import os

# === CONFIGURATION ===
checkpoint_path = r"D:\Research_Project\distortion predict\STEP_2_LINE_DETECT"
input_folder = r"C:\Users\91975\Downloads\Datensätze Training Avi-selected\diseased eyes"
output_folder = r"D:\final\New folder"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Create output directory if not exists ===
os.makedirs(output_folder, exist_ok=True)

# === Load Model ===
model = UNet(in_channels=1, out_classes=3).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# === Class-wise color map ===
color_map = {
    1: (255, 0, 0),    # Class 1: Blue
    2: (0, 0, 255),    # Class 2: Red
}

# === Recursively loop over all .tif images ===
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.lower().endswith(".tif"):
            image_path = os.path.join(root, filename)

            # Load and preprocess
            image = Image.open(image_path).convert("L")
            image_np = np.array(image, dtype=np.float32) / 255.0
            input_tensor = torch.tensor(image_np).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # Mask out edge 5-pixel columns
            valid_mask = np.ones_like(pred_mask, dtype=bool)
            valid_mask[:, :5] = False
            valid_mask[:, -5:] = False
            pred_mask[~valid_mask] = 0

            # Convert grayscale image to RGB
            image_rgb = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            # Process each class
            for class_id, color in color_map.items():
                class_mask = (pred_mask == class_id).astype(np.uint8)
                if np.any(class_mask):
                    # Faint fill
                    dimmed_color = tuple(int(c * 0.4) for c in color)
                    class_mask_3ch = np.stack([class_mask]*3, axis=-1)
                    image_rgb = np.where(class_mask_3ch == 1,
                                         (0.6 * image_rgb + 0.4 * np.array(dimmed_color, dtype=np.uint8)),
                                         image_rgb).astype(np.uint8)

                    # Outline
                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(class_mask, kernel, iterations=1)
                    outline = cv2.subtract(class_mask, eroded) * 255
                    contours, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(image_rgb, contours, -1, color, thickness=1)

            # Save with folder name in output if needed
            rel_path = os.path.relpath(image_path, input_folder)
            save_name = os.path.splitext(rel_path.replace(os.sep, "_"))[0] + "_overlay.png"
            save_path = os.path.join(output_folder, save_name)
            cv2.imwrite(save_path, image_rgb)
            print(f"Saved prediction: {save_path}")
