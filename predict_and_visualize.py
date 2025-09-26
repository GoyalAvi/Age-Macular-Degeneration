import os
import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from attention_unet import AttentionUNet

image_dir = r'C:\Users\91975\Downloads\AMD\AMD'
model_path = r"D:\auto_encoder\attention_unet_oct.pth"
image_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = AttentionUNet(input_channels=1, output_channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load random image ---
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
random_fname = random.choice(image_files)
img_path = os.path.join(image_dir, random_fname)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (image_size, image_size))
input_img = img_resized.astype(np.float32) / 255.0
input_tensor = torch.tensor(input_img).unsqueeze(0).unsqueeze(0).to(device)

# --- Predict mask ---
with torch.no_grad():
    output = model(input_tensor)
    pred = torch.sigmoid(output).squeeze().cpu().numpy()

# --- Threshold & prepare visuals ---
pred_mask = (pred > 0.4).astype(np.uint8) * 255
pred_colormap = cv2.applyColorMap((pred * 255).astype(np.uint8), cv2.COLORMAP_JET)
img_color = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
overlay = cv2.addWeighted(img_color, 0.7, pred_colormap, 0.3, 0)

# --- Plot everything ---
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.title("OCT Input")
plt.imshow(img_resized, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Predicted Heatmap")
plt.imshow(pred, cmap='hot')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Overlay (OCT + Prediction)")
plt.imshow(overlay)
plt.axis('off')

plt.suptitle(f"Prediction: {random_fname}", fontsize=14)
plt.tight_layout()
plt.show()
