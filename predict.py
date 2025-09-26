# plot_predictions.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import OCTLineDataset
from model import UNet
from PIL import Image

# === CONFIGURATION ===
checkpoint_path = "checkpoints/checkpoint_epoch_19.pth"
image_dir = "val/images"
mask_dir = "val/masks"
num_samples = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Model ===
model = UNet(in_channels=1, out_classes=3).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))

# === Load Dataset ===
dataset = OCTLineDataset(image_dir, mask_dir)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# === Visualization Function ===
def plot_sample(image, pred_mask, true_mask, index):
    image = image.squeeze().cpu().numpy()
    pred_mask = pred_mask.squeeze().cpu().numpy()
    true_mask = true_mask.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(pred_mask, cmap="jet", vmin=0, vmax=2)
    axs[1].set_title("Predicted Mask")
    axs[2].imshow(true_mask, cmap="jet", vmin=0, vmax=2)
    axs[2].set_title("Ground Truth Mask")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"prediction_{index}.png")
    plt.show()

# === Run Inference & Plot ===
with torch.no_grad():
    model.eval()
    for idx, (image, mask) in enumerate(loader):
        if idx >= num_samples:
            break

        image = image.to(device)
        outputs = model(image)
        preds = torch.argmax(outputs, dim=1)  # shape: [B, H, W]
        preds = preds.cpu()

        # === Print unique values ===
        print(f"[{idx}] Unique prediction values:", torch.unique(preds))
        print(f"[{idx}] Unique ground truth values:", torch.unique(mask))

        # === Plot ===
        plot_sample(image[0], preds[0], mask[0], idx)
