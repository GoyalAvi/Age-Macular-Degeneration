# train.py
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from model import UNet
from dataset import OCTLineDataset

# === Paths ===
train_image_dir = "train/images"
train_mask_dir = "train/masks"
val_image_dir = "val/images"
val_mask_dir = "val/masks"

os.makedirs("checkpoints", exist_ok=True)

# === Hyperparameters ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 25
batch_size = 4
lr = 1e-3

# === Datasets & Loaders ===
train_dataset = OCTLineDataset(train_image_dir, train_mask_dir)
val_dataset = OCTLineDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# === Model ===
model = UNet(in_channels=1, out_classes=3).to(device)

# === Loss with Class Weights ===
# Give low weight to background (class 0), normal to RPE (1), PR2 (2)
class_weights = torch.tensor([0.01, 1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# === Optimizer ===
optimizer = optim.Adam(model.parameters(), lr=lr)

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print(f"\n🚀 Epoch {epoch+1}/{num_epochs} starting...")

    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}")

    # === Validation ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == masks).sum().item()
            total += masks.numel()

    avg_val_loss = val_loss / len(val_loader)
    acc = 100 * correct / total
    print(f"📊 Val Loss: {avg_val_loss:.4f}, Accuracy: {acc:.2f}%")

    # === Save Model ===
    save_path = os.path.join("checkpoints", f"checkpoint_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Model saved to: {save_path}")
