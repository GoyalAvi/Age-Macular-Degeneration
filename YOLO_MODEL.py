from ultralytics import YOLO
from multiprocessing import freeze_support
import os
import shutil

from PIL import Image
import cv2

def classify_and_organize_images(input_dir, output_dir):
    model = YOLO(r"C:\Users\91975\Downloads\drive-download-20250811T091251Z-1-001\best.pt")

    healthy_dir = os.path.join(output_dir, "healthy")
    unhealthy_dir = os.path.join(output_dir, "unhealthy")

    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(unhealthy_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                img_path = os.path.join(root, filename)

                try:
                    # Optional: test loading with PIL
                    Image.open(img_path).verify()

                    # Run model prediction
                    result = model(img_path)[0]
                    pred_class = result.names[result.probs.top1]

                    print(f"{img_path} → {pred_class}")

                    if pred_class.lower() == "healthy":
                        shutil.copy(img_path, healthy_dir)
                    elif pred_class.lower() == "unhealthy":
                        shutil.copy(img_path, unhealthy_dir)

                except Exception as e:
                    print(f"❌ Skipping unreadable image: {img_path}\nError: {e}")

    print("✅ All valid images classified and copied.")


def main():
    input_dir = r"D:\Research_Project\Project-Age-Macular-Degeneration-Detection-master\Project-Age-Macular-Degeneration-Detection-master\Unhealthz Person eyes"         # Folder containing images to classify
    output_dir = r"D:\processed\New folder"      # Destination folder for healthy/unhealthy
    classify_and_organize_images(input_dir, output_dir)

if __name__ == '__main__':
    freeze_support()
    main()
