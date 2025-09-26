import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Load trained YOLOv8 segmentation model
model = YOLO(r"D:\Research_Project\distortion predict\STEP_3_distortion_predict\best.pt")

# Input and output directories
input_dir = r"C:\Users\91975\Downloads\SwinIR_result (2)"
output_dir = r"D:\New folder (2)\Val\Unhealthy_Annotated"
os.makedirs(output_dir, exist_ok=True)

# List all image files including .tif and .tiff
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]


def merge_boxes_with_confidences(boxes_conf, iou_threshold=0.01):
    """
    boxes_conf: list of (x1, y1, x2, y2, confidence)
    returns: list of (x1, y1, x2, y2, avg_confidence)
    """
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    merged = []
    used = [False] * len(boxes_conf)

    for i in range(len(boxes_conf)):
        if used[i]:
            continue
        x1, y1, x2, y2, conf = boxes_conf[i]
        sum_conf = conf
        count = 1
        for j in range(i + 1, len(boxes_conf)):
            if used[j]:
                continue
            if iou(boxes_conf[i][:4], boxes_conf[j][:4]) > iou_threshold:
                x1 = min(x1, boxes_conf[j][0])
                y1 = min(y1, boxes_conf[j][1])
                x2 = max(x2, boxes_conf[j][2])
                y2 = max(y2, boxes_conf[j][3])
                sum_conf += boxes_conf[j][4]
                count += 1
                used[j] = True
        used[i] = True
        avg_conf = sum_conf / count
        merged.append((x1, y1, x2, y2, avg_conf))
    return merged


# Process each image
for img_name in image_files:
    img_path = os.path.join(input_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run model prediction
    results = model(img_rgb)[0]

    blended = img_rgb.copy()
    boxes_conf = []

    if results.masks is not None:
        for i, mask in enumerate(results.masks.data):
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (img_rgb.shape[1], img_rgb.shape[0]))

            # Red semi-transparent mask
            colored_mask = np.zeros_like(img_rgb)
            colored_mask[:, :, 0] = (mask_resized * 255).astype(np.uint8)
            blended = cv2.addWeighted(blended, 1.0, colored_mask, 0.2, 0)

            # Contours for bounding box
            binary_mask = (mask_resized * 255).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            conf = float(results.boxes.conf[i]) if results.boxes is not None else 1.0  # Default to 1.0 if missing

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes_conf.append((x, y, x + w, y + h, conf))  # (x1, y1, x2, y2, confidence)

    # Merge overlapping/touching boxes and average confidence
    merged_boxes = merge_boxes_with_confidences(boxes_conf, iou_threshold=0.4)


    # Draw merged blue bounding boxes with confidence
    for (x1, y1, x2, y2, conf) in merged_boxes:
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
        label = f"{conf:.2f}"
        cv2.putText(blended, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, img_name)
    cv2.imwrite(output_path, blended_bgr)

print(f"\n✅ Processed {len(image_files)} images. Saved to: {output_dir}")
