#!/usr/bin/env python3
# pipeline_dual_unet.py
# Step1: YOLO classify → healthy/unhealthy
# Step2: YOLO segmentation overlays + merged boxes
# Step3: UNet line detection with two checkpoints (healthy vs unhealthy)

from ultralytics import YOLO
from multiprocessing import freeze_support
import os, shutil, cv2, numpy as np, torch
from PIL import Image
from typing import List, Tuple
from model import UNet

# =========================
# Config (edit paths only)
# =========================
# --- Step 1: classification ---
YOLO_CLS_PT = r"/home/avigoyal1/CVGithub/ModelPipeline/best_models/step_1.pt"

# --- Step 2: segmentation ---
YOLO_SEG_PT = r"/home/avigoyal1/CVGithub/ModelPipeline/best_models/step_2.pt"

# --- Step 3: line detection (two UNet checkpoints) ---
UNET_HEALTHY_CKPT   = r"/home/avigoyal1/CVGithub/ModelPipeline/best_models/step_3_healthy.pth"
UNET_UNHEALTHY_CKPT = r"/home/avigoyal1/CVGithub/ModelPipeline/best_models/step_3_unhealthy.pth"

# Visual params for UNet overlays
COLOR_MAP = {1: (255, 0, 0), 2: (0, 0, 255)}  # BGR: 1→blue, 2→red
ALPHA_FILL = 0.40
OUTLINE_THICKNESS = 1
FORCE_MIN_BAND_PX = 2
BORDER_TRIM = 5

# =========================
# Step 1 — classification
# =========================
def classify_and_organize_images(input_dir: str, output_dir: str):
    model = YOLO(YOLO_CLS_PT)
    healthy_dir = os.path.join(output_dir, "healthy")
    unhealthy_dir = os.path.join(output_dir, "unhealthy")
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(unhealthy_dir, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.lower().endswith(exts):
                continue
            img_path = os.path.join(root, filename)
            try:
                Image.open(img_path).verify()
                result = model(img_path)[0]
                pred_class = result.names[result.probs.top1]
                print(f"{img_path} → {pred_class}")
                if pred_class.lower() == "healthy":
                    shutil.copy(img_path, healthy_dir)
                elif pred_class.lower() == "unhealthy":
                    shutil.copy(img_path, unhealthy_dir)
            except Exception as e:
                print(f"Skipping unreadable: {img_path}\nError: {e}")

    print("Step 1 complete.")

# =========================
# Step 2 — segmentation
# =========================
def merge_boxes_with_confidences(boxes_conf: List[Tuple[int,int,int,int,float]], iou_threshold=0.4):
    def iou(a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (a[2]-a[0]) * (a[3]-a[1])
        areaB = (b[2]-b[0]) * (b[3]-b[1])
        return inter / float(areaA + areaB - inter)

    merged, used = [], [False]*len(boxes_conf)
    for i in range(len(boxes_conf)):
        if used[i]:
            continue
        x1, y1, x2, y2, conf = boxes_conf[i]
        sum_conf, cnt = conf, 1
        for j in range(i+1, len(boxes_conf)):
            if used[j]:
                continue
            if iou(boxes_conf[i][:4], boxes_conf[j][:4]) > iou_threshold:
                x1 = min(x1, boxes_conf[j][0]); y1 = min(y1, boxes_conf[j][1])
                x2 = max(x2, boxes_conf[j][2]); y2 = max(y2, boxes_conf[j][3])
                sum_conf += boxes_conf[j][4]; cnt += 1
                used[j] = True
        used[i] = True
        merged.append((x1, y1, x2, y2, sum_conf/cnt))
    return merged

def run_segmentation(input_dir: str, output_dir: str):
    model = YOLO(YOLO_SEG_PT)
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    for root, _, files in os.walk(input_dir):
        for img_name in files:
            if not img_name.lower().endswith(image_extensions):
                continue
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Unreadable image: {img_path}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = model(img_rgb)[0]

            blended = img_rgb.copy()
            boxes_conf = []

            if results.masks is not None:
                for i, mask in enumerate(results.masks.data):
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (img_rgb.shape[1], img_rgb.shape[0]))
                    colored_mask = np.zeros_like(img_rgb)
                    colored_mask[:, :, 0] = (mask_resized * 255).astype(np.uint8)
                    blended = cv2.addWeighted(blended, 1.0, colored_mask, 0.2, 0)

                    binary_mask = (mask_resized * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    conf = float(results.boxes.conf[i]) if results.boxes is not None else 1.0
                    for cnt in contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        boxes_conf.append((x, y, x+w, y+h, conf))

            for (x1, y1, x2, y2, conf) in merge_boxes_with_confidences(boxes_conf, iou_threshold=0.4):
                cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(blended, f"{conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            out = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, img_name), out)

    print(f"Step 2 complete: {input_dir}")

# =========================
# Step 3 — dual UNet line detection
# =========================
def _ensure_min_band(mask_bin: np.ndarray, min_band_px: int) -> np.ndarray:
    if min_band_px <= 1:
        return mask_bin
    k_h = max(3, min_band_px + 1)
    kernel = np.ones((k_h, 1), np.uint8)  # vertical dilation only
    return cv2.dilate(mask_bin, kernel, iterations=1)

def _load_unet(ckpt_path: str, device: torch.device) -> UNet:
    model = UNet(in_channels=1, out_classes=3).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

def _run_unet_folder(model: UNet, in_dir: str, out_dir: str, device: torch.device):
    os.makedirs(out_dir, exist_ok=True)
    for fn in os.listdir(in_dir):
        if not fn.lower().endswith(".tif"):
            continue
        p = os.path.join(in_dir, fn)
        im = Image.open(p).convert("L")
        im_np = np.array(im, dtype=np.float32) / 255.0
        inp = torch.from_numpy(im_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)            # softmax omitted since argmax invariant
            pred = torch.argmax(logits, 1).squeeze().cpu().numpy().astype(np.uint8)

        # trim left/right borders
        pred[:, :BORDER_TRIM] = 0
        pred[:, -BORDER_TRIM:] = 0

        rgb = cv2.cvtColor((im_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for cls, color in COLOR_MAP.items():
            cls_mask = (pred == cls).astype(np.uint8)
            if not cls_mask.any():
                continue

            cls_mask_band = _ensure_min_band(cls_mask, FORCE_MIN_BAND_PX)

            dim = np.array([int(c * ALPHA_FILL) for c in color], dtype=np.uint8)
            mask3 = np.repeat(cls_mask_band[:, :, None], 3, axis=2)
            rgb = np.where(mask3 == 1, (0.6*rgb + 0.4*dim).astype(np.uint8), rgb)

            kernel3 = np.ones((3, 3), np.uint8)
            er = cv2.erode(cls_mask_band, kernel3, 1)
            outline = ((cls_mask_band - er) > 0).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts:
                cv2.drawContours(rgb, cnts, -1, color, thickness=OUTLINE_THICKNESS)

        outp = os.path.join(out_dir, os.path.splitext(fn)[0] + "_overlay.png")
        cv2.imwrite(outp, rgb)
        print("saved:", outp)

def run_dual_unet(step1_output_dir: str, step3_output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Folders created by Step 1
    healthy_in  = os.path.join(step1_output_dir, "healthy")
    unhealthy_in = os.path.join(step1_output_dir, "unhealthy")

    # Output folders
    healthy_out  = os.path.join(step3_output_dir, "healthy_unet")
    unhealthy_out = os.path.join(step3_output_dir, "unhealthy_unet")
    os.makedirs(step3_output_dir, exist_ok=True)

    # Load models
    if os.path.isdir(healthy_in) and len(os.listdir(healthy_in)) > 0:
        print("Loading UNet (healthy)...")
        m_h = _load_unet(UNET_HEALTHY_CKPT, device)
        _run_unet_folder(m_h, healthy_in, healthy_out, device)
    else:
        print("No healthy inputs found. Skipping healthy UNet.")

    if os.path.isdir(unhealthy_in) and len(os.listdir(unhealthy_in)) > 0:
        print("Loading UNet (unhealthy)...")
        m_u = _load_unet(UNET_UNHEALTHY_CKPT, device)
        _run_unet_folder(m_u, unhealthy_in, unhealthy_out, device)
    else:
        print("No unhealthy inputs found. Skipping unhealthy UNet.")

    print("Step 3 complete.")

# =========================
# Main
# =========================
def main():
    # --- Inputs/Outputs ---
    step1_input  = r"/home/avigoyal1/CVGithub/ModelPipeline/Images/Healthy/004"
    step1_output = r"/home/avigoyal1/CVGithub/ModelPipeline/Output_Segregation_folder"   # will contain healthy/ and unhealthy/
    step2_output = r"/home/avigoyal1/CVGithub/ModelPipeline/Output1"
    step3_output = r"/home/avigoyal1/CVGithub/ModelPipeline/Output2"

    # Step 1
    classify_and_organize_images(step1_input, step1_output)

    # Step 2: run segmentation on both healthy and unhealthy trees from Step1
    run_segmentation(step1_output, step2_output)

    # Step 3: dual UNet by class
    run_dual_unet(step1_output, step3_output)

if __name__ == "__main__":
    freeze_support()
    main()
