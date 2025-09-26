import cv2
import numpy as np
import os

def extract_red_mask(rgb_mask_path, diff_thresh=30):
    img = cv2.imread(rgb_mask_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    red = img[:, :, 2]

    # Detect where red is significantly stronger than grayscale
    diff = red.astype(int) - gray.astype(int)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[diff > diff_thresh] = 255

    return mask



def process_and_save_masks(mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(mask_dir):
        if fname.endswith(".png") or fname.endswith(".jpg"):
            bin_mask = extract_red_mask(os.path.join(mask_dir, fname))
            out_path = os.path.join(output_dir, fname)
            cv2.imwrite(out_path, bin_mask)

if __name__ == "__main__":
    process_and_save_masks(
        mask_dir=r"D:\New folder (3)\annotated_unh",
        output_dir=r"D:\New folder (3)\preprocessed"
    )
