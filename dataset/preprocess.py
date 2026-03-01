import cv2
import os
from tqdm import tqdm

SOURCE = "dataset/raw/sample"
TARGET = "dataset/resized"

os.makedirs(TARGET, exist_ok=True)

images = [f for f in os.listdir(SOURCE) if f.endswith(".jpg") or f.endswith(".png")]

print(f"Resizing {len(images)} images from {SOURCE} to {TARGET}...")

for img_name in tqdm(images):
    img_path = os.path.join(SOURCE, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read image: {img_name}")
        continue

    h, w = img.shape[:2]
    
    # Calculate scale so the longest side is 512
    scale = 512 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(os.path.join(TARGET, img_name), resized)

print(f"Resizing complete. Output saved to {TARGET}")
