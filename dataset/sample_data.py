import os
import random
import shutil

SOURCE = "dataset/raw/train2017" # This will be the path after extraction, might need to adjust based on Kaggle output format
TARGET = "dataset/raw/sample"
N = 5000

print(f"Sampling {N} images from {SOURCE} to {TARGET}...")

os.makedirs(TARGET, exist_ok=True)

try:
    images = [f for f in os.listdir(SOURCE) if f.endswith(".jpg")]
    
    # In case there are fewer than N images
    sample_size = min(N, len(images))
    sample = random.sample(images, sample_size)
    
    for img in sample:
        shutil.copy(os.path.join(SOURCE, img),
                    os.path.join(TARGET, img))
                    
    print(f"Successfully sampled {sample_size} images.")
except Exception as e:
    print(f"Error during sampling: {e}")
