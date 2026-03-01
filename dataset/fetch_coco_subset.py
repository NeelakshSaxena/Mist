import os
import random
import requests
from tqdm import tqdm
import zipfile
import io

def download_coco_subset(num_images=1000, target_dir="dataset/raw/sample"):
    """
    Downloads a subset of COCO 2017 images using the unofficial coco dataset URLs
    We'll actually download the val2017 dataset which is much smaller (1GB, ~5k images)
    and sample 1000 images from it to avoid downloading 25GB training set.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    ANNOTATIONS_URL = "http://images.cocodataset.org/zips/val2017.zip"
    TEMP_ZIP = "dataset/raw/val2017.zip"
    
    os.makedirs("dataset/raw", exist_ok=True)
    
    print(f"Downloading COCO val2017 dataset (~1GB)...")
    
    # Download the ZIP file
    response = requests.get(ANNOTATIONS_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(TEMP_ZIP, 'wb') as file, tqdm(
            desc=TEMP_ZIP,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print("Extracting images...")
    with zipfile.ZipFile(TEMP_ZIP, 'r') as zip_ref:
        # Get list of files, shuffle them, and select num_images
        all_files = [f for f in zip_ref.namelist() if f.endswith('.jpg')]
        sample_files = random.sample(all_files, min(num_images, len(all_files)))
        
        for file in tqdm(sample_files, desc="Extracting samples"):
            zip_ref.extract(file, "dataset/raw/")
            
            # Move from the extracted val2017/xxx.jpg to target_dir/xxx.jpg
            basename = os.path.basename(file)
            os.rename(os.path.join("dataset/raw", file), os.path.join(target_dir, basename))
            
    # Cleanup
    print("Cleaning up temporary files...")
    os.remove(TEMP_ZIP)
    
    # Clean up empty val2017 dir if it exists
    val_dir = "dataset/raw/val2017"
    if os.path.exists(val_dir) and not os.listdir(val_dir):
        os.rmdir(val_dir)
        
    print(f"Successfully downloaded and sampled {len(sample_files)} images to {target_dir}")

if __name__ == "__main__":
    download_coco_subset()
