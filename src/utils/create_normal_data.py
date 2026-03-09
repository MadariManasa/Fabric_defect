import os
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def extract_normal_patches(src_dir, dst_dir, patch_size=512, patches_per_image=2):
    """
    Extract patches from the corners of images in src_dir and save them to dst_dir.
    Assuming corners are likely to be defect-free.
    """
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    
    for root, _, files in os.walk(src_dir):
        if "Normal" in root: continue # Skip if already in Normal
        
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is None: continue
                
                h, w = img.shape[:2]
                
                # Extract top-left corner
                patch1 = img[0:patch_size, 0:patch_size]
                # Extract bottom-right corner
                patch2 = img[h-patch_size:h, w-patch_size:w]
                
                cv2.imwrite(os.path.join(dst_dir, f"normal_{file}_tl.jpg"), patch1)
                cv2.imwrite(os.path.join(dst_dir, f"normal_{file}_br.jpg"), patch2)
                
                count += 2
                if count % 20 == 0:
                    print(f"Extracted {count} patches...")

    print(f"Finished! Extracted {count} patches to {dst_dir}")

if __name__ == "__main__":
    src = "data/captured"
    dst = "data/captured/Normal"
    extract_normal_patches(src, dst)
