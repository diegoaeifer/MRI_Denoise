import os
import pydicom
import numpy as np
import argparse
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_batch(file_paths, threshold_range=1000):
    """
    Processes a list of files and returns those matching the criteria.
    """
    problematic = []
    for fp in file_paths:
        try:
            # Use force=True to handle missing headers noted in previous logs
            ds = pydicom.dcmread(fp, force=True)
            if not hasattr(ds, 'PixelData'):
                continue
                
            image = ds.pixel_array.astype(np.float32)
            
            p99 = np.percentile(image, 99)
            p1 = np.percentile(image, 1)
            dynamic_range = p99 - p1
            
            if dynamic_range < threshold_range:
                problematic.append((fp, dynamic_range))
                
        except Exception as e:
            # logger.error(f"Error reading {fp}: {e}")
            pass
    return problematic

def check_background_batched(data_path, threshold_range=1000, batch_size=500, auto_delete=False):
    logger.info(f"Scanning {data_path} (Batch Size: {batch_size}, Auto-Delete: {auto_delete})")
    
    all_dcm_files = []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.lower().endswith('.dcm'):
                all_dcm_files.append(os.path.join(root, file))
    
    total_files = len(all_dcm_files)
    logger.info(f"Total DICOM files found: {total_files}")
    
    problematic_all = []
    deleted_count = 0
    
    # Process in batches
    for i in tqdm(range(0, total_files, batch_size)):
        batch = all_dcm_files[i : i + batch_size]
        found_in_batch = process_batch(batch, threshold_range)
        
        if found_in_batch:
            problematic_all.extend(found_in_batch)
            
            if auto_delete:
                for fp, _ in found_in_batch:
                    try:
                        os.remove(fp)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {fp}: {e}")
        
    logger.info(f"Finished. Found {len(problematic_all)} problematic files.")
    if auto_delete:
        logger.info(f"Successfully deleted {deleted_count} files.")
        
    # Plot top 10 from what was found (even if deleted)
    if problematic_all:
        problematic_all.sort(key=lambda x: x[1]) # Lowest range first
        
        num_plot = min(10, len(problematic_all))
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(len(axes)):
            if i < num_plot:
                path, dr = problematic_all[i]
                try:
                    # Reread if exists, or show text if deleted
                    if os.path.exists(path):
                        ds = pydicom.dcmread(path, force=True)
                        img = ds.pixel_array
                        axes[i].imshow(img, cmap='gray')
                    else:
                        axes[i].text(0.5, 0.5, "DELETED", ha='center', va='center')
                    
                    axes[i].set_title(f"{os.path.basename(path)}\nRange: {dr:.1f}")
                    axes[i].axis('off')
                except:
                    axes[i].axis('off')
            else:
                axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("background_samples.png")
        plt.close()
        logger.info("Saved samples to background_samples.png")

        # Save report
        with open("background_report.txt", "w") as f:
            f.write(f"Background Report (Range Threshold: {threshold_range}, Auto-Deleted: {auto_delete})\n")
            for path, dr in problematic_all:
                status = "Deleted" if auto_delete else "Flagged"
                f.write(f"[{status}] {path}: Range {dr:.2f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=1000, help='Dynamic range threshold')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--delete', action='store_true', help='Delete without confirmation')
    args = parser.parse_args()
    
    check_background_batched(args.data_path, args.threshold, args.batch_size, args.delete)
