import os
import re
import pymupdf  # fitz

POSTERS_DIR = r"C:\projetos\Denoising posters\Denoising posters"
OUTPUT_FILE = r"C:\projetos\MRI_Training\FMImaging_MRI_Denoise\experiments\posters_summary.txt"

# Keywords/Regex to find model architectures, parameters, and losses
MODEL_REGEX = re.compile(r'\b(UNet|NAFNet|Restormer|DRUNet|SCUNet|SwinIR|Transformer|CNN|Diffusion)\b', re.IGNORECASE)
LOSS_REGEX = re.compile(r'\b(L1|L2|MSE|SSIM|MS-SSIM|Charbonnier|HAARpsi|PSNR|Perceptual|Adversarial|Loss)\b', re.IGNORECASE)
PARAM_REGEX = re.compile(r'(batch\s*size|epochs?|learning\s*rate|lr|optimizer|AdamW?)\s*[:=]?\s*([0-9eE.-]+|AdamW?)', re.IGNORECASE)

def extract_from_pdf(filepath):
    try:
        doc = pymupdf.open(filepath)
        text = ""
        # Read the first few pages (most likely to have abstracts/methods) and last pages (conclusion)
        # to save time across 107 PDFs
        num_pages = len(doc)
        pages_to_read = min(num_pages, 5) 
        
        for i in range(pages_to_read):
            text += doc[i].get_text("text") + "\n"
            
        models = list(set(MODEL_REGEX.findall(text)))
        losses = list(set(LOSS_REGEX.findall(text)))
        params = list(set(PARAM_REGEX.findall(text)))
        
        doc.close()
        return models, losses, params
    except Exception as e:
        return ["Error reading"], [str(e)], []

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("==== DENOISING POSTERS SUMMARY ====\n\n")
        
        if not os.path.exists(POSTERS_DIR):
            out.write("Directory not found!\n")
            return
            
        files = [f for f in os.listdir(POSTERS_DIR) if f.lower().endswith('.pdf')]
        out.write(f"Total PDFs found: {len(files)}\n\n")
        
        for file in files:
            path = os.path.join(POSTERS_DIR, file)
            print(f"Processing {file}...")
            m, l, p = extract_from_pdf(path)
            
            out.write(f"--- File: {file} ---\n")
            out.write(f"Models mentioned: {', '.join(m) if m else 'None'}\n")
            out.write(f"Losses mentioned: {', '.join(l) if l else 'None'}\n")
            out.write(f"Hyperparameters: {p if p else 'None'}\n\n")
            
if __name__ == "__main__":
    main()
