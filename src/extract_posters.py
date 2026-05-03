import os
import re
import pymupdf  # fitz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

POSTERS_DIR = os.path.join("data", "posters")
OUTPUT_FILE = os.path.join("experiments", "posters_summary.txt")

# Keywords/Regex to find model architectures, parameters, and losses
MODEL_REGEX = re.compile(
    r"\b(UNet|NAFNet|Restormer|DRUNet|SCUNet|SwinIR|Transformer|CNN|Diffusion)\b",
    re.IGNORECASE,
)
LOSS_REGEX = re.compile(
    r"\b(L1|L2|MSE|SSIM|MS-SSIM|Charbonnier|HAARpsi|PSNR|Perceptual|Adversarial|Loss)\b",
    re.IGNORECASE,
)
PARAM_REGEX = re.compile(
    r"(batch\s*size|epochs?|learning\s*rate|lr|optimizer|AdamW?)\s*[:=]?\s*([0-9eE.-]+|AdamW?)",
    re.IGNORECASE,
)


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


def process_file(file):
    path = os.path.join(POSTERS_DIR, file)
    print(f"Processing {file}...")
    m, ll, p = extract_from_pdf(path)
    return file, m, ll, p


def main(use_threads=False):
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    if not os.path.exists(POSTERS_DIR):
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
            out.write("==== DENOISING POSTERS SUMMARY ====\n\n")
            out.write("Directory not found!\n")
        return

    files = sorted([f for f in os.listdir(POSTERS_DIR) if f.lower().endswith(".pdf")])

    results = []
    # Use ProcessPoolExecutor for parallel processing by default
    # Fallback to ThreadPoolExecutor if requested (useful for tests with mocks)
    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with ExecutorClass() as executor:
        results = list(executor.map(process_file, files))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("==== DENOISING POSTERS SUMMARY ====\n\n")
        out.write(f"Total PDFs found: {len(files)}\n\n")

        for file, m, ll, p in results:
            out.write(f"--- File: {file} ---\n")
            out.write(f"Models mentioned: {', '.join(m) if m else 'None'}\n")
            out.write(f"Losses mentioned: {', '.join(ll) if ll else 'None'}\n")
            out.write(f"Hyperparameters: {p if p else 'None'}\n\n")


if __name__ == "__main__":
    main()
