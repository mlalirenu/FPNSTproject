from pathlib import Path
import torch
import os
import csv

# Resolve the base directory as the package root, not this file's directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def from_base(*path_parts):
    # Join paths relative to nst_batch_styler base directory.
    return os.path.join(BASE_DIR, *path_parts)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VGG_PATH = from_base("models", "vgg_normalised.pth")
DECODER_PATH = from_base("models", "decoder.pth")
LOG_PATH = from_base("log", "evaluation_results.csv")

PRESERVE_FACE = True
PRESERVE_TEXT = True
PRESERVE_LOGO = True
PRESERVE_EDGES = True

BLUR_AMOUNT = 0 #9 - Default

PRESERVE_MASK = any([PRESERVE_FACE, PRESERVE_TEXT, PRESERVE_LOGO, PRESERVE_EDGES])  # Whether to apply the preservation mask

CONTENT_SIZE = 512
STYLE_SIZE = 512
CROP = False
CONTENT_COLOR = False  # Whether to preserve content color in style transfer
DEFAULT_ALPHA = 1.0
MAX_BATCH_SIZE = 20
DEFAULT_BATCH_SIZE = 5
OUTPUT_FOLDER = "outputs"

# Convert to Path before mkdir
OUTPUT_DIR = Path(from_base(OUTPUT_FOLDER))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def check_and_create_csv(file_path):
    headers = [
        "Timestamp",
        "Image Name",
        "Content Loss (Raw)",
        "Content Similarity % (Range: 0-100)",
        "Style Loss (Raw)",
        "Style Similarity % (Range: 0-100)",
        "LPIPS Content",
        "LPIPS Style",
        "SSIM Content"
    ]
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(headers)
        except IOError as e:
            print(f"An error occurred while creating the file: {e}")
check_and_create_csv(LOG_PATH)