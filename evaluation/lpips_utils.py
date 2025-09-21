import lpips
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

# --- Unified loader ---
def image_to_tensor(img, size=(256, 256)):
    # handle orientation
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize(size, Image.BICUBIC)

    arr = np.array(img).astype(np.float32) / 255.0   # [0,1]
    arr = arr * 2.0 - 1.0                           # [-1,1]
    ten = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return ten

# --- LPIPS scores ---
def compute_lpips_scores(content_img, style_img, stylized_img, size=(256,256)):
    loss_fn = lpips.LPIPS(net='alex', version='0.1').eval()

    content_tensor   = image_to_tensor(content_img, size)
    style_tensor     = image_to_tensor(style_img, size)
    stylized_tensor  = image_to_tensor(stylized_img, size)

    with torch.no_grad():
        d_content = loss_fn(content_tensor, stylized_tensor).item()
        d_style   = loss_fn(style_tensor, stylized_tensor).item()

    return d_content, d_style
    