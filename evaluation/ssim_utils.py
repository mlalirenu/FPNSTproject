from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image
import torch

def compute_ssim(img1, img2):
    # Handle inputs from various sources
    if isinstance(img1, Image.Image):
        img1 = np.array(img1.convert("L"))
    elif isinstance(img1, torch.Tensor):
        img1 = img1.squeeze(0).cpu().detach().numpy() * 255
        img1 = np.uint8(img1.transpose(1, 2, 0))
    if isinstance(img2, Image.Image):
        img2 = np.array(img2.convert("L"))
    elif isinstance(img2, torch.Tensor):
        img2 = img2.squeeze(0).cpu().detach().numpy() * 255
        img2 = np.uint8(img2.transpose(1, 2, 0))

    # Rest of the SSIM calculation
    target_size = (256, 256)
    img1 = Image.fromarray(img1).resize(target_size)
    img2 = Image.fromarray(img2).resize(target_size)
    img1 = np.array(img1.convert("L"))
    img2 = np.array(img2.convert("L"))
    
    return ssim(img1, img2, data_range=255)