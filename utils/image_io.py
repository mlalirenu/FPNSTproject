import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from loguru import logger
import asyncio

from app.config import DEVICE


async def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    return image

def resize(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

async def save_tensor_img(tensor, path):
    logger.info(f"Saving tensor image to {path}")
    tensor = torch.clamp(tensor, 0, 1)
    save_image(tensor, path)
    await asyncio.sleep(0)
    logger.info(f"Image saved: {path}")