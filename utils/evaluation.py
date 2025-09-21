from skimage.metrics import structural_similarity as ssim
import numpy as np
import logging
import asyncio

logger = logging.getLogger(__name__)

async def compute_ssim(imageA, imageB):
    logger.info("Computing SSIM between two images.")
    result = ssim(imageA, imageB, data_range=imageB.max() - imageB.min(), multichannel=True)
    await asyncio.sleep(0)  # Yield control for async compatibility
    logger.info(f"SSIM computed: {result}")
    return result