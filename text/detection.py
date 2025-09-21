import asyncio
import numpy as np
import cv2
import easyocr
import torch

from loguru import logger

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

async def get_text_mask(image: np.ndarray) -> np.ndarray:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _detect_text_sync, image)
    except RuntimeError:
        logger.warning("No running event loop, executing synchronously")
        return _detect_text_sync(image)

def _detect_text_sync(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    text_mask = np.zeros((h, w), dtype=np.uint8)
    
    try:
        results = reader.readtext(image)
        for bbox, _, confidence in results:
            if confidence > 0.5:  # Filter low confidence detections
                pts = np.array(bbox, dtype=np.int32)
                cv2.fillPoly(text_mask, [pts], 255)
    except Exception as e:
        logger.exception(f"Text detection error: {e}")
    
    return text_mask