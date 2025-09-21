from app.config import PRESERVE_MASK, CONTENT_SIZE, STYLE_SIZE, CROP, DEVICE, CONTENT_COLOR, OUTPUT_DIR

import asyncio
import torch
import torch.nn.functional as F

from utils.image_io import resize
from utils.adain import adaptive_instance_normalization, coral

from core.preservation_mask import FusionMask

from loguru import logger

mask_generator = FusionMask()


async def style_image(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)

    logger.info("Extracting features from content and style...")
    content_f = vgg(content)
    style_f = vgg(style)

    logger.info("Applying adaptive instance normalization and alpha blending...")
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)

    logger.info("Decoding stylized image...")
    stylized = decoder(feat)
    h, w = content.shape[2:]
    stylized = F.interpolate(stylized, size=(h, w), mode='bilinear', align_corners=False)


    if PRESERVE_MASK:
        logger.info("PRESERVE_MASK is enabled — computing preservation mask...")

        mask = await mask_generator.compute_preservation_mask(
            content,
            edge_strength=1.2,
            blur_amount=9,
            face_weight=1.0,
            text_weight=1.0,
            logo_weight=1.0,
            edge_weight=1.0
        )

        logger.info("Blending stylized image with original content using mask...")

        output = mask * content + (1.0 - mask) * stylized

    else:
        logger.info("PRESERVE_MASK is disabled — using full stylization...")
        output = stylized

    await asyncio.sleep(0)
    return output

async def style_transfer(vgg, decoder, content, style, alpha=1.0):
    logger.info(f"Received style transfer request. Alpha: {alpha}")
    logger.info("Resizing content and style images...")

    content_tf = resize(CONTENT_SIZE, CROP)
    style_tf = resize(STYLE_SIZE, CROP)

    content = content_tf(content)
    style = style_tf(style)

    if CONTENT_COLOR:
        logger.info("Applying color correction using CORAL...")
        style = coral(style, content)

    style = style.to(DEVICE).unsqueeze(0)
    content = content.to(DEVICE).unsqueeze(0)

    logger.info("Performing style transfer via style_image()...")
    with torch.no_grad():
        output = await style_image(vgg, decoder, content, style, alpha)

    logger.info("Style transfer complete. Processing output...")
    return output.cpu()
