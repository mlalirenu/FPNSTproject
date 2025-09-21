import asyncio
from loguru import logger
from core.nst import style_transfer
from utils.image_io import load_image, save_tensor_img
from evaluation.eval import evaluate
from utils.file_io import save_to_csv

from app.config import DEVICE

async def process_image(content_path, style_path, output_path, alpha, vgg, decoder):
    try:
        logger.info(f"Loading content image: {content_path}")
        content = await load_image(content_path)

        logger.info(f"Loading style image: {style_path}")
        style = await load_image(style_path)

        logger.info("Performing style transfer...")
        output = await style_transfer(vgg, decoder, content, style, alpha=alpha)
        
        logger.info(f"Saving stylized image to: {output_path}")
        await save_tensor_img(output, output_path)

        logger.info("Starting evaluation pipeline")
        stylized_img = await load_image(output_path)
        result = await evaluate(vgg, stylized_img, content, style, output_path, DEVICE)
        save_to_csv(result)
        logger.success("Processing complete.")

    except Exception as e:
        logger.exception(f"Error processing images: {str(e)}")
        raise

async def pipeline(image_jobs, models):
    try:
        vgg, decoder = models  # Unpack the models
        vgg.to(DEVICE)
        decoder.to(DEVICE)
        logger.info("Models are ready, creating processing tasks.")

        tasks = [
            process_image(content, style, output, alpha, vgg, decoder)
            for content, style, output, alpha in image_jobs
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    