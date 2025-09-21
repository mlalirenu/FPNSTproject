import cv2
import numpy as np
import torch
import asyncio
import threading

from loguru import logger
from face.pipeline import get_face_mask
from text.detection import get_text_mask
from logo.detection import get_logo_mask  # Using the fixed version
from edge.advanced_fusion import AdvancedEdgeFusion
from app.config import PRESERVE_FACE, PRESERVE_TEXT, PRESERVE_LOGO, PRESERVE_EDGES, BLUR_AMOUNT

class FusionMask:

    def __init__(self):
        self.edge_fuser = AdvancedEdgeFusion()
        # FIXED: Don't create semaphore in __init__, create it when needed
        self._semaphore = None
        self._semaphore_lock = threading.Lock()

    def _get_semaphore(self):
        # Get or create a semaphore for the current event loop
        with self._semaphore_lock:
            try:
                # Check if we have a semaphore and if it's still valid
                if self._semaphore is not None:
                    # Try to get the current event loop
                    current_loop = asyncio.get_event_loop()
                    # If semaphore is bound to a different loop, recreate it
                    if hasattr(self._semaphore, '_loop') and self._semaphore._loop != current_loop:
                        self._semaphore = None
                
                # Create new semaphore if needed
                if self._semaphore is None:
                    self._semaphore = asyncio.Semaphore(2)
                    logger.debug("Created new semaphore for current event loop")
                    
            except RuntimeError:
                # No event loop running, will be created later
                self._semaphore = asyncio.Semaphore(2)
                logger.debug("Created semaphore outside event loop")
            
            return self._semaphore

    async def detect_edges(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        
        # Async wrapper for advanced edge detection
        
        try:
            mask = await self.edge_fuser.detect_edges_async(image, strength=strength)
            expected_shape = image.shape[:2]
            if mask.shape != expected_shape:
                logger.error(f"Edge mask shape mismatch: expected {expected_shape}, got {mask.shape}")
                mask = np.zeros(expected_shape, dtype=np.uint8)
            return mask
        except Exception as e:
            logger.exception(f"Edge detection failed: {str(e)}")
            h, w = image.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    def _validate_mask_shape(self, mask: np.ndarray, expected_shape: tuple, mask_name: str) -> np.ndarray:
        
        # Validate and fix mask shape if necessary
        
        if mask.shape != expected_shape:
            logger.warning(f"{mask_name} mask shape mismatch: expected {expected_shape}, got {mask.shape}")
            try:
                if len(mask.shape) == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]))
            except Exception as e:
                logger.error(f"Failed to fix {mask_name} mask shape: {e}")
                mask = np.zeros(expected_shape, dtype=np.uint8)

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        return mask

    async def compute_preservation_mask(
        self,
        image_tensor: torch.Tensor,
        face_method: str = 'precise',
        include_ears: bool = True,
        edge_strength: float = 1.0,
        blur_amount: int = 9,
        face_weight: float = 1.0,
        text_weight: float = 1.0,
        logo_weight: float = 1.0,
        edge_weight: float = 1.0,
        logo_text_prompt: str = "logo . brand . company logo . trademark",
        logo_box_threshold: float = 0.35,
        logo_text_threshold: float = 0.25,
        logo_enhance_mask: bool = True
    ) -> torch.Tensor:
        # FIXED: Get semaphore safely for current event loop
        semaphore = self._get_semaphore()
        
        # Use semaphore to limit concurrent computations
        async with semaphore:
            return await self._compute_preservation_mask_impl(
                image_tensor, face_method, include_ears, edge_strength, blur_amount,
                face_weight, text_weight, logo_weight, edge_weight,
                logo_text_prompt, logo_box_threshold, logo_text_threshold, logo_enhance_mask
            )

    async def _compute_preservation_mask_impl(
        self,
        image_tensor: torch.Tensor,
        face_method: str = 'precise',
        include_ears: bool = True,
        edge_strength: float = 1.0,
        blur_amount: int = 9,
        face_weight: float = 1.0,
        text_weight: float = 1.0,
        logo_weight: float = 1.0,
        edge_weight: float = 1.0,
        logo_text_prompt: str = "logo . brand . company logo . trademark",
        logo_box_threshold: float = 0.35,
        logo_text_threshold: float = 0.25,
        logo_enhance_mask: bool = True
    ) -> torch.Tensor:
        try:
            if image_tensor.dim() == 4:
                if image_tensor.shape[0] != 1:
                    logger.warning(f"Unexpected batch size: {image_tensor.shape[0]}, using first image")
                image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            elif image_tensor.dim() == 3:
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            else:
                raise ValueError(f"Invalid tensor dimensions: {image_tensor.shape}")

            image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
            if image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            h, w = image_np.shape[:2]
            logger.debug(f"Processing preservation mask for image: {h}x{w}")

            # Initialize all masks
            face_mask = np.zeros((h, w), dtype=np.uint8)
            text_mask = np.zeros((h, w), dtype=np.uint8)
            logo_mask = np.zeros((h, w), dtype=np.uint8)
            edge_mask = np.zeros((h, w), dtype=np.uint8)

            detection_tasks = []
            task_mapping = []

            if PRESERVE_FACE:
                detection_tasks.append(get_face_mask(image_np, face_method, include_ears))
                task_mapping.append('face')

            if PRESERVE_TEXT:
                detection_tasks.append(get_text_mask(image_np))
                task_mapping.append('text')

            if PRESERVE_LOGO:
                detection_tasks.append(get_logo_mask(
                    image_np,
                    text_prompt=logo_text_prompt,
                    box_threshold=logo_box_threshold,
                    text_threshold=logo_text_threshold,
                    enhance_mask=logo_enhance_mask
                ))
                task_mapping.append('logo')

            if PRESERVE_EDGES:
                detection_tasks.append(self.detect_edges(image_np, edge_strength))
                task_mapping.append('edge')

            if detection_tasks:
                logger.debug(f"Running {len(detection_tasks)} detection tasks: {task_mapping}")
                
                # FIXED: Process tasks with proper error handling and cleanup
                try:
                    results = await asyncio.gather(*detection_tasks, return_exceptions=True)
                    
                    # Small delay to help with GPU memory management
                    await asyncio.sleep(0.01)
                    
                except Exception as gather_error:
                    logger.error(f"Task gathering failed: {gather_error}")
                    results = [np.zeros((h, w), dtype=np.uint8) for _ in detection_tasks]

                for i, (result, task_type) in enumerate(zip(results, task_mapping)):
                    if isinstance(result, Exception):
                        logger.error(f"{task_type} detection failed: {result}")
                        result = np.zeros((h, w), dtype=np.uint8)

                    result = self._validate_mask_shape(result, (h, w), task_type)

                    if task_type == 'face':
                        face_mask = result
                    elif task_type == 'text':
                        text_mask = result
                    elif task_type == 'logo':
                        logo_mask = result
                    elif task_type == 'edge':
                        edge_mask = result

            # Ensure all masks have correct shape
            face_mask = self._validate_mask_shape(face_mask, (h, w), "face")
            text_mask = self._validate_mask_shape(text_mask, (h, w), "text")
            logo_mask = self._validate_mask_shape(logo_mask, (h, w), "logo")
            edge_mask = self._validate_mask_shape(edge_mask, (h, w), "edge")

            # Apply weights
            face_mask = (face_mask.astype(np.float32) / 255.0) * face_weight
            text_mask = (text_mask.astype(np.float32) / 255.0) * text_weight
            logo_mask = (logo_mask.astype(np.float32) / 255.0) * logo_weight
            edge_mask = (edge_mask.astype(np.float32) / 255.0) * edge_weight

            # Combine masks
            combined = np.maximum.reduce([face_mask, text_mask, logo_mask, edge_mask])

            if BLUR_AMOUNT > 0:
                kernel_size = BLUR_AMOUNT if BLUR_AMOUNT % 2 == 1 else BLUR_AMOUNT + 1
                combined = cv2.GaussianBlur(combined, (kernel_size, kernel_size), 0)

            combined = np.clip(combined, 0, 1)

            if combined.shape != (h, w):
                logger.error(f"Final combined mask shape error: expected ({h}, {w}), got {combined.shape}")
                combined = np.zeros((h, w), dtype=np.float32)

            combined_tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            expected_tensor_shape = (1, 1, h, w)
            if combined_tensor.shape != expected_tensor_shape:
                logger.error(f"Final tensor shape error: expected {expected_tensor_shape}, got {combined_tensor.shape}")
                combined_tensor = torch.zeros(expected_tensor_shape, dtype=torch.float32)

            logger.info(f"Preservation mask completed: shape {combined_tensor.shape}, preserved pixels: {torch.sum(combined_tensor > 0.1).item()}")

            # Ensure tensor is on the correct device
            return combined_tensor.to(image_tensor.device)

        except Exception as e:
            logger.error(f"Preservation mask computation failed: {e}")
            # Force cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            h, w = image_tensor.shape[-2:]
            return torch.zeros((1, 1, h, w), dtype=torch.float32, device=image_tensor.device)