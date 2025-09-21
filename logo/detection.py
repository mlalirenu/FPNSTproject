import cv2
import numpy as np
import torch
import asyncio
from typing import Tuple
from PIL import Image
import os
from loguru import logger
import threading
import asyncio

# Import GroundingDINO components
try:
    from groundingdino.util.inference import load_model
    import groundingdino.datasets.transforms as T_gd
    GROUNDING_DINO_AVAILABLE = True
    logger.info("GroundingDINO imported successfully")
except ImportError as e:
    logger.error(f"GroundingDINO not available: {e}")
    GROUNDING_DINO_AVAILABLE = False

class FastLogoDetector:
    """
    FIXED: Single model instance with proper synchronization for batch processing.
    Eliminates meta tensor errors by using a single model with locks.
    """
    
    def __init__(self,
                 config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 weights_path: str = "logo/weights/groundingdino_swint_ogc.pth"):
        
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError("GroundingDINO is required but not available. Please install it.")

        # Resolve paths to be absolute and check for existence
        self.config_path = os.path.abspath(config_path)
        self.weights_path = os.path.abspath(weights_path)
        
        if not os.path.exists(self.config_path):
            logger.error(f"GroundingDINO config not found: {self.config_path}")
            raise FileNotFoundError(f"GroundingDINO config not found: {self.config_path}")
        if not os.path.exists(self.weights_path):
            logger.error(f"GroundingDINO weights not found: {self.weights_path}")
            raise FileNotFoundError(f"GroundingDINO weights not found: {self.weights_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # FIXED: Single model instance with proper locking for Streamlit
        self.model = None
        self.model_lock = asyncio.Lock()
        #self.model_thread_lock = threading.Lock()
        
        logger.info(f"FastLogoDetector initialized on device: {self.device}")

    async def _get_model(self):
        """Get the model instance with proper async locking."""
        async with self.model_lock:
            if self.model is None:
                try:
                    # Load model only once
                    self.model = load_model(self.config_path, self.weights_path)
                    self.model = self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Loaded GroundingDINO model on device {self.device}")
                except Exception as e:
                    logger.error(f"Failed to load GroundingDINO model: {e}")
                    raise RuntimeError("Failed to load GroundingDINO model.") from e
            return self.model

    def _validate_image_shape(self, image: np.ndarray) -> Tuple[int, int]:
        """Validate image shape and return height, width."""
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be a numpy array, got {type(image)}")
            
        if image.ndim != 3 or image.shape[2] not in [3, 4]:
            raise ValueError(f"Expected 3 or 4-channel image (H, W, C), got shape: {image.shape}")
        
        h, w, c = image.shape
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {h}x{w}")
        
        return h, w

    def _preprocess_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Preprocess image for GroundingDINO."""
        pil_image = Image.fromarray(image_rgb)
        
        transform = T_gd.Compose([
            T_gd.RandomResize([800], max_size=1333),
            T_gd.ToTensor(),
            T_gd.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_transformed, _ = transform(pil_image, None)
        return image_transformed

    def _create_mask_from_boxes(self, boxes: torch.Tensor, image_height: int, image_width: int) -> np.ndarray:
        """Create a binary mask from normalized bounding boxes."""
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        if boxes is None or boxes.numel() == 0:
            logger.debug("No logo boxes to create mask from.")
            return mask

        boxes_np = boxes.detach().cpu().numpy()
        
        if boxes_np.ndim == 1:
            if boxes_np.size == 4:
                boxes_np = boxes_np.reshape(1, 4)
            else:
                logger.warning(f"Unexpected 1D box shape: {boxes_np.shape}")
                return mask
        elif boxes_np.ndim != 2 or boxes_np.shape[1] != 4:
            logger.warning(f"Unexpected box shape: {boxes_np.shape}. Expected (N, 4).")
            return mask
        
        for box in boxes_np:
            cx, cy, w_norm, h_norm = box
            
            x1 = int((cx - w_norm / 2) * image_width)
            y1 = int((cy - h_norm / 2) * image_height)
            x2 = int((cx + w_norm / 2) * image_width)
            y2 = int((cy + h_norm / 2) * image_height)

            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = min(image_height, max(y1, y2))
            
            if x2 > x1 and y2 > y1:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    async def _detect_logos_safe(self,
                                 image: np.ndarray,
                                 text_prompt: str,
                                 box_threshold: float,
                                 text_threshold: float) -> np.ndarray:
        """
        FIXED: Safe logo detection with proper device handling and model synchronization.
        """
        h, w = self._validate_image_shape(image)
        
        try:
            model = await self._get_model()
            
            # Convert BGR to RGB if necessary
            if image.shape[2] == 4:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # FIXED: Ensure proper device placement and tensor handling
            with torch.no_grad():
                image_tensor = self._preprocess_image(image_rgb)
                
                # Ensure tensor is on the correct device
                image_tensor = image_tensor.unsqueeze(0).to(self.device, non_blocking=True)
                
                # Clear any cached computations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Run inference with proper error handling
                try:
                    outputs = model(image_tensor, captions=[text_prompt])
                except Exception as model_error:
                    logger.error(f"Model inference failed: {model_error}")
                    # Force cleanup and retry once
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Single retry
                    outputs = model(image_tensor, captions=[text_prompt])
                
                # Process outputs safely
                prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
                prediction_boxes = outputs["pred_boxes"].cpu()[0]
                
                # Clear GPU memory after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logits_filt = prediction_logits.max(dim=1)[0]
                boxes_filt = prediction_boxes[logits_filt > box_threshold]
                logits_filt = logits_filt[logits_filt > box_threshold]
                
                logger.info(f"GroundingDINO detected {len(boxes_filt)} logo(s) with scores: {logits_filt.tolist() if len(logits_filt) > 0 else []}")
            
            mask = self._create_mask_from_boxes(boxes_filt, h, w)
            
            return mask
            
        except Exception as e:
            logger.error(f"Logo detection failed: {e}")
            # Force GPU cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return np.zeros((h, w), dtype=np.uint8)

    def _enhance_mask(self, mask: np.ndarray,
                      dilation_kernel_size: int = 5,
                      blur_kernel_size: int = 7) -> np.ndarray:
        """Enhance the logo mask with morphological operations."""
        if not np.any(mask):
            return mask
        
        try:
            enhanced_mask = mask.copy()
            
            if dilation_kernel_size > 0:
                kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
                enhanced_mask = cv2.dilate(enhanced_mask, kernel, iterations=1)

            if blur_kernel_size > 0:
                if blur_kernel_size % 2 == 0:
                    blur_kernel_size += 1
                enhanced_mask = cv2.GaussianBlur(enhanced_mask, (blur_kernel_size, blur_kernel_size), 0)
                
            enhanced_mask = (enhanced_mask > 0).astype(np.uint8) * 255

        except Exception as e:
            logger.error(f"Mask enhancement failed: {e}")
            return mask

        return enhanced_mask

    async def detect_logos_async(self,
                                 image: np.ndarray,
                                 text_prompt: str = "logo . brand . company logo . trademark",
                                 box_threshold: float = 0.35,
                                 text_threshold: float = 0.25,
                                 enhance_mask: bool = True,
                                 dilation_size: int = 5,
                                 blur_size: int = 7) -> np.ndarray:
        """
        FIXED: Async logo detection with proper synchronization for batch processing.
        """
        h, w = self._validate_image_shape(image)
        
        try:
            # Use the async-safe detection method
            mask = await self._detect_logos_safe(
                image, text_prompt, box_threshold, text_threshold
            )
            
            if mask.shape != (h, w):
                logger.error(f"Detection returned an invalid mask shape: {mask.shape}. Expected: ({h}, {w})")
                return np.zeros((h, w), dtype=np.uint8)

            if enhance_mask and np.any(mask):
                # Run enhancement in executor to avoid blocking
                loop = asyncio.get_event_loop()
                mask = await loop.run_in_executor(
                    None,
                    self._enhance_mask,
                    mask, dilation_size, blur_size
                )
                if mask.shape != (h, w):
                    logger.error(f"Enhancement returned an invalid mask shape: {mask.shape}. Expected: ({h}, {w})")
                    return np.zeros((h, w), dtype=np.uint8)

            return mask
            
        except Exception as e:
            logger.error(f"Async logo detection pipeline failed: {e}")
            return np.zeros((h, w), dtype=np.uint8)


# FIXED: Global detector with proper initialization
_fast_detector = None
_detector_lock = asyncio.Lock()

async def get_fast_logo_detector(config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                weights_path: str = "logo/weights/groundingdino_swint_ogc.pth") -> FastLogoDetector:
    """Get or create the fast logo detector instance with async safety."""
    global _fast_detector
    async with _detector_lock:
        if _fast_detector is None:
            _fast_detector = FastLogoDetector(config_path, weights_path)
    return _fast_detector

async def get_logo_mask(image: np.ndarray, 
                        text_prompt: str = "logo . brand . company logo . trademark",
                        box_threshold: float = 0.35,
                        text_threshold: float = 0.25,
                        enhance_mask: bool = True) -> np.ndarray:
    """
    
    Args:
        image: Input image as a numpy array (BGR or BGRA format).
        text_prompt: Text prompt for logo detection.
        box_threshold: Box confidence threshold.
        text_threshold: Text confidence threshold.
        enhance_mask: Whether to enhance the mask.
        
    Returns:
        A binary mask (numpy array) with detected logo regions. Guaranteed to have the correct shape.
    """
    try:
        # Get the singleton detector instance with async safety
        detector = await get_fast_logo_detector()
        
        # Use the async detection method
        mask = await detector.detect_logos_async(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            enhance_mask=enhance_mask
        )
        
        h, w = image.shape[:2]
        if mask.shape != (h, w):
            logger.error(f"Logo mask final shape mismatch: expected ({h}, {w}), got {mask.shape}")
            return np.zeros((h, w), dtype=np.uint8)
        
        logger.info(f"Logo detection completed for image {w}x{h}. Total pixels in mask: {np.sum(mask > 0)}")
        return mask
        
    except Exception as e:
        logger.error(f"Logo detection pipeline failed with unhandled exception: {e}")
        h, w = image.shape[:2] if isinstance(image, np.ndarray) and image.ndim >= 2 else (0, 0)
        return np.zeros((h, w), dtype=np.uint8)