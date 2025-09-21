import cv2
import numpy as np
import torch
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional
from loguru import logger

from edge.hed_edge import HEDEdgeDetector
from edge.frequency_domain import FrequencyDomainPreservation
from edge.directional_gradient import DirectionalGradientFusion
from edge.colorspace_saliency import ColorSpaceSaliencyMaps
from edge.bayesian_confidence import BayesianConfidenceBlender

class AdvancedEdgeFusion:
    def __init__(self, device='cpu', max_workers=6):
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.detectors = {}
        self.enabled = {}
        self._init_detectors()
        logger.info(f"AdvancedEdgeFusion initialized with device={device} and max_workers={max_workers}")

    def _init_detectors(self):
        logger.info("Initializing edge detection modules...")
        self.detectors['hed'] = HEDEdgeDetector()
        self.enabled['hed'] = True
        logger.info(f"HED loaded | Enabled: {self.enabled['hed']}")

        self.detectors['frequency_domain'] = FrequencyDomainPreservation()
        self.enabled['frequency_domain'] = True
        logger.info("Frequency-Domain Preservation enabled")

        self.detectors['directional_gradient'] = DirectionalGradientFusion(num_orientations=8)
        self.enabled['directional_gradient'] = True # Time-taking
        logger.info("Directional Gradient Fusion enabled")

        self.detectors['colorspace_saliency'] = ColorSpaceSaliencyMaps()
        self.enabled['colorspace_saliency'] = True
        logger.info("Color-Space Saliency Maps enabled")

        self.blender = BayesianConfidenceBlender()
        logger.success("All detectors initialized successfully")

    def _detect(self, method, image, shape, **kwargs):
        if method not in self.enabled or not self.enabled[method]:
            logger.warning(f"Skipping disabled detector: {method}")
            return np.zeros(shape, dtype=np.uint8)

        detector = self.detectors[method]
        func = getattr(detector, 'detect_' + method, None)
        if not func:
            logger.error(f"No method detect_{method} found in {detector.__class__.__name__}")
            return np.zeros(shape, dtype=np.uint8)

        logger.debug(f"Running detector: {method}")
        start = time.perf_counter()
        mask = func(image, **kwargs.get(method, {}))
        elapsed = time.perf_counter() - start
        logger.debug(f"{method} completed in {elapsed:.3f}s")

        return self._format_mask(mask, shape)

    def _format_mask(self, mask, shape):
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 3 and mask.shape[2] == 3:
            mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        mask = np.squeeze(mask)
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]))
        return (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)

    async def detect_edges_async(self, image: np.ndarray, strength=1.0, methods: Optional[List[str]] = None) -> np.ndarray:
        shape = image.shape[:2]
        methods = methods or [k for k, v in self.enabled.items() if v]
        logger.info(f"Starting async edge detection with methods: {methods}")

        start_time = time.perf_counter()

        tasks = {
            m: asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda m=m: self._detect(m, image, shape, strength=strength)
            ) for m in methods
        }

        results = {m: await t for m, t in tasks.items()}
        masks = list(results.values())
        names = list(results.keys())

        logger.info("Blending masks with Bayesian Confidence Blender")
        final, weights = self.blender.blend_masks(masks, names, image=image)
        logger.success(f"Edge detection complete in {time.perf_counter() - start_time:.2f}s")

        for name, confidence_map in zip(names, weights):
            mean_confidence = np.mean(confidence_map)
            logger.debug(f"Mean Confidence for {name}: {mean_confidence:.3f}")

        return self._format_mask(final, shape)

    def detect_edges(self, image: np.ndarray, strength=1.0, methods: Optional[List[str]] = None) -> np.ndarray:
        logger.info("Running synchronous edge detection...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.detect_edges_async(image, strength, methods))
        loop.close()
        return result
