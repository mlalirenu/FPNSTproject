"""
Color-Space Saliency Maps using Lab/HSV Variance
Advanced saliency detection across multiple color spaces for edge preservation.
"""

import cv2
import numpy as np
from loguru import logger
from typing import Tuple, Dict, List, Optional
import asyncio
from scipy import ndimage
from skimage import color


class ColorSpaceSaliencyMaps:
    """Multi-colorspace saliency detection for robust edge preservation"""
    
    def __init__(self):
        self.color_spaces = ['lab', 'hsv', 'rgb', 'yuv', 'xyz']
        self.initialized = True
        logger.info("ColorSpace saliency detector initialized")
    
    def _validate_shape(self, mask: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Validate and fix mask shape"""
        try:
            if mask.shape != expected_shape:
                logger.warning(f"ColorSpace mask shape mismatch: expected {expected_shape}, got {mask.shape}")
                mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]))
            
            if mask.dtype != np.uint8:
                mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
            
            return mask
        except Exception as e:
            logger.error(f"ColorSpace shape validation failed: {e}")
            return np.zeros(expected_shape, dtype=np.uint8)
    
    def _convert_colorspace(self, image: np.ndarray, target_space: str) -> np.ndarray:
        """Convert image to target color space"""
        try:
            # Ensure image is in RGB format first
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            if target_space.lower() == 'lab':
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
            elif target_space.lower() == 'hsv':
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            elif target_space.lower() == 'rgb':
                return rgb_image
            elif target_space.lower() == 'yuv':
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
            elif target_space.lower() == 'xyz':
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2XYZ)
            else:
                logger.warning(f"Unknown color space: {target_space}, using RGB")
                return rgb_image
                
        except Exception as e:
            logger.error(f"Color space conversion failed for {target_space}: {e}")
            return image
    
    def _compute_channel_variance_saliency(self, channel: np.ndarray, window_size: int = 15) -> np.ndarray:
        """Compute saliency based on local variance in a channel"""
        try:
            # Normalize channel to [0, 1]
            channel_norm = channel.astype(np.float32)
            if channel_norm.max() > 1.0:
                channel_norm = channel_norm / 255.0
            
            # Compute local mean using uniform filter
            local_mean = ndimage.uniform_filter(channel_norm, size=window_size)
            
            # Compute local variance
            local_var = ndimage.uniform_filter(channel_norm**2, size=window_size) - local_mean**2
            
            # Apply non-linear transformation to enhance saliency
            saliency = np.exp(local_var * 10) - 1
            
            # Normalize to [0, 1]
            if saliency.max() > 0:
                saliency = saliency / saliency.max()
            
            return saliency
            
        except Exception as e:
            logger.error(f"Channel variance saliency computation failed: {e}")
            return np.zeros_like(channel, dtype=np.float32)
    
    def _compute_center_surround_saliency(self, channel: np.ndarray, scales: List[int] = None) -> np.ndarray:
        """Compute center-surround saliency using difference of Gaussians"""
        try:
            if scales is None:
                scales = [3, 7, 15, 31]
            
            channel_float = channel.astype(np.float32)
            if channel_float.max() > 1.0:
                channel_float = channel_float / 255.0
            
            saliency_maps = []
            
            for i in range(len(scales) - 1):
                center_scale = scales[i]
                surround_scale = scales[i + 1]
                
                # Apply Gaussian blur
                center = cv2.GaussianBlur(channel_float, (center_scale, center_scale), 0)
                surround = cv2.GaussianBlur(channel_float, (surround_scale, surround_scale), 0)
                
                # Compute difference
                diff = np.abs(center - surround)
                saliency_maps.append(diff)
            
            # Combine multi-scale saliency
            combined_saliency = np.mean(saliency_maps, axis=0)
            
            # Normalize
            if combined_saliency.max() > 0:
                combined_saliency = combined_saliency / combined_saliency.max()
            
            return combined_saliency
            
        except Exception as e:
            logger.error(f"Center-surround saliency computation failed: {e}")
            return np.zeros_like(channel, dtype=np.float32)
    
    def _compute_color_contrast_saliency(self, image_colorspace: np.ndarray) -> np.ndarray:
        """Compute saliency based on color contrast"""
        try:
            h, w, c = image_colorspace.shape
            
            # Convert to float
            image_float = image_colorspace.astype(np.float32)
            if image_float.max() > 1.0:
                image_float = image_float / 255.0
            
            # Compute global color statistics
            global_mean = np.mean(image_float.reshape(-1, c), axis=0)
            
            # Compute pixel-wise distance from global mean
            pixel_distances = np.zeros((h, w), dtype=np.float32)
            for i in range(c):
                pixel_distances += (image_float[:, :, i] - global_mean[i])**2
            
            pixel_distances = np.sqrt(pixel_distances)
            
            # Apply spatial weighting (center bias)
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            spatial_weight = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * (min(h, w) / 4)**2))
            
            # Combine distance and spatial weighting
            contrast_saliency = pixel_distances * (1 + 0.5 * spatial_weight)
            
            # Normalize
            if contrast_saliency.max() > 0:
                contrast_saliency = contrast_saliency / contrast_saliency.max()
            
            return contrast_saliency
            
        except Exception as e:
            logger.error(f"Color contrast saliency computation failed: {e}")
            return np.zeros(image_colorspace.shape[:2], dtype=np.float32)
    
    def _compute_lab_saliency(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute saliency in LAB color space"""
        try:
            lab_image = self._convert_colorspace(image, 'lab')
            L, a, b = cv2.split(lab_image)
            
            # Luminance saliency (L channel)
            L_variance = self._compute_channel_variance_saliency(L)
            L_center_surround = self._compute_center_surround_saliency(L)
            
            # Color opponent saliency (a and b channels)
            a_saliency = self._compute_channel_variance_saliency(a)
            b_saliency = self._compute_channel_variance_saliency(b)
            
            # Color contrast saliency
            color_contrast = self._compute_color_contrast_saliency(lab_image)
            
            return {
                'luminance_variance': L_variance,
                'luminance_center_surround': L_center_surround,
                'color_a': a_saliency,
                'color_b': b_saliency,
                'color_contrast': color_contrast
            }
            
        except Exception as e:
            logger.error(f"LAB saliency computation failed: {e}")
            h, w = image.shape[:2]
            zeros = np.zeros((h, w), dtype=np.float32)
            return {
                'luminance_variance': zeros,
                'luminance_center_surround': zeros,
                'color_a': zeros,
                'color_b': zeros,
                'color_contrast': zeros
            }
    
    def _compute_hsv_saliency(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute saliency in HSV color space"""
        try:
            hsv_image = self._convert_colorspace(image, 'hsv')
            H, S, V = cv2.split(hsv_image)
            
            # Hue saliency (circular variance)
            H_saliency = self._compute_hue_saliency(H)
            
            # Saturation saliency
            S_variance = self._compute_channel_variance_saliency(S)
            S_center_surround = self._compute_center_surround_saliency(S)
            
            # Value (brightness) saliency
            V_variance = self._compute_channel_variance_saliency(V)
            V_center_surround = self._compute_center_surround_saliency(V)
            
            # Color contrast in HSV
            color_contrast_hsv = self._compute_color_contrast_saliency(hsv_image)
            
            return {
                'hue_circular': H_saliency,
                'saturation_variance': S_variance,
                'saturation_center_surround': S_center_surround,
                'value_variance': V_variance,
                'value_center_surround': V_center_surround,
                'color_contrast_hsv': color_contrast_hsv
            }
            
        except Exception as e:
            logger.error(f"HSV saliency computation failed: {e}")
            h, w = image.shape[:2]
            zeros = np.zeros((h, w), dtype=np.float32)
            return {key: zeros for key in ['hue_circular', 'saturation_variance', 
                                         'saturation_center_surround', 'value_variance',
                                         'value_center_surround', 'color_contrast_hsv']}
    
    def _compute_hue_saliency(self, hue_channel: np.ndarray, window_size: int = 15) -> np.ndarray:
        """Compute saliency for hue channel considering circular nature"""
        try:
            # Convert hue to radians for circular statistics
            hue_rad = hue_channel.astype(np.float32) * 2 * np.pi / 180.0
            
            # Compute circular mean and variance using complex representation
            complex_hue = np.exp(1j * hue_rad)
            
            # Local circular mean
            real_part = ndimage.uniform_filter(np.real(complex_hue), size=window_size)
            imag_part = ndimage.uniform_filter(np.imag(complex_hue), size=window_size)
            local_circular_mean = real_part + 1j * imag_part
            
            # Circular variance (1 - |mean|)
            circular_variance = 1 - np.abs(local_circular_mean)
            
            # Normalize
            if circular_variance.max() > 0:
                circular_variance = circular_variance / circular_variance.max()
            
            return circular_variance
            
        except Exception as e:
            logger.error(f"Hue saliency computation failed: {e}")
            return np.zeros_like(hue_channel, dtype=np.float32)
    
    def _compute_rgb_saliency(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute saliency in RGB color space"""
        try:
            rgb_image = self._convert_colorspace(image, 'rgb')
            R, G, B = cv2.split(rgb_image)
            
            # Individual channel saliency
            R_saliency = self._compute_channel_variance_saliency(R)
            G_saliency = self._compute_channel_variance_saliency(G)
            B_saliency = self._compute_channel_variance_saliency(B)
            
            # Color opponency saliency
            RG_opponency = self._compute_channel_variance_saliency(R.astype(np.float32) - G.astype(np.float32))
            BY_opponency = self._compute_channel_variance_saliency(B.astype(np.float32) - (R.astype(np.float32) + G.astype(np.float32)) / 2)
            
            # Color contrast
            color_contrast_rgb = self._compute_color_contrast_saliency(rgb_image)
            
            return {
                'red_variance': R_saliency,
                'green_variance': G_saliency,
                'blue_variance': B_saliency,
                'rg_opponency': RG_opponency,
                'by_opponency': BY_opponency,
                'color_contrast_rgb': color_contrast_rgb
            }
            
        except Exception as e:
            logger.error(f"RGB saliency computation failed: {e}")
            h, w = image.shape[:2]
            zeros = np.zeros((h, w), dtype=np.float32)
            return {key: zeros for key in ['red_variance', 'green_variance', 'blue_variance',
                                         'rg_opponency', 'by_opponency', 'color_contrast_rgb']}
    
    def _compute_yuv_saliency(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute saliency in YUV color space"""
        try:
            yuv_image = self._convert_colorspace(image, 'yuv')
            Y, U, V = cv2.split(yuv_image)
            
            # Luminance saliency
            Y_variance = self._compute_channel_variance_saliency(Y)
            Y_center_surround = self._compute_center_surround_saliency(Y)
            
            # Chrominance saliency
            U_saliency = self._compute_channel_variance_saliency(U)
            V_saliency = self._compute_channel_variance_saliency(V)
            
            # Color contrast
            color_contrast_yuv = self._compute_color_contrast_saliency(yuv_image)
            
            return {
                'luma_variance': Y_variance,
                'luma_center_surround': Y_center_surround,
                'chroma_u': U_saliency,
                'chroma_v': V_saliency,
                'color_contrast_yuv': color_contrast_yuv
            }
            
        except Exception as e:
            logger.error(f"YUV saliency computation failed: {e}")
            h, w = image.shape[:2]
            zeros = np.zeros((h, w), dtype=np.float32)
            return {key: zeros for key in ['luma_variance', 'luma_center_surround',
                                         'chroma_u', 'chroma_v', 'color_contrast_yuv']}
    
    def _fuse_colorspace_saliency(
        self, 
        saliency_maps: Dict[str, Dict[str, np.ndarray]], 
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """Fuse saliency maps from different color spaces"""
        try:
            if weights is None:
                weights = {
                    'lab': 0.4,
                    'hsv': 0.3,
                    'rgb': 0.2,
                    'yuv': 0.1
                }
            
            # Collect all saliency maps
            all_maps = []
            map_weights = []
            
            for space, space_maps in saliency_maps.items():
                space_weight = weights.get(space, 0.1)
                for map_name, saliency_map in space_maps.items():
                    all_maps.append(saliency_map)
                    # Weight based on color space and map importance
                    if 'contrast' in map_name:
                        map_weights.append(space_weight * 1.2)
                    elif 'variance' in map_name:
                        map_weights.append(space_weight * 1.0)
                    elif 'center_surround' in map_name:
                        map_weights.append(space_weight * 0.8)
                    else:
                        map_weights.append(space_weight * 0.6)
            
            if not all_maps:
                logger.warning("No saliency maps to fuse")
                return np.zeros((100, 100), dtype=np.float32)
            
            # Normalize weights
            map_weights = np.array(map_weights)
            map_weights = map_weights / np.sum(map_weights)
            
            # Weighted fusion
            fused_saliency = np.zeros_like(all_maps[0], dtype=np.float32)
            for saliency_map, weight in zip(all_maps, map_weights):
                fused_saliency += weight * saliency_map
            
            # Normalize final result
            if fused_saliency.max() > 0:
                fused_saliency = fused_saliency / fused_saliency.max()
            
            return fused_saliency
            
        except Exception as e:
            logger.error(f"Saliency fusion failed: {e}")
            if all_maps:
                return all_maps[0]
            return np.zeros((100, 100), dtype=np.float32)
    
    async def detect_colorspace_saliency_async(
        self,
        image: np.ndarray,
        colorspaces: List[str] = None,
        fusion_weights: Dict[str, float] = None,
        edge_threshold: float = 0.3,
        use_morphology: bool = True
    ) -> np.ndarray:
        """
        Asynchronous multi-colorspace saliency detection
        
        Args:
            image: Input image in BGR format
            colorspaces: List of color spaces to use
            fusion_weights: Weights for different color spaces
            edge_threshold: Threshold for final edge detection
            use_morphology: Whether to apply morphological operations
            
        Returns:
            Colorspace saliency edge mask as uint8 numpy array
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._detect_colorspace_saliency_sync,
                image,
                colorspaces,
                fusion_weights,
                edge_threshold,
                use_morphology
            )
            return result
        except Exception as e:
            logger.error(f"Async colorspace saliency detection failed: {e}")
            return self._fallback_saliency_detection(image)
    
    def _detect_colorspace_saliency_sync(
        self,
        image: np.ndarray,
        colorspaces: Optional[List[str]],
        fusion_weights: Optional[Dict[str, float]],
        edge_threshold: float,
        use_morphology: bool
    ) -> np.ndarray:
        """Synchronous multi-colorspace saliency detection"""
        try:
            h, w = image.shape[:2]
            expected_shape = (h, w)
            
            if colorspaces is None:
                colorspaces = ['lab', 'hsv', 'rgb']
            
            # Compute saliency in each color space
            saliency_maps = {}
            
            for space in colorspaces:
                if space.lower() == 'lab':
                    saliency_maps['lab'] = self._compute_lab_saliency(image)
                elif space.lower() == 'hsv':
                    saliency_maps['hsv'] = self._compute_hsv_saliency(image)
                elif space.lower() == 'rgb':
                    saliency_maps['rgb'] = self._compute_rgb_saliency(image)
                elif space.lower() == 'yuv':
                    saliency_maps['yuv'] = self._compute_yuv_saliency(image)
                else:
                    logger.warning(f"Unknown color space: {space}")
            
            # Fuse saliency maps
            fused_saliency = self._fuse_colorspace_saliency(saliency_maps, fusion_weights)
            
            # Apply threshold
            edge_mask = (fused_saliency > edge_threshold).astype(np.float32)
            
            # Morphological operations if requested
            if use_morphology:
                # Close small gaps
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel_close)
                
                # Remove small noise
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel_open)
            
            # Convert to uint8 and validate shape
            edge_mask = (edge_mask * 255).astype(np.uint8)
            edge_mask = self._validate_shape(edge_mask, expected_shape)
            
            return edge_mask
            
        except Exception as e:
            logger.error(f"Colorspace saliency detection failed: {e}")
            return self._fallback_saliency_detection(image)
    
    def _fallback_saliency_detection(self, image: np.ndarray) -> np.ndarray:
        """Fallback saliency detection using simple methods"""
        try:
            # Convert to LAB and use simple contrast
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            L = lab[:, :, 0]
            
            # Simple center-surround difference
            blur_small = cv2.GaussianBlur(L, (5, 5), 1)
            blur_large = cv2.GaussianBlur(L, (25, 25), 5)
            
            saliency = np.abs(blur_small.astype(np.float32) - blur_large.astype(np.float32))
            
            # Normalize and threshold
            saliency = (saliency / saliency.max() * 255).astype(np.uint8)
            
            return saliency
            
        except Exception as e:
            logger.error(f"Fallback saliency detection failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def detect_colorspace_saliency(
        self,
        image: np.ndarray,
        colorspaces: List[str] = None,
        fusion_weights: Dict[str, float] = None,
        edge_threshold: float = 0.3,
        use_morphology: bool = True
    ) -> np.ndarray:
        """
        Synchronous wrapper for multi-colorspace saliency detection
        
        Args:
            image: Input image in BGR format
            colorspaces: List of color spaces to use (['lab', 'hsv', 'rgb', 'yuv'])
            fusion_weights: Weights for different color spaces
            edge_threshold: Threshold for final edge detection (0.0 to 1.0)
            use_morphology: Whether to apply morphological operations
            
        Returns:
            Colorspace saliency edge mask as uint8 numpy array
        """
        return self._detect_colorspace_saliency_sync(
            image, colorspaces, fusion_weights, edge_threshold, use_morphology
        )
    
    def get_colorspace_statistics(
        self, 
        saliency_mask: np.ndarray, 
        saliency_maps: Dict[str, Dict[str, np.ndarray]] = None
    ) -> dict:
        """Get statistics about colorspace saliency detection"""
        try:
            total_pixels = saliency_mask.size
            salient_pixels = np.sum(saliency_mask > 0)
            saliency_density = salient_pixels / total_pixels
            
            stats = {
                'total_pixels': total_pixels,
                'salient_pixels': int(salient_pixels),
                'saliency_density': float(saliency_density),
                'colorspaces_used': list(self.color_spaces)
            }
            
            if saliency_maps:
                # Compute per-colorspace statistics
                colorspace_stats = {}
                for space, space_maps in saliency_maps.items():
                    space_total_saliency = sum(np.sum(smap) for smap in space_maps.values())
                    colorspace_stats[space] = {
                        'num_maps': len(space_maps),
                        'total_saliency': float(space_total_saliency),
                        'map_names': list(space_maps.keys())
                    }
                
                stats['colorspace_breakdown'] = colorspace_stats
            
            return stats
            
        except Exception as e:
            logger.error(f"Colorspace statistics computation failed: {e}")
            return {
                'total_pixels': 0,
                'salient_pixels': 0,
                'saliency_density': 0.0,
                'colorspaces_used': self.color_spaces
            }