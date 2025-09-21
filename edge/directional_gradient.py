"""
Directional Gradient Fusion with Multi-Orientation Sobel Filters
Advanced edge detection using multiple directional gradients and fusion techniques.
"""

import cv2
import numpy as np
from loguru import logger
from typing import Tuple, List, Optional
import asyncio
from scipy import ndimage
from skimage.filters import gabor

class DirectionalGradientFusion:
    """Multi-directional gradient fusion for robust edge detection"""
    
    def __init__(self, num_orientations: int = 8):
        self.num_orientations = num_orientations
        # Orientations from 0 to just under pi (180 degrees)
        # For 8 orientations: 0, 22.5, 45, ..., 157.5 degrees
        self.orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)
        self._initialize_kernels()
        self.gabor_frequencies = [0.1, 0.2, 0.3] # Fixed frequencies for Gabor

    def _initialize_kernels(self):
        """Initialize directional filter kernels"""
        try:
            self.sobel_kernels = []
            
            # Create rotated Sobel kernels
            base_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            base_sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
            
            for angle in self.orientations:
                # Rotation matrix
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                
                # Rotate Sobel kernels
                # K_rotated = R * K_original
                # For a 2D filter, this is more involved than just a 2x2 matrix multiplication
                # We can approximate by rotating the gradient components
                rotated_x = cos_a * base_sobel_x + sin_a * base_sobel_y # Corrected rotation logic for kernels
                rotated_y = -sin_a * base_sobel_x + cos_a * base_sobel_y

                self.sobel_kernels.append((rotated_x, rotated_y))
            
            logger.info(f"Initialized {len(self.sobel_kernels)} directional Sobel kernels")
            
        except Exception as e:
            logger.error(f"Kernel initialization failed: {e}")
            self.sobel_kernels = []
    
    def _validate_and_normalize_mask(self, mask: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Validate, resize if necessary, and normalize mask to uint8"""
        try:
            if mask.shape != expected_shape:
                logger.warning(f"Mask shape mismatch: expected {expected_shape}, got {mask.shape}. Resizing.")
                mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Normalize to 0-255 uint8
            if mask.max() > 0:
                mask = mask / mask.max()
            mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
            
            return mask
        except Exception as e:
            logger.error(f"Mask validation/normalization failed: {e}")
            return np.zeros(expected_shape, dtype=np.uint8)
    
    def _compute_directional_sobel(self, gray_image: np.ndarray, strength: float = 1.0) -> List[np.ndarray]:
        """Compute multi-directional Sobel gradients"""
        try:
            directional_gradients = []
            
            for kernel_x, kernel_y in self.sobel_kernels:
                # Apply directional Sobel filters
                grad_x = cv2.filter2D(gray_image, cv2.CV_32F, kernel_x * strength)
                grad_y = cv2.filter2D(gray_image, cv2.CV_32F, kernel_y * strength)
                
                # Compute gradient magnitude
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                directional_gradients.append(magnitude)
            
            return directional_gradients
            
        except Exception as e:
            logger.error(f"Directional Sobel computation failed: {e}")
            h, w = gray_image.shape[:2]
            return [np.zeros((h, w), dtype=np.float32) for _ in range(self.num_orientations)]
    
    def _compute_gabor_responses(self, gray_image: np.ndarray) -> List[np.ndarray]:
        """Compute Gabor filter responses at multiple orientations and frequencies"""
        try:
            gabor_responses = []
            
            for freq in self.gabor_frequencies:
                for angle in self.orientations:
                    # Apply Gabor filter
                    # The gabor function returns real and imaginary parts
                    real, _ = gabor(gray_image, frequency=freq, theta=angle)
                    
                    # Use absolute value of the real part as response
                    response = np.abs(real)
                    gabor_responses.append(response)
            
            return gabor_responses
            
        except Exception as e:
            logger.error(f"Gabor filter computation failed: {e}")
            h, w = gray_image.shape[:2]
            # Return a list of zeros if computation fails, matching expected output structure
            return [np.zeros((h, w), dtype=np.float32) for _ in range(len(self.gabor_frequencies) * self.num_orientations)]
    
    def _compute_structure_tensor(self, gray_image: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute structure tensor for orientation analysis"""
        try:
            # Compute gradients
            grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
            
            # Structure tensor components
            # Use ndimage.gaussian_filter for potentially better performance with specific sigmas
            J11 = ndimage.gaussian_filter(grad_x * grad_x, sigma)
            J22 = ndimage.gaussian_filter(grad_y * grad_y, sigma)
            J12 = ndimage.gaussian_filter(grad_x * grad_y, sigma)
            
            return J11, J22, J12
            
        except Exception as e:
            logger.error(f"Structure tensor computation failed: {e}")
            h, w = gray_image.shape[:2]
            zeros = np.zeros((h, w), dtype=np.float32)
            return zeros, zeros, zeros
    
    def _compute_coherence_and_orientation(self, J11: np.ndarray, J22: np.ndarray, J12: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute coherence and dominant orientation from structure tensor"""
        try:
            # Eigenvalue analysis
            trace = J11 + J22
            det = J11 * J22 - J12 * J12
            
            # Eigenvalues
            # Add a small epsilon to sqrt argument to prevent NaN for very small negative values
            sqrt_val = np.sqrt(np.maximum(0, trace**2 - 4*det))
            lambda1 = 0.5 * (trace + sqrt_val)
            lambda2 = 0.5 * (trace - sqrt_val)
            
            # Coherence measure: (lambda1 - lambda2) / (lambda1 + lambda2)
            # Add a small epsilon to denominator to prevent division by zero
            coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
            coherence = np.nan_to_num(coherence, nan=0.0) # Handle potential NaN from 0/0
            
            # Dominant orientation: 0.5 * arctan2(2*J12, J11 - J22)
            # arctan2 handles quadrants correctly
            orientation = 0.5 * np.arctan2(2*J12, J11 - J22)
            
            return coherence, orientation
            
        except Exception as e:
            logger.error(f"Coherence computation failed: {e}")
            h, w = J11.shape
            return np.zeros((h, w), dtype=np.float32), np.zeros((h, w), dtype=np.float32)
    
    def _fuse_directional_gradients(
        self, 
        gradients: List[np.ndarray], 
        fusion_method: str = 'max'
    ) -> np.ndarray:
        """Fuse multiple directional gradients"""
        try:
            if not gradients:
                logger.warning("No gradients to fuse")
                # Return an empty array of a reasonable size or raise an error
                return np.zeros((100, 100), dtype=np.float32) # Default size if no gradients

            gradients_stack = np.stack(gradients, axis=0) # Shape: (N, H, W)
            
            if fusion_method == 'max':
                # Maximum response across orientations
                fused = np.max(gradients_stack, axis=0)
            elif fusion_method == 'mean':
                # Average response
                fused = np.mean(gradients_stack, axis=0)
            elif fusion_method == 'weighted_sum':
                # Weighted sum based on total gradient strength (simple heuristic)
                # Compute weights based on the sum of values in each gradient map
                weights = np.array([np.sum(g) for g in gradients], dtype=np.float32)
                weights = weights / (np.sum(weights) + 1e-10) # Normalize weights
                
                # Reshape weights for broadcasting (N, 1, 1) to multiply (N, H, W)
                fused = np.sum(gradients_stack * weights[:, np.newaxis, np.newaxis], axis=0)
            elif fusion_method == 'selective':
                logger.warning(f"'selective' fusion method is complex and might not yield expected results without more sophisticated coherence estimation. Falling back to 'max'.")
                fused = np.max(gradients_stack, axis=0)
            else:
                logger.warning(f"Unknown fusion method: {fusion_method}, using max")
                fused = np.max(gradients_stack, axis=0)
            
            # Normalize fused gradient to 0-1 range
            if fused.max() > 0:
                fused = fused / fused.max()
            
            return fused
            
        except Exception as e:
            logger.error(f"Gradient fusion failed: {e}")
            return gradients[0] if gradients else np.zeros((100, 100))
    
    def _apply_non_maximum_suppression(self, magnitude: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression (vectorized)"""
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude, dtype=np.float32)

        # Convert orientation to degrees and normalize to [0, 180)
        angle = np.rad2deg(orientation) % 180 # Ensures angle is in [0, 180)

        padded_magnitude = np.pad(magnitude, 1, mode='constant', constant_values=0) # Or mode='edge'

        
        # Create arrays for row and column indices for the original magnitude map
        rows, cols = np.indices((h, w))
        
        # Initialize neighbor magnitude arrays
        # These will store the magnitude of the two pixels in the gradient direction
        M1 = np.zeros_like(magnitude, dtype=np.float32)
        M2 = np.zeros_like(magnitude, dtype=np.float32)

        # Define masks for each direction based on angle
        # Angles are quantized to represent horizontal, +45 deg, vertical, -45 deg
        angle_0_mask = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180)) # Horizontal
        angle_45_mask = (22.5 <= angle) & (angle < 67.5) # +45 degree diagonal
        angle_90_mask = (67.5 <= angle) & (angle < 112.5) # Vertical
        angle_135_mask = (112.5 <= angle) & (angle < 157.5) # -45 degree diagonal

        # Apply neighbor fetching based on masks
        # Remember to adjust indices for the padded_magnitude array (+1 for row/col)
        
        # Horizontal (0 / 180 degrees)
        M1[angle_0_mask] = padded_magnitude[rows[angle_0_mask] + 1, cols[angle_0_mask] + 1 - 1]
        M2[angle_0_mask] = padded_magnitude[rows[angle_0_mask] + 1, cols[angle_0_mask] + 1 + 1]

        # +45 degree diagonal (top-right to bottom-left)
        M1[angle_45_mask] = padded_magnitude[rows[angle_45_mask] + 1 - 1, cols[angle_45_mask] + 1 + 1]
        M2[angle_45_mask] = padded_magnitude[rows[angle_45_mask] + 1 + 1, cols[angle_45_mask] + 1 - 1]

        # Vertical (90 degrees)
        M1[angle_90_mask] = padded_magnitude[rows[angle_90_mask] + 1 - 1, cols[angle_90_mask] + 1]
        M2[angle_90_mask] = padded_magnitude[rows[angle_90_mask] + 1 + 1, cols[angle_90_mask] + 1]

        # -45 degree diagonal (top-left to bottom-right)
        M1[angle_135_mask] = padded_magnitude[rows[angle_135_mask] + 1 - 1, cols[angle_135_mask] + 1 - 1]
        M2[angle_135_mask] = padded_magnitude[rows[angle_135_mask] + 1 + 1, cols[angle_135_mask] + 1 + 1]
        
        # Apply suppression: if current pixel is greater than or equal to both neighbors
        suppressed = np.where((magnitude >= M1) & (magnitude >= M2), magnitude, 0)
        
        return suppressed
    
    async def detect_directional_gradient_async(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        fusion_method: str = 'max',
        use_gabor: bool = True,
        use_nms: bool = True,
        edge_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Asynchronous directional gradient edge detection
        
        Args:
            image: Input image in BGR format
            strength: Gradient strength multiplier
            fusion_method: Method for fusing directional gradients ('max', 'mean', 'weighted_sum', 'selective')
            use_gabor: Whether to include Gabor filter responses
            use_nms: Whether to apply non-maximum suppression
            edge_threshold: Threshold for final edge detection (0.0 to 1.0)
            
        Returns:
            Directional edge mask as uint8 numpy array
        """
        try:
            loop = asyncio.get_running_loop() # Use get_running_loop for modern async code
            result = await loop.run_in_executor(
                None, # Use default ThreadPoolExecutor
                self._detect_directional_gradient_sync,
                image,
                strength,
                fusion_method,
                use_gabor,
                use_nms,
                edge_threshold
            )
            return result
        except Exception as e:
            logger.error(f"Async directional edge detection failed: {e}")
            return self._fallback_edge_detection(image, strength)
    
    def _detect_directional_gradient_sync(
        self,
        image: np.ndarray,
        strength: float,
        fusion_method: str,
        use_gabor: bool,
        use_nms: bool,
        edge_threshold: float
    ) -> np.ndarray:
        """Synchronous directional gradient edge detection"""
        try:
            h, w = image.shape[:2]
            expected_shape = (h, w)
            
            # Convert to grayscale once and ensure float32 for processing
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            gray_image = gray_image.astype(np.float32)
            
            # Compute directional Sobel gradients
            sobel_gradients = self._compute_directional_sobel(gray_image, strength)
            
            all_gradients = sobel_gradients # Start with Sobel gradients
            
            # Add Gabor responses if requested
            if use_gabor:
                gabor_responses = self._compute_gabor_responses(gray_image)
                all_gradients.extend(gabor_responses)
            
            # Fuse all gradient responses
            fused_gradient = self._fuse_directional_gradients(all_gradients, fusion_method)
            
            # Apply non-maximum suppression if requested
            if use_nms:
                # Compute structure tensor for orientation (needed for NMS)
                J11, J22, J12 = self._compute_structure_tensor(gray_image)
                _, orientation = self._compute_coherence_and_orientation(J11, J22, J12)
                
                fused_gradient = self._apply_non_maximum_suppression(fused_gradient, orientation)
            
            # Apply threshold
            # Normalize fused_gradient to 0-1 range if not already done by fusion
            if fused_gradient.max() > 0:
                fused_gradient = fused_gradient / fused_gradient.max()
                
            edge_mask = (fused_gradient > edge_threshold).astype(np.float32)
            
            # Post-processing: morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel)
            edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, kernel) # Optional: remove small noise

            # Convert to uint8 and validate shape/normalize
            final_edge_mask = self._validate_and_normalize_mask(edge_mask, expected_shape)
            
            return final_edge_mask
            
        except Exception as e:
            logger.error(f"Directional edge detection failed: {e}")
            return self._fallback_edge_detection(image, strength)
    
    def _fallback_edge_detection(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Fallback edge detection using standard methods (Canny)"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Use Canny as a robust fallback
            low_threshold = int(50 * strength)
            high_threshold = int(150 * strength)
            
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            # Canny output is already uint8 (0 or 255)
            return edges
            
        except Exception as e:
            logger.error(f"Fallback edge detection failed: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def detect_directional_gradient(
        self,
        image: np.ndarray,
        strength: float = 1.0,
        fusion_method: str = 'max',
        use_gabor: bool = True,
        use_nms: bool = True,
        edge_threshold: float = 0.1
    ) -> np.ndarray:
        """
        Synchronous wrapper for directional gradient edge detection
        
        Args:
            image: Input image in BGR format
            strength: Gradient strength multiplier (0.1 to 3.0)
            fusion_method: 'max', 'mean', 'weighted_sum', or 'selective'
            use_gabor: Whether to include Gabor filter responses
            use_nms: Whether to apply non-maximum suppression 
            edge_threshold: Threshold for final edge detection (0.0 to 1.0)
            
        Returns:
            Directional edge mask as uint8 numpy array
        """
        return self._detect_directional_gradient_sync(
            image, strength, fusion_method, use_gabor, use_nms, edge_threshold
        )
    
    def get_gradient_statistics(self, edge_mask: np.ndarray, gradients: Optional[List[np.ndarray]] = None) -> dict:
        """Get statistics about directional gradients"""
        try:
            total_pixels = edge_mask.size
            edge_pixels = np.sum(edge_mask > 0)
            edge_density = float(edge_pixels) / total_pixels
            
            stats = {
                'total_pixels': total_pixels,
                'edge_pixels': int(edge_pixels),
                'edge_density': float(edge_density),
                'num_orientations': self.num_orientations
            }
            
            if gradients:
                # Compute orientation-wise statistics from initial Sobel gradients
                # Assuming the first `self.num_orientations` gradients are Sobel ones
                sobel_only_gradients = gradients[:self.num_orientations]
                orientation_strengths = [np.sum(grad) for grad in sobel_only_gradients]
                
                if np.sum(orientation_strengths) > 0: # Avoid division by zero
                    normalized_strengths = np.array(orientation_strengths) / np.sum(orientation_strengths)
                    dominant_orientation_idx = np.argmax(normalized_strengths)
                    dominant_angle_rad = self.orientations[dominant_orientation_idx]
                    dominant_angle_deg = dominant_angle_rad * 180 / np.pi
                else:
                    dominant_orientation_idx = -1
                    dominant_angle_deg = 0.0

                stats.update({
                    'orientation_strengths': [float(s) for s in orientation_strengths],
                    'dominant_orientation_idx': int(dominant_orientation_idx),
                    'dominant_angle_degrees': float(dominant_angle_deg)
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Gradient statistics computation failed: {e}")
            return {
                'total_pixels': 0,
                'edge_pixels': 0,
                'edge_density': 0.0,
                'num_orientations': self.num_orientations,
                'error': str(e)
            }