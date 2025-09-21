"""
HED-based Edge Detection for Advanced Preservation Masks
Implements HED (Holistically-Nested Edge Detection) using pre-trained weights.
Fixed version addressing halo effects and spatial misalignment issues.
"""

import cv2
import numpy as np
import os
from loguru import logger
from scipy import ndimage
from skimage import morphology, feature
from typing import Tuple


class HEDEdgeDetector:
    """Holistically-Nested Edge Detection (HED) using OpenCV DNN - Fixed Version"""

    def __init__(self):
        # Using absolute paths or robust path joining is crucial for deployment
        # Changed path separator to os.path.join for cross-platform compatibility
        proto_path = os.path.join('edge', 'weights', 'deploy.prototxt')
        model_path = os.path.join('edge', 'weights', 'hed_pretrained_bsds.caffemodel')

        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            logger.error(f"HED model files not found: {proto_path} and {model_path}")
            raise FileNotFoundError("Missing HED model files. Ensure 'weights' directory is in 'edge'.")

        try:
            self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            if self.net.empty():
                raise RuntimeError("Failed to load HED network (net object is empty). Check model files integrity.")
        except Exception as e:
            logger.error(f"Error loading HED network: {e}")
            raise RuntimeError(f"Failed to initialize HED network: {e}")

        # Setting backend and target for performance - CPU is default
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("HEDEdgeDetector initialized successfully.")

    def _remove_halo_effects(self, edge_map: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Remove halo effects and duplicate masks using advanced morphological operations
        and spatial filtering.
        """
        # 1. Remove small isolated noise pixels and weak responses
        binary_mask = edge_map > 0.15
        cleaned = morphology.remove_small_objects(binary_mask, min_size=15, connectivity=2)
        edge_map_cleaned = cleaned.astype(np.float32) * edge_map
        
        # 2. Detect and remove duplicate/offset patterns
        # Use morphological top-hat to detect thin artifacts
        kernel_large = morphology.disk(kernel_size * 2)
        tophat = morphology.white_tophat(edge_map_cleaned, kernel_large)
        
        # Remove top-hat artifacts (these are often duplicates/halos)
        dehalo_step1 = np.maximum(0, edge_map_cleaned - 0.8 * tophat)
        
        # 3. Apply opening with different kernel sizes to remove halos of different scales
        kernel_small = morphology.disk(1)
        kernel_medium = morphology.disk(kernel_size)
        
        opened_small = morphology.opening(dehalo_step1, kernel_small)
        opened_medium = morphology.opening(dehalo_step1, kernel_medium)
        
        # Keep the stronger result from multi-scale opening
        multi_scale_opened = np.maximum(opened_small, 0.7 * opened_medium)
        
        # 4. Distance transform to identify true edge centers and remove parallel duplicates
        if np.any(multi_scale_opened > 0.2):
            # Create binary mask for distance transform
            binary_edges = multi_scale_opened > 0.2
            
            # Distance transform to find ridge lines (true edge centers)
            distance = ndimage.distance_transform_edt(~binary_edges)
            
            # Fixed: Use feature.peak_local_maxima instead of morphology.local_maxima
            # Find local maxima in the distance transform (these are centers between duplicates)
            try:
                # Try the newer API first
                local_maxima_coords = feature.peak_local_maxima(distance, min_distance=3, threshold_abs=0.1)
                local_maxima = np.zeros_like(distance, dtype=bool)
                if len(local_maxima_coords) > 0 and len(local_maxima_coords[0]) > 0:
                    local_maxima[local_maxima_coords] = True
            except (AttributeError, TypeError):
                # Fallback: Use a simpler approach with maximum filter
                from scipy.ndimage import maximum_filter
                local_maxima = (distance == maximum_filter(distance, size=7)) & (distance > 0.1)
            
            # Suppress edges that are too close to local maxima (likely duplicates)
            suppression_mask = ndimage.binary_dilation(local_maxima, iterations=2)
            edge_suppressed = multi_scale_opened * (~suppression_mask | (multi_scale_opened > 0.6))
        else:
            edge_suppressed = multi_scale_opened
        
        # 5. Non-maximum suppression to thin duplicate edges
        # Calculate gradients
        grad_x = cv2.Sobel(edge_suppressed, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(edge_suppressed, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_dir = np.arctan2(grad_y, grad_x)
        
        # Apply non-maximum suppression
        nms_result = np.zeros_like(edge_suppressed)
        h, w = edge_suppressed.shape
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                if edge_suppressed[i, j] < 0.1:  # Skip weak edges
                    continue
                    
                # Get gradient direction (quantized to 4 directions)
                angle = grad_dir[i, j] * 180.0 / np.pi
                angle = angle % 180  # 0-180 degrees
                
                # Check neighbors along gradient direction
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):  # Horizontal
                    neighbors = [edge_suppressed[i, j-1], edge_suppressed[i, j+1]]
                elif 22.5 <= angle < 67.5:  # Diagonal /
                    neighbors = [edge_suppressed[i+1, j-1], edge_suppressed[i-1, j+1]]
                elif 67.5 <= angle < 112.5:  # Vertical
                    neighbors = [edge_suppressed[i-1, j], edge_suppressed[i+1, j]]
                else:  # Diagonal \
                    neighbors = [edge_suppressed[i-1, j-1], edge_suppressed[i+1, j+1]]
                
                # Keep pixel if it's a local maximum
                if edge_suppressed[i, j] >= max(neighbors):
                    nms_result[i, j] = edge_suppressed[i, j]
        
        # 6. Final cleanup - remove remaining thin artifacts
        final_cleaned = morphology.remove_small_objects(
            nms_result > 0.1, min_size=8, connectivity=2
        ).astype(np.float32) * nms_result
        
        return final_cleaned

    def _correct_spatial_alignment(self, edge_map: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Correct spatial misalignment issues caused by network processing.
        More aggressive correction for persistent border/duplicate issues.
        """
        # 1. Ensure the edge map is exactly the right size
        if edge_map.shape != original_shape:
            edge_map = cv2.resize(edge_map, (original_shape[1], original_shape[0]), 
                                interpolation=cv2.INTER_CUBIC)
        
        # 2. Detect and correct systematic shifts using template matching
        # Create a simple edge template from the center region
        h, w = edge_map.shape
        center_h, center_w = h // 4, w // 4
        template = edge_map[center_h:3*center_h, center_w:3*center_w]
        
        if np.sum(template > 0.3) < 10:  # Not enough edges for matching
            return edge_map
            
        # Find the best match location
        result = cv2.matchTemplate(edge_map, template, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        
        # Calculate shift from expected center position
        expected_x, expected_y = center_w, center_h
        actual_x, actual_y = max_loc
        
        shift_x = expected_x - actual_x
        shift_y = expected_y - actual_y
        
        # Apply more aggressive correction (allow up to 10 pixels)
        if abs(shift_x) <= 10 and abs(shift_y) <= 10 and (abs(shift_x) > 1 or abs(shift_y) > 1):
            # Create transformation matrix for translation
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            corrected = cv2.warpAffine(edge_map, M, (w, h), 
                                     flags=cv2.INTER_CUBIC, 
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            logger.debug(f"Applied template-based alignment correction: shift_x={shift_x}, shift_y={shift_y}")
            return corrected
        
        # 3. Fallback: Crop border regions if persistent border artifacts exist
        # Check for consistent edge responses along borders
        border_width = 3
        left_border = np.mean(edge_map[:, :border_width])
        right_border = np.mean(edge_map[:, -border_width:])
        top_border = np.mean(edge_map[:border_width, :])
        bottom_border = np.mean(edge_map[-border_width:, :])
        
        # If any border has consistently high values, crop it out
        crop_left = border_width if left_border > 0.2 else 0
        crop_right = border_width if right_border > 0.2 else 0
        crop_top = border_width if top_border > 0.2 else 0
        crop_bottom = border_width if bottom_border > 0.2 else 0
        
        if any([crop_left, crop_right, crop_top, crop_bottom]):
            cropped = edge_map[crop_top:h-crop_bottom or h, crop_left:w-crop_right or w]
            # Resize back to original size
            corrected = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
            logger.debug(f"Applied border cropping: L={crop_left}, R={crop_right}, T={crop_top}, B={crop_bottom}")
            return corrected
        
        return edge_map

    def _post_process_edges(self, edge_map: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """
        Apply comprehensive post-processing to remove artifacts and improve quality.
        """
        # 1. Correct spatial alignment first
        aligned = self._correct_spatial_alignment(edge_map, original_shape)
        
        # 2. Remove halo effects
        dehalo = self._remove_halo_effects(aligned)
        
        # 3. Apply morphological closing to connect nearby edge segments
        kernel = morphology.disk(1)
        closed = morphology.closing(dehalo, kernel)
        
        # 4. Final edge thinning to ensure single-pixel width edges where appropriate
        thinned = morphology.skeletonize(closed > 0.3)
        
        # 5. Dilate slightly to ensure visibility
        final_edges = morphology.dilation(thinned, morphology.disk(1)).astype(np.float32)
        
        # 6. Combine with strong original edges to preserve important features
        strong_original = (dehalo > 0.7).astype(np.float32)
        final_combined = np.maximum(final_edges, strong_original)
        
        return final_combined

    def detect_hed(self, image: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]

        # 1. Ensure 3-channel BGR image input
        if image.ndim == 2: # Grayscale
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            logger.debug("Converted grayscale input to BGR for HED.")
        elif image.ndim == 3 and image.shape[2] == 4: # RGBA
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            logger.debug("Converted RGBA input to BGR for HED.")
        elif image.ndim == 3 and image.shape[2] == 3: # Already BGR
            image_bgr = image
        else:
            logger.error(f"Unsupported image dimensions: {image.shape}. HED expects 2, 3 or 4 channels.")
            raise ValueError("Unsupported image format for HED detection.")
        
        # Ensure image is uint8 for blobFromImage, assuming it's usually 0-255
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
            logger.debug("Clipped and converted HED input image to uint8.")

        # 2. Use original dimensions with biased padding for network to compensate for shift
        # Pad to make divisible by 32 (deeper networks often need this) and maintain aspect ratio
        pad_h = ((orig_h + 31) // 32) * 32 - orig_h
        pad_w = ((orig_w + 31) // 32) * 32 - orig_w
        
        # Add extra padding to top-left to compensate for HED's bottom-right bias
        extra_pad_top = 4
        extra_pad_left = 4
        
        # Apply biased padding to compensate for systematic shift
        padded_image = cv2.copyMakeBorder(
            image_bgr, 
            pad_h // 2 + extra_pad_top, pad_h - pad_h // 2,  # more top padding
            pad_w // 2 + extra_pad_left, pad_w - pad_w // 2,  # more left padding
            cv2.BORDER_REFLECT_101  # Use reflection padding to avoid edge artifacts
        )
        
        net_h, net_w = padded_image.shape[:2]
        logger.debug(f"HED image padded to {net_w}x{net_h} (divisible by 32) for network input.")

        # 3. Create blob with improved normalization - use the padded image directly
        blob = cv2.dnn.blobFromImage(
            padded_image,
            scalefactor=1.0, 
            size=(net_w, net_h),
            mean=(103.939, 116.779, 123.68), # More precise ImageNet means for HED
            swapRB=False, 
            crop=False
        )
        
        logger.debug(f"HED blob shape: {blob.shape}, dtype: {blob.dtype}")

        self.net.setInput(blob)
        try:
            # Get all output layer names from the network
            output_layer_names = self.net.getUnconnectedOutLayersNames()
            logger.debug(f"Available output layers from HED network: {output_layer_names}")

            # Use only the most reliable DSN layers to avoid halo effects
            # DSN3 often provides the best balance of detail and clean edges
            dsn_layers_to_average = ['sigmoid-dsn2', 'sigmoid-dsn3']
            
            # Also get the fused output for comparison
            all_requested_outputs = list(set(dsn_layers_to_average + ['sigmoid-fuse']))

            outputs = self.net.forward(all_requested_outputs)
            
            # Map output layers to their corresponding blob for easy access
            output_dict = {name: output_blob for name, output_blob in zip(all_requested_outputs, outputs)}

            collected_dsn_masks = []
            for dsn_layer_name in dsn_layers_to_average:
                if dsn_layer_name in output_dict:
                    dsn_output = output_dict[dsn_layer_name][0, 0] # Extract the 2D edge map
                    
                    # Remove padding with compensation for the extra top-left bias
                    unpadded_output = dsn_output[(pad_h//2 + extra_pad_top):dsn_output.shape[0]-(pad_h-pad_h//2),
                                                (pad_w//2 + extra_pad_left):dsn_output.shape[1]-(pad_w-pad_w//2)]
                    
                    # Apply post-processing before resizing to maintain quality
                    processed_output = self._post_process_edges(unpadded_output, (orig_h, orig_w))
                    
                    # Resize to exact original dimensions with high-quality interpolation
                    dsn_mask_resized = cv2.resize(processed_output, (orig_w, orig_h), 
                                                interpolation=cv2.INTER_CUBIC)
                    
                    # Apply final manual offset correction - shift towards top-left
                    final_shift_x = -5  # negative = left
                    final_shift_y = -5  # negative = up
                    M_final = np.float32([[1, 0, final_shift_x], [0, 1, final_shift_y]])
                    dsn_mask_shifted = cv2.warpAffine(dsn_mask_resized, M_final, (orig_w, orig_h), 
                                                    flags=cv2.INTER_CUBIC, 
                                                    borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)
                    
                    collected_dsn_masks.append(dsn_mask_shifted)
                    logger.debug(f"Collected and processed {dsn_layer_name} output. Shape: {dsn_mask_shifted.shape}")
                else:
                    logger.warning(f"DSN layer '{dsn_layer_name}' not found in model outputs.")
            
            if not collected_dsn_masks:
                logger.error("No valid DSN layers were found or collected for averaging.")
                return np.zeros((orig_h, orig_w), dtype=np.uint8)

            # Smart averaging - weight by edge strength to reduce halos
            weights = []
            for mask in collected_dsn_masks:
                # Weight by average edge strength (stronger edges get higher weight)
                weight = np.mean(mask > 0.3) + 0.1  # Add small constant to avoid zero weights
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted average instead of simple mean
            averaged_mask = np.zeros_like(collected_dsn_masks[0])
            for mask, weight in zip(collected_dsn_masks, weights):
                averaged_mask += mask * weight
                
            logger.debug(f"Weighted averaged HED mask shape: {averaged_mask.shape}, "
                        f"min: {np.min(averaged_mask):.4f}, max: {np.max(averaged_mask):.4f}")

            # Final post-processing on the averaged result
            final_processed = self._post_process_edges(averaged_mask, (orig_h, orig_w))
            
            # Convert to uint8 [0, 255] for consistency
            final_mask = np.clip(final_processed * 255, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"HED detection failed during forward pass: {e}")
            logger.debug(f"Available output layers: {self.net.getUnconnectedOutLayersNames()}")
            raise RuntimeError(f"Forward pass failed for HED network: {e}")

        # Final validation
        if final_mask.shape != (orig_h, orig_w):
            logger.error(f"HED mask shape mismatch: expected {(orig_h, orig_w)}, got {final_mask.shape}")
            # Force resize as fallback
            final_mask = cv2.resize(final_mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        logger.debug(f"HED detection completed. Final mask stats: "
                    f"mean={np.mean(final_mask):.2f}, std={np.std(final_mask):.2f}")

        return final_mask