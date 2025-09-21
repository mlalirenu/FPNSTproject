"""
Bayesian Confidence Blending for Adaptive Mask Fusion
Advanced uncertainty modeling for combining multiple edge detection methods.
"""

import cv2
import numpy as np
from loguru import logger
from typing import List, Dict, Tuple, Optional
import asyncio
from scipy.special import softmax
from scipy.ndimage import uniform_filter
import os # Added for saving debug images

class BayesianConfidenceBlender:
    """Bayesian uncertainty modeling for adaptive mask fusion"""
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0, fixed_priors: Optional[Dict[str, float]] = None):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        if fixed_priors:
            # Normalize fixed priors to sum to 1 if they don't already
            total_fixed_prior = sum(fixed_priors.values())
            self.method_priors = {k: v / total_fixed_prior for k, v in fixed_priors.items()}
            logger.info(f"Fixed method priors set: {self.method_priors}")
        else:
            self.method_priors = {} # Will be updated adaptively or use default 0.5 initially
            logger.info("Bayesian confidence blender initialized with adaptive priors.")
        
    def _validate_shape(self, mask: np.ndarray, expected_shape: Tuple[int, int]) -> np.ndarray:
        """Validates and fixes mask shape, converts to float32 [0, 1] range."""
        try:
            if mask.shape != expected_shape:
                logger.warning(f"Mask shape mismatch: expected {expected_shape}, got {mask.shape}. Resizing mask.")
                mask = cv2.resize(mask, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            elif mask.dtype != np.float32:
                # Clip any values outside [0, 1] that might result from previous ops
                mask = np.clip(mask.astype(np.float32), 0.0, 1.0)
            
            # Check for NaNs/Infs after conversion
            if np.any(np.isnan(mask)):
                logger.error("NaNs detected in mask after validation. Replacing with zeros.")
                mask = np.nan_to_num(mask)
            if np.any(np.isinf(mask)):
                logger.error("Infs detected in mask after validation. Clipping to [0,1].")
                mask = np.clip(mask, 0.0, 1.0)

            logger.debug(f"Mask validated: shape={mask.shape}, dtype={mask.dtype}, min={mask.min():.4f}, max={mask.max():.4f}")
            return mask
        except Exception as e:
            logger.error(f"Shape validation failed: {e}. Returning zeros.")
            return np.zeros(expected_shape, dtype=np.float32)
    
    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Ensures mask is within [0, 1] range. Assumes float32 input."""
        return np.clip(mask, 0.0, 1.0)
    
    def _compute_local_entropy(self, mask_norm: np.ndarray, window_size: int = 9) -> np.ndarray:
        """Computes local entropy (approximated by normalized local variance) as uncertainty."""
        try:
            local_mean = uniform_filter(mask_norm, size=window_size, mode='reflect')
            local_sq_mean = uniform_filter(mask_norm**2, size=window_size, mode='reflect')
            local_variance = local_sq_mean - local_mean**2
            
            entropy_map = np.clip(local_variance, 0, None)
            if entropy_map.max() > 1e-6: # Use small epsilon to avoid division by near-zero
                entropy_map = entropy_map / entropy_map.max()
            else:
                entropy_map = np.zeros_like(entropy_map) # All variance is zero, so entropy is zero
            
            logger.debug(f"Entropy map: min={entropy_map.min():.4f}, max={entropy_map.max():.4f}, mean={entropy_map.mean():.4f}")
            return entropy_map
        except Exception as e:
            logger.error(f"Local entropy computation failed: {e}. Returning zeros.")
            return np.zeros_like(mask_norm, dtype=np.float32)
    
    def _compute_gradient_consistency(self, mask_norm: np.ndarray) -> np.ndarray:
        """Computes gradient consistency as confidence, based on inverse of local gradient variance."""
        try:
            # Convert to appropriate dtype for Sobel if not already
            if mask_norm.dtype != np.float32:
                mask_norm = mask_norm.astype(np.float32)

            grad_x = cv2.Sobel(mask_norm, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(mask_norm, cv2.CV_32F, 0, 1, ksize=3)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            kernel_size = 5
            local_mean_grad = uniform_filter(grad_magnitude, size=kernel_size, mode='reflect')
            local_sq_mean_grad = uniform_filter(grad_magnitude**2, size=kernel_size, mode='reflect')
            local_grad_var = local_sq_mean_grad - local_mean_grad**2
            
            local_grad_var = np.clip(local_grad_var, 0, None) # Ensure non-negative variance

            if local_grad_var.max() > 1e-6:
                local_grad_var_norm = local_grad_var / local_grad_var.max()
            else:
                local_grad_var_norm = np.zeros_like(local_grad_var)

            consistency_map = 1.0 / (1.0 + local_grad_var_norm)
            
            logger.debug(f"Gradient consistency map: min={consistency_map.min():.4f}, max={consistency_map.max():.4f}, mean={consistency_map.mean():.4f}")
            return consistency_map
        except Exception as e:
            logger.error(f"Gradient consistency computation failed: {e}. Returning ones.")
            return np.ones_like(mask_norm, dtype=np.float32)
    
    def _compute_spatial_coherence(self, mask_norm: np.ndarray, radius: int = 5) -> np.ndarray:
        """Computes spatial coherence as confidence, based on inverse of local mean difference."""
        try:
            window_size = 2 * radius + 1
            local_mean_neighbors = uniform_filter(mask_norm, size=window_size, mode='reflect')
            differences = np.abs(mask_norm - local_mean_neighbors)
            
            if differences.max() > 1e-6:
                differences_norm = differences / differences.max()
            else:
                differences_norm = np.zeros_like(differences)

            coherence_map = 1.0 - differences_norm
            
            logger.debug(f"Spatial coherence map: min={coherence_map.min():.4f}, max={coherence_map.max():.4f}, mean={coherence_map.mean():.4f}")
            return coherence_map
        except Exception as e:
            logger.error(f"Spatial coherence computation failed: {e}. Returning ones.")
            return np.ones_like(mask_norm, dtype=np.float32)
    
    def _compute_method_confidence(self, mask: np.ndarray, method_name: str) -> np.ndarray:
        """Combines multiple local confidence measures for a given method's mask."""
        try:
            mask_norm = self._normalize_mask(mask)
            
            entropy_conf = 1.0 - self._compute_local_entropy(mask_norm)
            gradient_conf = self._compute_gradient_consistency(mask_norm)
            spatial_conf = self._compute_spatial_coherence(mask_norm)
            
            weights = {'entropy': 0.4, 'gradient': 0.3, 'spatial': 0.3}
            
            combined_confidence = (
                weights['entropy'] * entropy_conf +
                weights['gradient'] * gradient_conf +
                weights['spatial'] * spatial_conf
            )
            
            # Apply method-specific prior, if it exists
            # If method_priors is empty (no fixed_priors and no adaptive learning yet),
            # .get will return default 0.5.
            method_prior_val = self.method_priors.get(method_name, 0.5) 
            combined_confidence *= method_prior_val
            
            logger.debug(f"Method confidence for {method_name}: min={combined_confidence.min():.4f}, max={combined_confidence.max():.4f}, mean={combined_confidence.mean():.4f} (prior: {method_prior_val:.2f})")
            
            return np.clip(combined_confidence, 0.0, 1.0)
        except Exception as e:
            logger.error(f"Method confidence computation failed for {method_name}: {e}. Returning default confidence of 0.5.")
            return np.full_like(mask, 0.5, dtype=np.float32) # Return a neutral confidence
    
    def _update_method_priors(self, method_performances: Dict[str, float]):
        """Updates method priors based on historical performance using a smoothed Bayesian approach."""
        try:
            total_performance = sum(method_performances.values())
            
            # Only update if there's actual performance data and not using fixed priors
            if total_performance > 0 and not self.method_priors: # Check if self.method_priors is empty, i.e., not fixed
                decay_rate = 0.1 # Controls how fast priors adapt
                for method, performance in method_performances.items():
                    if method not in self.method_priors:
                        self.method_priors[method] = self.prior_alpha / (self.prior_alpha + self.prior_beta)
                    
                    success_rate_this_round = performance / total_performance
                    current_prior_mean = self.method_priors[method]
                    
                    new_prior_mean = (1 - decay_rate) * current_prior_mean + decay_rate * success_rate_this_round
                    self.method_priors[method] = np.clip(new_prior_mean, 0.01, 0.99)
            
                logger.debug(f"Updated method priors: {self.method_priors}")
            elif self.method_priors and total_performance > 0:
                logger.debug(f"Using fixed method priors, not updating adaptively: {self.method_priors}")
            else:
                logger.debug("No performance data or using fixed priors, not updating method priors.")
        except Exception as e:
            logger.error(f"Prior update failed: {e}")
    
    def _compute_bayesian_weights(
        self, 
        masks: List[np.ndarray], 
        method_names: List[str],
        confidence_maps: List[np.ndarray]
    ) -> np.ndarray:
        """Computes pixel-wise Bayesian weights for mask fusion using confidence and method priors."""
        try:
            if not (len(masks) == len(method_names) == len(confidence_maps)):
                raise ValueError("Inconsistent input lengths for masks, method names, and confidence maps.")
            
            confidence_maps_stack = np.stack(confidence_maps, axis=0)
            
            # Use fixed prior values if set, otherwise use current adaptive priors (which default to 0.5)
            prior_values = np.array([self.method_priors.get(mn, 0.5) for mn in method_names])
            logger.debug(f"Prior values used for Bayesian weights: {prior_values}")

            raw_weights = confidence_maps_stack * prior_values[:, np.newaxis, np.newaxis]
            
            # Log raw weights before softmax
            logger.debug(f"Raw weights (pre-softmax): min={raw_weights.min():.4f}, max={raw_weights.max():.4f}, mean={raw_weights.mean():.4f}")

            # Use softmax to normalize weights to sum to 1 at each pixel across methods
            # Added a larger constant to avoid potential issues with very small numbers for softmax if all inputs are effectively zero
            # Softmax can make very small differences big if inputs are very small.
            # Adding a larger constant (e.g., 0.1) can stabilize if inputs are expected to be around zero.
            # Let's try 0.1 instead of 1e-6.
            normalized_weights = softmax(raw_weights + 0.1, axis=0) 
            
            logger.debug(f"Normalized Bayesian weights (post-softmax): min={normalized_weights.min():.4f}, max={normalized_weights.max():.4f}, mean={normalized_weights.mean():.4f}")
            
            return normalized_weights
        except Exception as e:
            logger.error(f"Bayesian weight computation failed: {e}. Returning uniform weights.")
            num_methods = len(masks)
            h, w = masks[0].shape[:2]
            return np.ones((num_methods, h, w), dtype=np.float32) / num_methods # Fallback to uniform weights
    
    def _apply_uncertainty_propagation(
        self, 
        fused_mask: np.ndarray, 
        uncertainty_maps: List[np.ndarray],
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Propagates uncertainty from individual masks to the fused mask."""
        try:
            uncertainty_maps_stack = np.stack(uncertainty_maps, axis=0)
            combined_uncertainty = np.sum(weights * uncertainty_maps_stack, axis=0)
            
            # Ensure combined_uncertainty is clipped to [0,1] before using in (1.0 - combined_uncertainty)
            combined_uncertainty = np.clip(combined_uncertainty, 0.0, 1.0)
            
            confidence_adjusted_mask = fused_mask * (1.0 - combined_uncertainty)
            
            logger.debug(f"Combined uncertainty: min={combined_uncertainty.min():.4f}, max={combined_uncertainty.max():.4f}, mean={combined_uncertainty.mean():.4f}")
            logger.debug(f"Confidence adjusted mask (after uncertainty propagation): min={confidence_adjusted_mask.min():.4f}, max={confidence_adjusted_mask.max():.4f}, mean={confidence_adjusted_mask.mean():.4f}")
            
            return confidence_adjusted_mask, combined_uncertainty
        except Exception as e:
            logger.error(f"Uncertainty propagation failed: {e}. Returning original fused mask and zero uncertainty.")
            return fused_mask, np.zeros_like(fused_mask)
    
    async def blend_masks_async(
        self,
        masks: List[np.ndarray],
        method_names: List[str],
        image: Optional[np.ndarray] = None, # Unused for now, kept for API consistency
        adaptive_threshold: bool = True,
        uncertainty_weight: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Asynchronous wrapper for Bayesian mask blending with uncertainty quantification."""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                self._blend_masks_sync,
                masks,
                method_names,
                image,
                adaptive_threshold,
                uncertainty_weight
            )
            return result
        except Exception as e:
            logger.error(f"Async Bayesian mask blending failed: {e}")
            return self._fallback_blend(masks)
    
    def _blend_masks_sync(
        self,
        masks: List[np.ndarray],
        method_names: List[str],
        image: Optional[np.ndarray], # Unused for now
        adaptive_threshold: bool,
        uncertainty_weight: float,
        use_uncertainity = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous core implementation for Bayesian mask blending."""
        try:
            if not masks:
                raise ValueError("No masks provided for blending.")
            
            h, w = masks[0].shape[:2]
            expected_shape = (h, w)
            
            validated_normalized_masks = []
            for i, mask in enumerate(masks):
                # Validate and normalize each input mask to float [0,1]
                val_norm_mask = self._validate_shape(mask, expected_shape)
                validated_normalized_masks.append(val_norm_mask)
               

            confidence_maps = []
            uncertainty_maps = []
            for mask_norm, method_name in zip(validated_normalized_masks, method_names):
                confidence = self._compute_method_confidence(mask_norm, method_name)
                uncertainty = 1.0 - confidence # Uncertainty is inverse of confidence
                confidence_maps.append(confidence)
                uncertainty_maps.append(uncertainty)
                
            bayesian_weights = self._compute_bayesian_weights(validated_normalized_masks, method_names, confidence_maps)
           
            # Calculate the initial fused mask (weighted sum of normalized input masks)
            fused_mask = np.sum(bayesian_weights * np.stack(validated_normalized_masks, axis=0), axis=0)
            
            if use_uncertainity:
                # Apply uncertainty propagation
                confidence_adjusted_mask, combined_uncertainty = self._apply_uncertainty_propagation(
                    fused_mask, uncertainty_maps, bayesian_weights
                )
                
                if adaptive_threshold:
                    base_threshold = 0.5
                    uncertainty_factor = combined_uncertainty * uncertainty_weight
                    adaptive_thresholds = base_threshold + uncertainty_factor
                    # DEBUG LOG: Thresholds before final binarization
                    logger.debug(f"Adaptive thresholds: min={adaptive_thresholds.min():.4f}, max={adaptive_thresholds.max():.4f}, mean={adaptive_thresholds.mean():.4f}")
                    final_mask = (confidence_adjusted_mask > adaptive_thresholds).astype(np.float32)
                else:
                    final_mask = (confidence_adjusted_mask > 0.5).astype(np.float32)
                
                logger.debug(f"Final mask after adaptive thresholding: min={final_mask.min():.4f}, max={final_mask.max():.4f}, sum={final_mask.sum():.0f} (total pixels: {final_mask.size})")

                # Apply uncertainty-aware morphology
                final_mask = self._uncertainty_aware_morphology(final_mask, combined_uncertainty)
                logger.debug(f"Final mask after morphology: min={final_mask.min():.4f}, max={final_mask.max():.4f}, sum={final_mask.sum():.0f}")
                
                # Final conversion to uint8 [0, 255]
                final_mask_uint8 = (np.clip(final_mask, 0.0, 1.0) * 255).astype(np.uint8)
                uncertainty_uint8 = (np.clip(combined_uncertainty, 0.0, 1.0) * 255).astype(np.uint8)
                
                # Final validation after all operations
                final_mask_uint8 = self._validate_shape(final_mask_uint8, expected_shape)
                uncertainty_uint8 = self._validate_shape(uncertainty_uint8, expected_shape)

                # Update method performance for adaptive priors (if not using fixed priors)
                self._update_method_performance(validated_normalized_masks, method_names, final_mask)
                
                return final_mask_uint8, uncertainty_uint8
            
            else:
                return (fused_mask * 255).astype(np.uint8), np.zeros_like(fused_mask, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Bayesian mask blending failed in _blend_masks_sync: {e}")
            return self._fallback_blend(masks)
    
    def _uncertainty_aware_morphology(self, mask_float: np.ndarray, uncertainty_float: np.ndarray) -> np.ndarray:
        """Applies morphological operations tailored by local uncertainty levels."""
        try:
            processed_mask = mask_float.copy()
            
            # Define uncertainty regions
            # Ensure regions are boolean masks of the same shape as mask_float
            low_uncertainty_region = uncertainty_float < 0.3
            med_uncertainty_region = (uncertainty_float >= 0.3) & (uncertainty_float < 0.7)
            high_uncertainty_region = uncertainty_float >= 0.7
            
            # Define kernels
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            
            # Apply morphology only to relevant regions
            if np.any(low_uncertainty_region):
                temp_mask_low = cv2.morphologyEx(mask_float, cv2.MORPH_CLOSE, kernel_large)
                processed_mask[low_uncertainty_region] = temp_mask_low[low_uncertainty_region]
            
            if np.any(med_uncertainty_region):
                temp_mask_med = cv2.morphologyEx(mask_float, cv2.MORPH_CLOSE, kernel_medium)
                processed_mask[med_uncertainty_region] = temp_mask_med[med_uncertainty_region]
            
            if np.any(high_uncertainty_region):
                temp_mask_high = cv2.morphologyEx(mask_float, cv2.MORPH_OPEN, kernel_small)
                processed_mask[high_uncertainty_region] = temp_mask_high[high_uncertainty_region]
            
            logger.debug("Applied uncertainty-aware morphology.")
            return processed_mask
        except Exception as e:
            logger.error(f"Uncertainty-aware morphology failed: {e}. Returning original mask.")
            return mask_float
    
    def _update_method_performance(self, masks_norm: List[np.ndarray], method_names: List[str], final_mask_float: np.ndarray):
        """Calculates performance of each method against the final fused mask and updates priors."""
        try:
            method_performances = {}
            for mask_norm, method_name in zip(masks_norm, method_names):
                # Ensure final_mask_float is aligned in size
                if mask_norm.shape != final_mask_float.shape:
                    logger.warning(f"Shape mismatch in _update_method_performance for {method_name}. Resizing final_mask_float.")
                    final_mask_float = cv2.resize(final_mask_float, (mask_norm.shape[1], mask_norm.shape[0]), interpolation=cv2.INTER_LINEAR)

                agreement = 1.0 - np.mean(np.abs(mask_norm - final_mask_float))
                edge_consistency = np.mean(self._compute_gradient_consistency(mask_norm)) # Re-use gradient consistency as a metric
                edge_density = np.mean(mask_norm)
                
                # Performance metric: weighted sum of agreement, consistency, and a term for balanced density
                performance = 0.5 * agreement + 0.3 * edge_consistency + 0.2 * (1.0 - abs(edge_density - 0.1)) # 0.1 is an ideal edge density target
                method_performances[method_name] = performance
            
            self._update_method_priors(method_performances)
        except Exception as e:
            logger.error(f"Method performance update failed: {e}")
    
    def _fallback_blend(self, masks: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Provides a simple average blend as a fallback in case of errors."""
        try:
            if not masks:
                h, w = (500, 500) # Default size if no masks
                logger.warning("No masks provided for fallback blend. Returning default empty masks.")
                return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
            
            h, w = masks[0].shape[:2]
            expected_shape = (h, w)
            normalized_masks = [self._validate_shape(mask, expected_shape) for mask in masks] # Ensure float [0,1]
            
            averaged = np.mean(normalized_masks, axis=0)
            uncertainty = np.var(normalized_masks, axis=0)
            uncertainty = uncertainty / (uncertainty.max() + 1e-10) # Normalize uncertainty
            
            # Apply a simple threshold for the fallback final mask
            final_mask = (averaged > 0.5).astype(np.uint8) * 255
            uncertainty_mask = (uncertainty * 255).astype(np.uint8)
            
            logger.info("Performed fallback blending due to error.")
            return final_mask, uncertainty_mask
        except Exception as e:
            logger.error(f"Fallback blending itself failed: {e}. Returning black masks.")
            h, w = masks[0].shape[:2] if masks else (500, 500)
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    
    def blend_masks(
        self,
        masks: List[np.ndarray],
        method_names: List[str],
        image: Optional[np.ndarray] = None,
        adaptive_threshold: bool = True,
        uncertainty_weight: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Synchronous wrapper for Bayesian mask blending."""
        return self._blend_masks_sync(masks, method_names, image, adaptive_threshold, uncertainty_weight)
    
    def compute_ensemble_uncertainty(
        self, 
        masks: List[np.ndarray], 
        method_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Computes and returns various types of uncertainty maps for the ensemble of masks."""
        try:
            if not masks:
                logger.warning("No masks provided for ensemble uncertainty computation. Returning empty dict.")
                return {}
            
            h, w = masks[0].shape[:2]
            expected_shape = (h, w)
            normalized_masks = [self._validate_shape(mask, expected_shape) for mask in masks]
            masks_stack = np.stack(normalized_masks, axis=0)
            
            # Epistemic uncertainty (model uncertainty, disagreement between models)
            epistemic = np.var(masks_stack, axis=0)
            
            # Aleatoric uncertainty (inherent noise/randomness in data) - approximated by local entropy
            aleatoric_maps = [self._compute_local_entropy(mask_norm) for mask_norm in normalized_masks]
            aleatoric = np.mean(aleatoric_maps, axis=0)
            
            total = epistemic + aleatoric
            
            # Normalize uncertainties to [0,1] for consistent visualization/interpretation
            uncertainties = {
                'epistemic': epistemic / (epistemic.max() + 1e-10) if epistemic.max() > 0 else epistemic,
                'aleatoric': aleatoric / (aleatoric.max() + 1e-10) if aleatoric.max() > 0 else aleatoric,
                'total': total / (total.max() + 1e-10) if total.max() > 0 else total
            }
            
            for mask_norm, method_name in zip(normalized_masks, method_names):
                method_uncertainty = 1.0 - self._compute_method_confidence(mask_norm, method_name)
                uncertainties[f'{method_name}_uncertainty'] = method_uncertainty # These are specific per-method uncertainty maps
            
            return uncertainties
        except Exception as e:
            logger.error(f"Ensemble uncertainty computation failed: {e}")
            return {}
    
    def get_blending_statistics(
        self, 
        blended_mask: np.ndarray, 
        uncertainty_map: np.ndarray,
        original_masks: Optional[List[np.ndarray]] = None,
        method_names: Optional[List[str]] = None
    ) -> dict:
        """Gathers comprehensive statistics about the blending process and results."""
        try:
            # Normalize inputs to float [0,1] for consistent calculations
            blended_norm = self._validate_shape(blended_mask, blended_mask.shape[:2])
            uncertainty_norm = self._validate_shape(uncertainty_map, uncertainty_map.shape[:2])

            total_pixels = blended_norm.size
            edge_pixels = np.sum(blended_norm > 0)
            edge_density = edge_pixels / total_pixels
            
            mean_uncertainty = np.mean(uncertainty_norm)
            uncertainty_std = np.std(uncertainty_norm)
            high_uncertainty_pixels = np.sum(uncertainty_norm > 0.5)
            
            stats = {
                'total_pixels': total_pixels,
                'edge_pixels': int(edge_pixels),
                'edge_density': float(edge_density),
                'mean_uncertainty': float(mean_uncertainty),
                'uncertainty_std': float(uncertainty_std),
                'high_uncertainty_pixels': int(high_uncertainty_pixels),
                'high_uncertainty_ratio': float(high_uncertainty_pixels / total_pixels) if total_pixels > 0 else 0.0,
                'method_priors': dict(self.method_priors)
            }
            
            if original_masks and method_names:
                agreements = {}
                for mask, method_name in zip(original_masks, method_names):
                    mask_norm = self._validate_shape(mask, blended_mask.shape[:2])
                    agreement = 1.0 - np.mean(np.abs(mask_norm - blended_norm))
                    agreements[method_name] = float(agreement)
                stats['method_agreements'] = agreements
            
            return stats
        except Exception as e:
            logger.error(f"Blending statistics computation failed: {e}")
            return {
                'total_pixels': 0, 'edge_pixels': 0, 'edge_density': 0.0,
                'mean_uncertainty': 0.0, 'uncertainty_std': 0.0,
                'high_uncertainty_pixels': 0, 'high_uncertainty_ratio': 0.0,
                'method_priors': {}, 'error': str(e)
            }
    
    def save_priors(self, filepath: str):
        """Saves learned method priors to a JSON file."""
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(self.method_priors, f, indent=2)
            logger.info(f"Method priors saved to {filepath}.")
        except Exception as e:
            logger.error(f"Failed to save method priors to {filepath}: {e}")
    
    def load_priors(self, filepath: str):
        """Loads method priors from a JSON file."""
        try:
            import json
            with open(filepath, 'r') as f:
                self.method_priors = json.load(f)
            logger.info(f"Method priors loaded from {filepath}.")
        except Exception as e:
            logger.error(f"Failed to load method priors from {filepath}: {e}")