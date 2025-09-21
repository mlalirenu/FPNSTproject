"""
This provides methods for edge-preserving saliency detection using
frequency domain techniques.
"""

import cv2
import numpy as np
from typing import Optional

class FrequencyDomainPreservation:
    def __init__(self):
        pass

    def detect_frequency_domain(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Extracts and enhances high-frequency components of the image for edge-preserving saliency.

        Args:
            image (np.ndarray): Input image (BGR or grayscale).
            strength (float): Amplification factor for high-frequency components.

        Returns:
            np.ndarray: Binary mask emphasizing high-frequency (edge) regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = gray.astype(np.float32) / 255.0

        # Perform FFT
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)

        # Create a high-pass filter mask
        rows, cols = gray.shape
        crow, ccol = rows // 2 , cols // 2
        r = min(crow, ccol) // 4
        mask = np.ones((rows, cols), np.uint8)
        cv2.circle(mask, (ccol, crow), r, 0, -1)

        # Apply mask and inverse FFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # Normalize and amplify
        #img_back = (img_back - img_back.min()) / (img_back.ptp() + 1e-8)
        img_back = (img_back - img_back.min()) / (np.ptp(img_back) + 1e-8)
        img_back = np.clip(img_back * strength, 0, 1)

        # Threshold to get binary edge-like mask
        mask_out = (img_back > 0.2).astype(np.uint8) * 255
        return mask_out
