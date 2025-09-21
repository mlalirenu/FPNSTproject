import asyncio
import numpy as np
from loguru import logger

from face.fallback_detection import fallback_face_detection
from face.landmarks import detect_face_landmarks
from face.face_masking import create_precise_face_mask, create_segmentation_face_mask

async def get_face_mask(image: np.ndarray, method: str = 'precise', include_ears: bool = True) -> np.ndarray:
    """Enhanced face detection with multiple methods"""
    logger.info("Running face detection in thread:")
    
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _detect_faces_sync, image, method, include_ears)
    except RuntimeError:
        logger.warning("No running event loop, executing synchronously")
        return _detect_faces_sync(image, method, include_ears)

def _detect_faces_sync(image: np.ndarray, method: str = 'precise', 
                        include_ears: bool = True) -> np.ndarray:
    """Enhanced synchronous face detection with better fallback handling"""
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)

    print(f"Starting face detection with method: {method}")
    print(f"Image shape: {image.shape}")

    try:
        if method == 'precise':
            face_landmarks = detect_face_landmarks(image)
            if face_landmarks:
                print(f"Precise method: Found landmarks for {len(face_landmarks)} faces")
                face_mask = create_precise_face_mask(image, face_landmarks, include_ears)
            else:
                print("Precise method: No face landmarks detected, using enhanced fallback")
                face_mask = fallback_face_detection(image)
        
        elif method == 'segmentation':
            face_landmarks = detect_face_landmarks(image)
            if face_landmarks:
                print(f"Segmentation method: Found landmarks for {len(face_landmarks)} faces")
                face_mask = create_segmentation_face_mask(image, face_landmarks)
            else:
                print("Segmentation method: No face landmarks detected, using enhanced fallback")
                face_mask = fallback_face_detection(image)
        
        else:
            print(f"Unknown method: {method}, using precise detection with fallback")
            face_landmarks = detect_face_landmarks(image)
            if face_landmarks:
                face_mask = create_precise_face_mask(image, face_landmarks, include_ears)
            else:
                face_mask = fallback_face_detection(image)

    except Exception as e:
        print(f"Face detection error: {str(e)}. Using enhanced fallback method.")
        face_mask = fallback_face_detection(image)

    # Check if we got any face detection at all
    if np.max(face_mask) == 0:
        print("WARNING: No faces detected by any method!")
    else:
        print(f"Face detection successful. Mask max value: {np.max(face_mask)}")

    return face_mask