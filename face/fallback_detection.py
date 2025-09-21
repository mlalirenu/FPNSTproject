import cv2
import numpy as np
import torch
import mediapipe as mp
import time
from typing import Dict, List, Tuple

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range (2 meters), 1 for full-range (5 meters)
    min_detection_confidence=0.3  # Lowered threshold for better detection
)

def detect_faces_basic(image: np.ndarray) -> List[Dict]:
    """Basic face detection using MediaPipe Face Detection"""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_image)
    
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = image.shape
            
            # Convert relative coordinates to pixel coordinates
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            
            # Expand bounding box to include ears (add 20% padding on sides)
            padding_x = int(w * 0.2)
            padding_y = int(h * 0.1)
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(iw - x, w + 2 * padding_x)
            h = min(ih - y, h + 2 * padding_y)
            
            faces.append({
                'bbox': (x, y, w, h),
                'confidence': detection.score[0]
            })

            print(f"Detected face at: {x}, {y}, {w}, {h} with confidence: {detection.score[0]}")

    return faces


def fallback_face_detection(image: np.ndarray) -> np.ndarray:
    """Enhanced fallback face detection using cascaded approaches"""
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    print("Using fallback face detection")
    
    # First try basic face detection
    faces = detect_faces_basic(image)
    
    if faces:
        print(f"Fallback: Found {len(faces)} faces with basic detection")
        for face in faces:
            if face['confidence'] > 0.3:  # Allow lower confidence in fallback
                x, y, w_box, h_box = face['bbox']
                
                # Create multiple elliptical masks for better coverage
                # Main face ellipse
                center = (x + w_box//2, y + h_box//2)
                axes = (int(w_box*0.65), int(h_box*0.85))
                cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)
                
                # Slightly larger ellipse with lower opacity
                axes_large = (int(w_box*0.75), int(h_box*0.95))
                temp_mask = np.zeros_like(face_mask)
                cv2.ellipse(temp_mask, center, axes_large, 0, 0, 360, 128, -1)
                face_mask = cv2.max(face_mask, temp_mask)
                
                # Expanded region for ears and hair
                center = (x + w_box//2, y + h_box//2)
                # Extended region for ears (wider ellipse)
                ear_axes = (int(w_box*0.9), int(h_box*0.7))
                temp_mask = np.zeros_like(face_mask)
                cv2.ellipse(temp_mask, center, ear_axes, 0, 0, 360, 180, -1)
                face_mask = cv2.max(face_mask, temp_mask)
                
                # Hair region (taller ellipse with gradient)
                hair_center = (center[0], center[1] - int(h_box*0.1))  # Shift up slightly
                hair_axes = (int(w_box*0.8), int(h_box*1.2))
                temp_mask = np.zeros_like(face_mask)
                cv2.ellipse(temp_mask, hair_center, hair_axes, 0, 0, 360, 150, -1)
                
                # Create gradual falloff for hair region
                temp_mask = cv2.GaussianBlur(temp_mask, (15, 15), 5)
                face_mask = cv2.max(face_mask, temp_mask)
    else:
        print("Fallback: No faces detected with basic detection either")

    # Enhanced smoothing with larger kernels for better blending
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
    face_mask = cv2.GaussianBlur(face_mask, (13, 13), 4)
    
    return face_mask