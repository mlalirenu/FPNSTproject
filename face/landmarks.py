import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Tuple

mp_face_mesh = mp.solutions.face_mesh

def detect_face_landmarks(image: np.ndarray) -> List[List[Tuple[float, float]]]:
    """FIXED: Detailed face landmark detection including face contour with improved error handling"""
    try:
        # Input validation and preprocessing
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("Invalid image format")
            return []

        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Optional: Resize if image is too small
        h, w = rgb_image.shape[:2]
        min_size = 300
        if min(h, w) < min_size:
            scale = min_size / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            rgb_image = cv2.resize(rgb_image, (new_w, new_h))
            scale_back = True
        else:
            scale_back = False
            scale = 1.0

        # Process with error handling and retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create a copy of the image to prevent modification
                image_copy = rgb_image.copy()
                
                # Ensure image is contiguous and writable
                if not image_copy.flags.writeable:
                    image_copy = np.copy(image_copy)
                
                with mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=10,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                ) as face_mesh:
                    results = face_mesh.process(image_copy)
                
                if results and results.multi_face_landmarks:
                    print(f"Successfully detected {len(results.multi_face_landmarks)} faces on attempt {attempt + 1}")
                    face_landmarks = []
                    for face_landmark in results.multi_face_landmarks:
                        landmarks = []
                        for landmark in face_landmark.landmark:
                            # Scale coordinates back if image was resized
                            if scale_back:
                                x = landmark.x / scale
                                y = landmark.y / scale
                            else:
                                x = landmark.x
                                y = landmark.y
                            landmarks.append((x, y))
                        face_landmarks.append(landmarks)
                    return face_landmarks
                else:
                    print(f"No faces detected on attempt {attempt + 1}")
                    if attempt == max_retries - 1:  # Last attempt
                        return []
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:  # Last attempt
                    print("All attempts failed, returning empty list")
                    return []
                # Short delay before retry
                time.sleep(0.1)
                
    except Exception as e:
        print(f"Fatal error in face landmark detection: {str(e)}")
        return []
    
    return []