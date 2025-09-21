import cv2
import numpy as np
from typing import List, Tuple

from face.fallback_detection import fallback_face_detection

def create_precise_face_mask(image: np.ndarray, face_landmarks: List[List[Tuple[float, float]]], 
                            include_ears: bool = True) -> np.ndarray:
    """Create pixel-perfect mask using detailed face contour landmarks"""
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not face_landmarks:
        # Fallback to basic detection if no landmarks
        print("No landmarks provided, using fallback detection")
        return fallback_face_detection(image)
    
    print(f"Creating precise face mask with {len(face_landmarks)} faces")
    
    # MediaPipe Face Mesh complete face outline indices (more comprehensive)
    FACE_OVAL = [
        # Complete face boundary - clockwise from top
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    # Extended boundary with forehead and hair area
    EXTENDED_BOUNDARY = [
        # Forehead and hairline
        9, 10, 151, 337, 299, 333, 298, 301, 368, 389, 356, 454, 323, 361, 288,
        # Right side of face
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
        # Chin and jaw
        132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        # Left side back to forehead
        10, 338, 297, 332, 284, 251, 389
    ]
    
    for i, landmarks in enumerate(face_landmarks):
        print(f"Processing face {i+1} with {len(landmarks)} landmarks")
        
        # Try multiple approaches for better coverage
        masks_to_combine = []
        
        # Method 1: Use predefined contours
        contour_indices = EXTENDED_BOUNDARY if include_ears else FACE_OVAL
        contour_points = []
        valid_indices = 0
        
        for idx in contour_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx][0] * w)
                y = int(landmarks[idx][1] * h)
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                contour_points.append([x, y])
                valid_indices += 1
        
        print(f"Valid contour points: {valid_indices}/{len(contour_indices)}")
        
        if len(contour_points) > 3:
            contour_points = np.array(contour_points, dtype=np.int32)
            hull = cv2.convexHull(contour_points)
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [hull], 255)
            masks_to_combine.append(temp_mask)
            print(f"Created contour mask with {len(hull)} hull points")
        
        # Method 2: Use outer boundary landmarks with expansion
        outer_landmarks = landmarks[::8]  # Sample every 8th landmark for outer boundary  
        if len(outer_landmarks) > 10:
            outer_points = []
            for landmark in outer_landmarks:
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                x = max(0, min(w-1, x))
                y = max(0, min(h-1, y))
                outer_points.append([x, y])
            
            outer_points = np.array(outer_points, dtype=np.int32)
            hull = cv2.convexHull(outer_points)
            
            # Expand the hull slightly for better coverage
            center = np.mean(hull, axis=0)
            expanded_hull = []
            for point in hull:
                # Expand each point outward from center by 5%
                direction = point - center
                expanded_point = point + direction * 0.05
                expanded_hull.append(expanded_point)
            
            expanded_hull = np.array(expanded_hull, dtype=np.int32)
            temp_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(temp_mask, [expanded_hull], 255)
            masks_to_combine.append(temp_mask)
            print(f"Created expanded boundary mask")
        
        # Combine all masks
        if masks_to_combine:
            combined_mask = masks_to_combine[0]
            for mask in masks_to_combine[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Apply morphological operations for smoother edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_DILATE, kernel, iterations=1)
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)
            
            face_mask = cv2.bitwise_or(face_mask, combined_mask)
            print(f"Combined mask for face {i+1}")
        else:
            print(f"No valid masks created for face {i+1}")
    
    return face_mask

def create_segmentation_face_mask(self, image: np.ndarray, face_landmarks: List[List[Tuple[float, float]]]) -> np.ndarray:
    """Create ultra-precise segmentation mask using complete face shape"""
    h, w = image.shape[:2]
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    if not face_landmarks:
        return fallback_face_detection(image)
    
    for landmarks in face_landmarks:
        # Convert all landmarks to pixel coordinates
        points = []
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            x = max(0, min(w-1, x))
            y = max(0, min(h-1, y))
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        if len(points) >= 50:  # Ensure we have enough landmarks
            # Method 1: Alpha shape / concave hull for more natural face boundary
            try:
                # Use all points to create comprehensive mask
                hull = cv2.convexHull(points)
                cv2.fillPoly(face_mask, [hull], 255)
                
                # Create additional mask from boundary points only
                boundary_points = points[::5]  # Every 5th point for boundary
                if len(boundary_points) > 10:
                    boundary_hull = cv2.convexHull(boundary_points)
                    
                    # Expand boundary slightly
                    center = np.mean(boundary_hull, axis=0)
                    expanded_boundary = []
                    for point in boundary_hull:
                        direction = point - center
                        expanded_point = point + direction * 0.08  # 8% expansion
                        expanded_boundary.append(expanded_point)
                    
                    expanded_boundary = np.array(expanded_boundary, dtype=np.int32)
                    temp_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(temp_mask, [expanded_boundary], 255)
                    
                    # Combine masks
                    face_mask = cv2.bitwise_or(face_mask, temp_mask)
            
            except Exception:
                # Fallback to simple convex hull
                hull = cv2.convexHull(points)
                cv2.fillPoly(face_mask, [hull], 255)
        
        # Smooth and expand the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_CLOSE, kernel)
        face_mask = cv2.morphologyEx(face_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        face_mask = cv2.GaussianBlur(face_mask, (7, 7), 0)
    
    return face_mask