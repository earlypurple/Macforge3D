import cv2
import numpy as np
import os
from pathlib import Path
import trimesh
from typing import List, Tuple, Optional

def extract_features(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract SIFT features from an image."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return np.float32([kp.pt for kp in keypoints]), descriptors

def match_features(desc1: np.ndarray, desc2: np.ndarray) -> List[Tuple[int, int]]:
    """Match features between two images using FLANN."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    
    return good

def reconstruct_points(K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Reconstruct 3D points from 2D correspondences."""
    pts1_norm = cv2.undistortPoints(pts1, K, None)
    pts2_norm = cv2.undistortPoints(pts2, K, None)
    
    points_4d = cv2.triangulatePoints(
        np.eye(3, 4),  # First camera matrix [I|0]
        np.hstack((R, t.reshape(3, 1))),  # Second camera matrix [R|t]
        pts1_norm,
        pts2_norm
    )
    
    points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous to 3D
    return points_3d.T

def create_point_cloud(image_paths: List[str], output_path: str) -> Optional[str]:
    """Create a point cloud from a series of images."""
    try:
        # Estimate camera matrix (assuming a standard configuration)
        focal_length = 1000  # can be adjusted based on your camera
        center = (640, 480)  # assuming 1280x960 images
        K = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])

        all_points_3d = []
        
        # Process image pairs
        for i in range(len(image_paths) - 1):
            # Extract features
            pts1, desc1 = extract_features(image_paths[i])
            pts2, desc2 = extract_features(image_paths[i + 1])
            
            # Match features
            matches = match_features(desc1, desc2)
            
            if len(matches) < 8:
                continue
                
            # Get matched points
            pts1_matched = np.float32([pts1[m[0]] for m in matches])
            pts2_matched = np.float32([pts2[m[1]] for m in matches])
            
            # Essential matrix
            E, mask = cv2.findEssentialMat(pts1_matched, pts2_matched, K)
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1_matched, pts2_matched, K)
            
            # Triangulate points
            points_3d = reconstruct_points(
                K, 
                pts1_matched[mask.ravel() == 1], 
                pts2_matched[mask.ravel() == 1],
                R, t
            )
            
            all_points_3d.extend(points_3d)

        if not all_points_3d:
            return None

        # Convert to point cloud
        points = np.array(all_points_3d)
        
        # Create mesh from point cloud using trimesh
        cloud = trimesh.PointCloud(points)
        
        # Save as PLY file
        output_path = str(Path(output_path).with_suffix('.ply'))
        cloud.export(output_path)
        
        return output_path
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        return None

if __name__ == "__main__":
    # Test with example images
    test_image_dir = "Examples/photogrammetry_test_images"
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Create test images if they don't exist
    test_images = []
    for i in range(5):
        path = Path(test_image_dir) / f"test_image_{i}.jpg"
        if not path.exists():
            img = np.zeros((960, 1280, 3), dtype=np.uint8)
            cv2.putText(img, f"Test {i}", (100, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            cv2.imwrite(str(path), img)
        test_images.append(str(path))
    
    output_path = "Examples/generated_photogrammetry/test_reconstruction.ply"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    result = create_point_cloud(test_images, output_path)
    if result:
        print(f"Successfully created point cloud: {result}")
    else:
        print("Failed to create point cloud")
