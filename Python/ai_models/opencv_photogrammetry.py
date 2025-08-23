import cv2
import numpy as np
import os
from pathlib import Path
import trimesh
from typing import List, Tuple, Optional

def extract_features(image_path: str, detector_type: str = "SIFT") -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from an image using various detectors.
    
    Args:
        image_path: Path to the input image
        detector_type: Type of feature detector ("SIFT", "ORB", "AKAZE", "SURF")
    
    Returns:
        Tuple of (keypoints, descriptors)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Select detector based on type
    if detector_type == "SIFT":
        detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.04, edgeThreshold=10)
    elif detector_type == "ORB":
        detector = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    elif detector_type == "AKAZE":
        detector = cv2.AKAZE_create()
    else:  # Default to SIFT
        detector = cv2.SIFT_create(nfeatures=5000)
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    
    if descriptors is None:
        return np.array([]), np.array([])
    
    return np.float32([kp.pt for kp in keypoints]), descriptors

def match_features(desc1: np.ndarray, desc2: np.ndarray, ratio_threshold: float = 0.75) -> List[Tuple[int, int]]:
    """Match features between two images using enhanced FLANN matching.
    
    Args:
        desc1: Descriptors from first image
        desc2: Descriptors from second image
        ratio_threshold: Lowe's ratio threshold for filtering matches
    
    Returns:
        List of good matches as (query_idx, train_idx) tuples
    """
    if desc1.size == 0 or desc2.size == 0:
        return []
    
    # Use different matching strategy based on descriptor type
    if desc1.dtype == np.uint8:  # ORB descriptors
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(desc1, desc2, k=2)
    else:  # SIFT/SURF descriptors
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test with enhanced filtering
    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good.append((m.queryIdx, m.trainIdx))
    
    return good

def reconstruct_points(K: np.ndarray, pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Reconstruct 3D points from 2D correspondences with enhanced validation.
    
    Args:
        K: Camera intrinsic matrix
        pts1: Points in first image
        pts2: Points in second image  
        R: Rotation matrix between cameras
        t: Translation vector between cameras
    
    Returns:
        3D points array
    """
    if len(pts1) == 0 or len(pts2) == 0:
        return np.array([])
    
    # Ensure points are in correct format
    pts1 = np.array(pts1, dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array(pts2, dtype=np.float32).reshape(-1, 1, 2)
    
    # Undistort points
    pts1_norm = cv2.undistortPoints(pts1, K, None)
    pts2_norm = cv2.undistortPoints(pts2, K, None)
    
    # Create projection matrices
    P1 = K @ np.eye(3, 4)  # First camera matrix [K|0]
    P2 = K @ np.hstack((R, t.reshape(3, 1)))  # Second camera matrix [K*[R|t]]
    
    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1_norm, pts2_norm)
    
    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / (points_4d[3] + 1e-8)  # Add small epsilon to avoid division by zero
    points_3d = points_3d.T
    
    # Filter out points that are too far or have negative depth
    valid_mask = (points_3d[:, 2] > 0) & (np.linalg.norm(points_3d, axis=1) < 1000)
    
    return points_3d[valid_mask]

def estimate_camera_poses(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate camera poses using essential matrix decomposition.
    
    Args:
        pts1: Points in first image
        pts2: Points in second image
        K: Camera intrinsic matrix
    
    Returns:
        Tuple of (Essential matrix, Rotation matrix, Translation vector)
    """
    # Find essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, 
        method=cv2.RANSAC, 
        prob=0.999, 
        threshold=1.0,
        maxIters=5000
    )
    
    if E is None:
        raise ValueError("Could not compute essential matrix")
    
    # Recover pose from essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    
    return E, R, t

def create_point_cloud(
    image_paths: List[str], 
    output_path: str, 
    detector_type: str = "SIFT",
    quality: str = "balanced",
    min_matches: int = 50
) -> Optional[str]:
    """Create an enhanced point cloud from a series of images.
    
    Args:
        image_paths: List of image file paths
        output_path: Output path for the point cloud file
        detector_type: Feature detector type ("SIFT", "ORB", "AKAZE")
        quality: Quality preset ("fast", "balanced", "high")
        min_matches: Minimum number of matches required between image pairs
    
    Returns:
        Path to created point cloud file or None if failed
    """
    try:
        print(f"🔄 Processing {len(image_paths)} images with {detector_type} detector...")
        print(f"⚙️  Quality: {quality}, Min matches: {min_matches}")
        
        if len(image_paths) < 2:
            print("❌ Need at least 2 images for photogrammetry")
            return None
        
        # Quality-based parameters
        quality_params = {
            "fast": {"focal_mult": 1.0, "ratio_thresh": 0.8, "ransac_thresh": 2.0},
            "balanced": {"focal_mult": 1.2, "ratio_thresh": 0.75, "ransac_thresh": 1.5},
            "high": {"focal_mult": 1.5, "ratio_thresh": 0.7, "ransac_thresh": 1.0}
        }
        params = quality_params.get(quality, quality_params["balanced"])
        
        # Load first image to get dimensions for camera matrix estimation
        first_img = cv2.imread(image_paths[0])
        if first_img is None:
            print(f"❌ Could not load first image: {image_paths[0]}")
            return None
            
        height, width = first_img.shape[:2]
        
        # Enhanced camera matrix estimation
        focal_length = max(width, height) * params["focal_mult"]
        center = (width / 2.0, height / 2.0)
        K = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        print(f"📷 Estimated camera parameters: f={focal_length:.1f}, center=({center[0]:.1f}, {center[1]:.1f})")

        all_points_3d = []
        successful_pairs = 0
        
        # Process image pairs with enhanced pipeline
        for i in range(len(image_paths) - 1):
            try:
                print(f"🔍 Processing pair {i+1}/{len(image_paths)-1}: {Path(image_paths[i]).name} <-> {Path(image_paths[i+1]).name}")
                
                # Extract features with selected detector
                pts1, desc1 = extract_features(image_paths[i], detector_type)
                pts2, desc2 = extract_features(image_paths[i + 1], detector_type)
                
                print(f"   Features: {len(pts1)} <-> {len(pts2)}")
                
                if len(pts1) < 50 or len(pts2) < 50:
                    print(f"   ⚠️  Insufficient features, skipping pair")
                    continue
                
                # Match features with quality-based threshold
                matches = match_features(desc1, desc2, params["ratio_thresh"])
                print(f"   Matches: {len(matches)}")
                
                if len(matches) < min_matches:
                    print(f"   ⚠️  Insufficient matches ({len(matches)} < {min_matches}), skipping pair")
                    continue
                    
                # Get matched points
                pts1_matched = np.float32([pts1[m[0]] for m in matches])
                pts2_matched = np.float32([pts2[m[1]] for m in matches])
                
                # Estimate camera poses with enhanced parameters
                try:
                    E, R, t = estimate_camera_poses(pts1_matched, pts2_matched, K)
                    
                    # Triangulate points with validation
                    points_3d = reconstruct_points(K, pts1_matched, pts2_matched, R, t)
                    
                    if len(points_3d) > 0:
                        all_points_3d.extend(points_3d)
                        successful_pairs += 1
                        print(f"   ✅ Reconstructed {len(points_3d)} 3D points")
                    else:
                        print(f"   ⚠️  No valid 3D points reconstructed")
                        
                except Exception as e:
                    print(f"   ⚠️  Pose estimation failed: {e}")
                    continue
                    
            except Exception as e:
                print(f"   ❌ Error processing pair {i+1}: {e}")
                continue

        print(f"📊 Successfully processed {successful_pairs}/{len(image_paths)-1} image pairs")
        
        if not all_points_3d:
            print("❌ No 3D points could be reconstructed")
            return None

        # Convert to numpy array and apply filtering
        points = np.array(all_points_3d)
        print(f"🎯 Total reconstructed points: {len(points)}")
        
        # Enhanced point cloud filtering
        if len(points) > 1000:  # Only filter if we have enough points
            # Remove statistical outliers
            from scipy.spatial.distance import cdist
            from scipy.stats import zscore
            
            # Filter by Z-score
            z_scores = np.abs(zscore(points, axis=0))
            outlier_mask = np.any(z_scores > 2.5, axis=1)
            points = points[~outlier_mask]
            print(f"🧹 After outlier removal: {len(points)} points")
        
        # Create and save point cloud
        colors = np.full((len(points), 3), [128, 128, 128], dtype=np.uint8)  # Gray color
        
        # Save as PLY file
        try:
            import trimesh
            cloud = trimesh.PointCloud(vertices=points, colors=colors)
            cloud.export(output_path)
            print(f"✅ Point cloud saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"⚠️  Trimesh export failed, using basic PLY format: {e}")
            # Fallback to manual PLY writing
            with open(output_path, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                
                for i, (point, color) in enumerate(zip(points, colors)):
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
            
            print(f"✅ Point cloud saved (basic format): {output_path}")
            return output_path
        
    except Exception as e:
        print(f"❌ Critical error in point cloud creation: {e}")
        return None
        
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
