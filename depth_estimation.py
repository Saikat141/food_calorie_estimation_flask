"""
Depth Estimation Module using Depth Pro for volume calculation
"""

import numpy as np
import cv2
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os

class DepthEstimator:
    """Depth estimation using Depth Pro model for food volume calculation"""

    def __init__(self, model_path: str = "models/depth_pro.pt"):
        """Initialize the depth estimator"""
        self.model = None
        self.transform = None
        self.model_path = model_path
        self.device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            if self.device == 'mps':
                torch.set_float32_matmul_precision('medium')
        except Exception:
            pass
        self.load_depth_pro()

    def load_depth_pro(self):
        """Load Depth Pro model and transforms"""
        try:
            # Check if local model file exists
            import os
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠ Local depth model not found at {self.model_path}. Using fallback depth estimation.")
                self.model = None
                self.transform = None
                return

            # Try to load the model using PyTorch directly if depth_pro package is not available
            try:
                import depth_pro
                logger.info(f"Depth Pro package found, loading local model from {self.model_path}...")

                # Load model from local file
                precision = torch.float16 if self.device == 'mps' else torch.float32
                self.model, self.transform = depth_pro.create_model_and_transforms(
                    device=self.device,
                    precision=precision
                )

                # Load the state dict from the local file
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.eval()
                self.model.to(self.device)
                logger.info(f"✓ Local Depth Pro model loaded successfully on {self.device}")

            except ImportError as e:
                logger.warning("⚠ Depth Pro package not available. Trying to load model directly with PyTorch...")
                try:
                    # Try to load as a standard PyTorch model
                    checkpoint = torch.load(self.model_path, map_location=self.device)
                    logger.info(f"✓ Model checkpoint loaded from {self.model_path}")
                    logger.info("⚠ Using fallback depth estimation since depth_pro package is not available.")
                    self.model = None
                    self.transform = None
                except Exception as torch_error:
                    logger.error(f"⚠ Failed to load model with PyTorch: {torch_error}")
                    logger.info("Using improved fallback depth estimation.")
                    self.model = None
                    self.transform = None

            except Exception as model_error:
                logger.error(f"⚠ Local Depth Pro model loading failed: {model_error}")
                logger.info("Using fallback depth estimation instead.")
                self.model = None
                self.transform = None
        except Exception as e:
            logger.error(f"Error during Depth Pro initialization: {e}")
            self.model = None
            self.transform = None

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map for the input image

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            Depth map as numpy array or None if estimation fails
        """
        if self.model is None or self.transform is None:
            return self._fallback_depth_estimation(image)

        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            # Apply transforms
            image_tensor = self.transform(pil_image)
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            # Estimate depth
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
                depth_map = prediction["depth"].cpu().numpy().squeeze()

            logger.info(f"Depth estimation completed. Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
            return depth_map

        except Exception as e:
            logger.error(f"Error during depth estimation: {e}")
            return self._fallback_depth_estimation(image)

    def _fallback_depth_estimation(self, image: np.ndarray) -> np.ndarray:
        """
        Improved fallback depth estimation using multiple cues

        Args:
            image: Input image as numpy array

        Returns:
            Estimated depth map
        """
        logger.info("Using improved fallback depth estimation")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Brightness-based depth (brighter = closer for food images)
        brightness_depth = 1.0 - (gray.astype(np.float32) / 255.0)

        # 2. Edge-based depth (sharper edges = closer)
        edges = cv2.Canny(gray, 50, 150)
        edge_depth = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0

        # 3. Gradient-based depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_depth = gradient_magnitude / np.max(gradient_magnitude)

        # 4. Position-based depth (center is typically closer for food photos)
        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        position_depth = 1.0 - (distance_from_center / max_distance)

        # 5. Color saturation depth (more saturated = closer for food)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0

        # Combine all depth cues with weights
        depth_map = (
            0.3 * brightness_depth +
            0.2 * edge_depth +
            0.2 * gradient_depth +
            0.15 * position_depth +
            0.15 * saturation
        )

        # Smooth the depth map
        depth_map = cv2.GaussianBlur(depth_map, (7, 7), 0)

        # Scale to reasonable depth range (0.5 to 2.5 meters)
        depth_map = depth_map * 2.0 + 0.5

        return depth_map

    def calculate_volume(self, depth_map: np.ndarray, mask: np.ndarray,
                         pixel_size_mm: float = 0.5, class_name: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate volume of food item using depth map and segmentation mask
        with robust height estimation:
        - Fit a local plane around the object to approximate the base (table/plate)
        - Compute thickness as positive residuals from that plane
        - Use percentile-based height and clamp to plausible bounds

        Args:
            depth_map: Depth map from depth estimation (meters)
            mask: Binary segmentation mask for the food item
            pixel_size_mm: Size of each pixel in millimeters (camera calibration dependent)

        Returns:
            Dictionary containing volume calculations
        """
        try:
            # Ensure mask is binary
            mask_binary = (mask > 0).astype(np.uint8)

            # Erode mask slightly to remove rims/edges that inflate area
            # Tunable via env: CALORIE_AI_ERODE_KERNEL (odd int), CALORIE_AI_ERODE_ITER (int)
            try:
                k = int(os.getenv('CALORIE_AI_ERODE_KERNEL', '7'))
            except Exception:
                k = 7
            # keep kernel reasonable and odd-sized
            k = max(1, min(31, int(k)))
            if k % 2 == 0:
                k += 1
            try:
                iters = int(os.getenv('CALORIE_AI_ERODE_ITER', '2'))
            except Exception:
                iters = 2
            iters = max(1, min(5, int(iters)))

            erode_kernel = np.ones((k, k), np.uint8)
            try:
                mask_binary = cv2.erode(mask_binary, erode_kernel, iterations=iters)
            except Exception:
                pass

            # Resize depth map to match mask dimensions if needed
            if depth_map.shape != mask.shape:
                depth_map = cv2.resize(depth_map, (mask.shape[1], mask.shape[0]))

            # Early exit if mask empty (after erosion)
            ys, xs = np.where(mask_binary > 0)
            if ys.size == 0:
                return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}

            h, w = mask_binary.shape

            # Build a ring region around the object to estimate base plane
            ksize = max(5, int(0.02 * max(h, w)))
            if ksize % 2 == 0:
                ksize += 1
            kernel = np.ones((ksize, ksize), np.uint8)
            dilated = cv2.dilate(mask_binary, kernel, iterations=1)
            ring = ((dilated > 0).astype(np.uint8) - mask_binary).clip(0, 1)

            ry, rx = np.where(ring > 0)
            z_ring = depth_map[ry, rx].astype(np.float64) if ry.size > 0 else np.array([])
            if z_ring.size > 0:
                valid = np.isfinite(z_ring) & (z_ring > 0)
                rx = rx[valid]
                ry = ry[valid]
                z_ring = z_ring[valid]

            have_plane = False
            a = b = 0.0
            c = None

            # Fit plane z = a*x + b*y + c if enough ring points
            if z_ring.size >= 200:
                A = np.stack([rx, ry, np.ones_like(rx)], axis=1).astype(np.float64)
                coeffs, *_ = np.linalg.lstsq(A, z_ring, rcond=None)
                a, b, c = coeffs
                # Robust pass: remove outliers by MAD and refit
                z_pred = A @ coeffs
                resid = z_ring - z_pred
                med = np.median(resid)
                mad = np.median(np.abs(resid - med)) + 1e-6
                keep = np.abs(resid - med) < 3.5 * mad
                if keep.sum() >= 100:
                    A2 = A[keep]
                    z2 = z_ring[keep]
                    coeffs, *_ = np.linalg.lstsq(A2, z2, rcond=None)
                    a, b, c = coeffs
                have_plane = True

            # Depth within the mask
            z_food = depth_map[ys, xs].astype(np.float64)
            z_food = z_food[np.isfinite(z_food) & (z_food > 0)]
            if z_food.size == 0:
                return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}

            # Compute base plane values across the mask
            if have_plane:
                plane_vals = a * xs + b * ys + c
            else:
                # Fallback: constant plane using high-percentile neighborhood depth
                neighborhood = depth_map[cv2.dilate(mask_binary, kernel, iterations=2) > 0]
                neighborhood = neighborhood[np.isfinite(neighborhood) & (neighborhood > 0)]
                if neighborhood.size == 0:
                    neighborhood = z_food
                z0 = float(np.percentile(neighborhood, 95))
                plane_vals = np.full_like(xs, z0, dtype=np.float64)

            # Thickness as positive residuals; meters -> mm
            z_mask_vals = depth_map[ys, xs].astype(np.float64)
            thickness_mm = (plane_vals - z_mask_vals) * 1000.0
            thickness_mm = np.clip(thickness_mm, 0.0, None)
            if thickness_mm.size == 0:
                return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}

            # Robust height estimate with bounds and per-class caps
            avg_height_mm = float(np.percentile(thickness_mm, 80))
            # Global bounds (tightened)
            MIN_H_MM, MAX_H_MM = 3.0, 45.0
            avg_height_mm = float(np.clip(avg_height_mm, MIN_H_MM, MAX_H_MM))

            # Default per-class caps (mm) with config overrides
            default_caps = {
                'Rice': 35.0,
                'Potato mash': 30.0,
                'chicken curry': 28.0,
                'Biriyani': 38.0,
                'Beef curry': 32.0,
                'Egg': 40.0,
                'egg curry': 30.0,
                'Eggplants': 30.0,
                'Fish': 35.0,
                'Khichuri': 34.0,
            }
            # Apply per-class height caps (use defaults since no config system)
            if class_name is not None:
                cap = default_caps.get(class_name)
                if cap is not None:
                    # Clamp cap to safe range
                    cap = float(max(MIN_H_MM, min(MAX_H_MM, float(cap))))
                    avg_height_mm = float(min(avg_height_mm, cap))

            # Area and volume
            area_pixels = int(np.sum(mask_binary))
            area_mm2 = float(area_pixels) * (float(pixel_size_mm) ** 2)
            # Default shape factor (no config overrides for now)
            shape_factor = 0.6
            volume_mm3 = area_mm2 * avg_height_mm * shape_factor
            volume_ml = volume_mm3 / 1000.0

            # Apply reasonable volume limits based on typical food serving sizes
            max_volume_limits = {
                'Rice': 400.0,           # ~2.5 cups cooked rice
                'Khichuri': 350.0,       # ~2 cups porridge
                'Biriyani': 500.0,       # ~3 cups biriyani
                'biriyani': 500.0,       # lowercase alias
                'Beef curry': 300.0,     # ~1.5 cups curry
                'chicken curry': 300.0,  # ~1.5 cups curry
                'Fish': 250.0,           # ~1.5 cups fish curry
                'egg curry': 250.0,      # ~1.5 cups curry
                'Egg': 100.0,            # ~2-3 eggs
                'Eggplants': 200.0,      # ~1 cup cooked eggplant
                'Potato mash': 300.0     # ~1.5 cups mashed potato
            }

            # Apply volume limit if available for this food class
            if class_name and class_name in max_volume_limits:
                max_vol = max_volume_limits[class_name]
                if volume_ml > max_vol:
                    logger.warning(f"Volume {volume_ml:.1f}ml exceeds limit for {class_name} ({max_vol}ml), capping")
                    volume_ml = max_vol

            # Depth stats for telemetry
            min_depth = float(np.min(z_food)) if z_food.size else 0.0
            max_depth = float(np.max(z_food)) if z_food.size else 0.0

            result = {
                'volume_ml': float(volume_ml),
                'area_mm2': float(area_mm2),
                'avg_height_mm': float(avg_height_mm),
                'area_pixels': area_pixels,
                'depth_range': (min_depth, max_depth)
            }
            logger.info(
                f"Volume: {volume_ml:.1f} ml | Area: {area_mm2:.1f} mm² | Height: {avg_height_mm:.1f} mm | Pixels: {area_pixels} | Class: {class_name or 'Unknown'}"
            )
            return result

        except Exception as e:
            logger.error(f"Error calculating volume: {e}")
            return {'volume_ml': 0.0, 'area_mm2': 0.0, 'avg_height_mm': 0.0}

    def process_food_regions(self, image: np.ndarray, food_detections: List[Dict],
                           pixel_size_mm: float = 0.5) -> List[Dict]:
        """
        Process multiple food regions to estimate volumes

        Args:
            image: Original image
            food_detections: List of food detections with masks
            pixel_size_mm: Pixel size calibration

        Returns:
            List of food detections with volume information added
        """
        # Estimate depth for the entire image
        depth_map = self.estimate_depth(image)

        if depth_map is None:
            logger.error("Failed to estimate depth")
            return food_detections

        # Process each food detection
        enhanced_detections = []
        for detection in food_detections:
            enhanced_detection = detection.copy()

            # Calculate volume for this food item
            volume_info = self.calculate_volume(
                depth_map,
                detection['mask'],
                pixel_size_mm,
                class_name=detection.get('class_name')
            )

            # Add volume information to detection
            enhanced_detection.update(volume_info)
            enhanced_detections.append(enhanced_detection)

        return enhanced_detections

def test_depth_estimator():
    """Test function for the depth estimator"""
    estimator = DepthEstimator()

    # Create a dummy image for testing
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create a dummy mask
    test_mask = np.zeros((480, 640), dtype=np.uint8)
    test_mask[200:300, 250:400] = 1  # Rectangular region

    # Estimate depth
    depth_map = estimator.estimate_depth(test_image)

    if depth_map is not None:
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")

        # Calculate volume
        volume_info = estimator.calculate_volume(depth_map, test_mask)
        print(f"Volume estimation: {volume_info}")
    else:
        print("Depth estimation failed")

if __name__ == "__main__":
    test_depth_estimator()