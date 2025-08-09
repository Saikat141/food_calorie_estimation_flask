#!/usr/bin/env python3
"""
Quick test to verify volume calculation is working
"""

import os
import numpy as np
import cv2
import logging

# Set environment variables
os.environ['CALORIE_AI_PIXEL_SIZE_MM'] = '0.18'
os.environ['CALORIE_AI_ERODE_KERNEL'] = '11'
os.environ['CALORIE_AI_ERODE_ITER'] = '3'

from depth_estimation import DepthEstimator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_volume_calculation():
    """Test volume calculation with a simple case"""
    logger.info("Testing volume calculation...")
    
    try:
        # Initialize depth estimator
        estimator = DepthEstimator()
        
        # Create a simple test image
        test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

        # Create a larger circular mask to survive erosion
        mask = np.zeros((200, 200), dtype=np.uint8)
        center = (100, 100)
        radius = 50  # Larger radius
        cv2.circle(mask, center, radius, 1, -1)
        
        # Test volume calculation
        volume_info = estimator.calculate_volume(
            depth_map=np.ones((200, 200)) * 1.5,  # Simple depth map
            mask=mask,
            pixel_size_mm=0.18,
            class_name="Rice"
        )
        
        logger.info(f"Volume calculation result: {volume_info}")
        
        if volume_info['volume_ml'] > 0:
            logger.info("✅ Volume calculation is working!")
            return True
        else:
            logger.error("❌ Volume calculation returned 0")
            return False
            
    except Exception as e:
        logger.error(f"❌ Volume calculation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_volume_calculation()
    if success:
        print("✅ Volume calculation test passed!")
    else:
        print("❌ Volume calculation test failed!")
