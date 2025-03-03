import cv2
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

def get_top_down_view(frame_1, frame_2, table_pts_cam1, table_pts_cam2):
    # Compute homography matrices
    output_size = (800, 400)
    table_rect = np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]])

    H1, _ = cv2.findHomography(table_pts_cam1, table_rect)
    H2, _ = cv2.findHomography(table_pts_cam2, table_rect)

    top_down_view1 = cv2.warpPerspective(frame_1, H1, output_size)
    top_down_view2 = cv2.warpPerspective(frame_2, H2, output_size)

    # Stack frames vertically
    top_down_view2 = cv2.rotate(top_down_view2, cv2.ROTATE_180)
    stitched_frame = np.vstack((top_down_view1, top_down_view2))

    return stitched_frame

def augment_image(image):
    # Random brightness adjustment
    brightness_factor = random.uniform(0.7, 1.3)
    image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

    # Add random Gaussian noise
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = cv2.add(image.astype(np.int16), noise)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Apply slight blur randomly
    if random.random() > 0.5:
        ksize = random.choice([3, 5])  # Random kernel size
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    return image