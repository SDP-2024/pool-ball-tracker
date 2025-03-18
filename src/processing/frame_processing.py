import cv2
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)

def get_top_down_view(frame, table_pts):
    output_height = 600
    output_width = 1200

    table_rect = np.float32([
        [0, 0], 
        [output_width, 0], 
        [0, output_height], 
        [output_width, output_height]
    ])

    # Compute homography matrices
    M = cv2.getPerspectiveTransform(table_pts,table_rect)

    # Warp the frames
    top_down_view = cv2.warpPerspective(frame, M, (output_width, output_height), borderMode=cv2.BORDER_REPLICATE)

    return top_down_view


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