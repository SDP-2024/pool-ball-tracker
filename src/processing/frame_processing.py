import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

def get_top_down_view(frame_1, table_pts_cam1):
    """
    Get the top down view of the table given the selected points on the table

    Args:
        frame_1 (numpy.ndarray): Input frame to get top down view
        table_pts_cam1 (numpy.array): Array of selected points 

    Returns:
        numpy.ndarray: Output frame warped to top down perspective 
    """
    output_height = 1040  
    output_width = 460

    table_rect = np.float32([
        [0, 0], 
        [output_width, 0], 
        [0, output_height], 
        [output_width, output_height]
    ])

    # Compute homography matrices
    H1, _ = cv2.findHomography(table_pts_cam1, table_rect)

    # Warp the frames
    top_down_view1 = cv2.warpPerspective(frame_1, H1, (output_width, output_height))

    return top_down_view1