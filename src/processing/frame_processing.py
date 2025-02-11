import cv2 as cv
import logging
import numpy as np

logger = logging.getLogger(__name__)

def frame_to_gs(frame):
    """"
    Converts a frame to grayscale and applies Gaussian blur.

    Args:
        frame (np.ndarray): The input image/frame to be processed.

    Returns:
        np.ndarray: The processed image, which is a blurred grayscale version of the input frame.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5,5), 0)

    return blur

def crop_to_middle(frame, crop_fraction=0.8):
    """
    Crop the middle region of the frame.

    Args:
        frame (numpy.ndarray): The input frame to crop.
        crop_fraction (float): The fraction of the frame to crop from each side (default: 0.2).

    Returns:
        numpy.ndarray: The cropped frame.
    """
    height, width = frame.shape[:2]
    crop_x = int(width * crop_fraction)
    crop_y = int(height * crop_fraction)

    # Define the ROI: middle section of the frame
    start_x, end_x = crop_x, width - crop_x
    start_y, end_y = crop_y, height - crop_y

    # Return the cropped frame
    return frame[start_y:end_y, start_x:end_x]


def get_top_down_view(frame_1, frame_2, table_pts_cam1, table_pts_cam2):
    # Compute homography matrices
    output_size = (800, 400)
    table_rect = np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]])

    H1 = cv.getPerspectiveTransform(table_pts_cam1, table_rect)
    H2 = cv.getPerspectiveTransform(table_pts_cam2, table_rect)

    top_down_view1 = cv.warpPerspective(frame_1, H1, output_size)
    top_down_view2 = cv.warpPerspective(frame_2, H2, output_size)

    # Stack frames vertically
    stitched_frame = np.vstack((top_down_view1, top_down_view2))

    return stitched_frame