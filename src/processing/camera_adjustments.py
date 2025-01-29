import numpy as np
import cv2 as cv
import cv2 as cv
import numpy as np


def compute_homography(corners, output_size):
    """
    Computes the homography matrix to transform a set of corners into a top-down view.

    Args:
        corners (np.ndarray): An array of 4 points representing the corners in the original image.
                              These points should be in the order: top-left, top-right, bottom-right, bottom-left.
        output_size (tuple): The desired output size as (width, height).

    Returns:
        np.ndarray: The 3x3 homography matrix that maps the input corners to the specified top-down view.
    """
    # Define the desired top-down view corners
    width, height = output_size
    table_corners = np.array([
        [0, 0], # Top-left
        [width - 1, 0], # Top-right
        [width - 1, height - 1], # Bottom-right
        [0, height - 1] # Bottom-left
    ], dtype=np.float32)

    # Compute the homography
    homography_matrix, _ = cv.findHomography(corners, table_corners)
    return homography_matrix



def warp_and_stitch(left_frame, right_frame, left_homography, right_homography, output_size):
    """
    Warps two frames using their respective homographies and stitches them together.

    Args:
        left_frame (np.ndarray): The left image/frame to be warped.
        right_frame (np.ndarray): The right image/frame to be warped.
        left_homography (np.ndarray): The 3x3 homography matrix for the left frame.
        right_homography (np.ndarray): The 3x3 homography matrix for the right frame.
        output_size (tuple): The size of the output stitched image as (width, height).

    Returns:
        np.ndarray: The stitched image created by blending the two warped frames.
    """
    # Warp both frames
    left_warped = cv.warpPerspective(left_frame, left_homography, output_size)
    right_warped = cv.warpPerspective(right_frame, right_homography, output_size)

    # Blend the two images
    stitched = cv.addWeighted(left_warped, 0.5, right_warped, 0.5, 0)

    return stitched
