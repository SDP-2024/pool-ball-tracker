import numpy as np
import cv2 as cv
import cv2 as cv
import numpy as np
from skimage import exposure
from skimage.exposure import match_histograms

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



def warp_and_stitch(frame_1, frame_2, homography_1, homography_2, output_size):
    """
    Warps two frames using their respective homographies and stitches them together.

    Args:
        frame_1 (np.ndarray): The image/frame from camera 1 to be warped.
        frame_2 (np.ndarray): The image/frame from camera 2 to be warped.
        homography_1 (np.ndarray): The 3x3 homography matrix for frame 1.
        homography_2 (np.ndarray): The 3x3 homography matrix for frame 2.
        output_size (tuple): The size of the output stitched image as (width, height).

    Returns:
        np.ndarray: The stitched image created by blending the two warped frames.
    """
    # Warp both frames
    warped_1 = cv.warpPerspective(frame_1, homography_1, output_size)
    warped_2 = cv.warpPerspective(frame_2, homography_2, output_size)

    # Blend the two images
    stitched = cv.addWeighted(warped_1, 0.5, warped_2, 0.5, 0)

    return stitched

def equalize_frame(frame):
    # Convert from BGR to YCrCb color space
    ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    
    # Split the channels
    y, cr, cb = cv.split(ycrcb)
    
    # Equalize the histogram of the Y channel (luminance)
    y_eq = cv.equalizeHist(y)
    
    # Merge the channels back
    ycrcb_eq = cv.merge((y_eq, cr, cb))
    
    # Convert back to BGR color space
    frame_eq = cv.cvtColor(ycrcb_eq, cv.COLOR_YCrCb2BGR)
    
    return frame_eq


def match_colors(ref_frame, to_match):
    ref_frame = cv.cvtColor(ref_frame, cv.COLOR_BGR2RGB)
    to_match = cv.cvtColor(to_match,cv.COLOR_BGR2RGB)
    
    matched_frame = match_histograms(to_match, ref_frame, channel_axis = -1)
    matched_frame = cv.cvtColor((matched_frame * 255).astype(np.uint8), cv.COLOR_RGB2BGR)

    return matched_frame