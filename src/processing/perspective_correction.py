import numpy as np
import cv2 as cv

import numpy as np
import cv2 as cv

def correct_perspective(frame, corners):
    """
    Corrects the perspective of an image based on the detected corners.

    This function applies a perspective transformation to align the detected corners of 
    an object (e.g., a table) in the frame to a corrected, centered perspective while 
    maintaining the object's aspect ratio.

    Args:
        frame (np.ndarray): The input image/frame to be processed.
        corners (list or np.ndarray): The four detected corners of the object in the frame 
                                      (e.g., the corners of a table).

    Returns:
        np.ndarray: The corrected image with the perspective adjusted.
    """
    # Get the frame dimensions (height and width)
    frame_h, frame_w = frame.shape[:2]

    # Ensure corners are in the correct format
    corners = np.array(corners, dtype=np.float32)

    # Calculate the aspect ratio of the detected table
    table_w = np.linalg.norm(corners[1] - corners[0])  # Width of the table
    table_h = np.linalg.norm(corners[3] - corners[0])  # Height of the table
    table_aspect_ratio = table_w / table_h

    # Calculate the dimensions of the table while keeping the aspect ratio
    if frame_w / frame_h > table_aspect_ratio:
        # Frame is wider than the table; height is the limiting factor
        output_h = frame_h
        output_w = int(output_h * table_aspect_ratio)
    else:
        # Frame is taller than the table; width is the limiting factor
        output_w = frame_w
        output_h = int(output_w / table_aspect_ratio)

    # Define corrected corners to center the table in the frame
    x_offset = (frame_w - output_w) / 2
    y_offset = (frame_h - output_h) / 2

    corners_corrected = np.array([
        [x_offset, y_offset],  # Top-left
        [x_offset + output_w - 1, y_offset],  # Top-right
        [x_offset + output_w - 1, y_offset + output_h - 1],  # Bottom-right
        [x_offset, y_offset + output_h - 1]  # Bottom-left
    ], dtype=np.float32)

    # Get the perspective transformation matrix
    perspective_transform = cv.getPerspectiveTransform(corners, corners_corrected)

    # Apply the perspective warp with the output dimensions being the full frame
    corrected = cv.warpPerspective(frame, perspective_transform, (frame_w, frame_h))

    return corrected


