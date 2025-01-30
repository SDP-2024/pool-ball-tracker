import cv2 as cv
import numpy as np
from processing.frame_processing import frame_to_gs


class TableDetector:
    """
    A class for detecting and tracking the corners of a table in a video frame.

    This class uses edge detection and contour approximation to identify the table in
    each frame. It smooths the detected corners over time to reduce jitter and applies
    a threshold for detecting significant changes in the detected corners between frames.
    """
    def __init__(self, config):
        """
        Initializes the TableDetector with configuration settings.

        Args:
            config (dict): The configuration dictionary containing detector settings.
                           The dictionary must include a "detector" key with relevant parameters.
        """
        self.config = config.get("detector", {})
        self.table_area = self.config["table_area"]
        self.previous_corners = None
        self.alpha = 0.5  # Smoothing weight (closer to 1 = more stable, less responsive)

    def smooth_corners(self, new_corners):
        """
        Applies exponential smoothing to the detected corners to reduce jitter.

        Args:
            new_corners (np.ndarray): The newly detected corners of the table.

        Returns:
            np.ndarray: The smoothed corners after applying the smoothing filter.
        """
        if self.previous_corners is None:
            self.previous_corners = new_corners
            return new_corners
        
        # Apply exponential smoothing
        smoothed_corners = self.alpha * self.previous_corners + (1 - self.alpha) * new_corners
        self.previous_corners = smoothed_corners
        return smoothed_corners

    def detect(self, frame):
        """
        Detects the edges of the table in the given frame.

        This function converts the frame to grayscale, applies edge detection, finds
        contours, and approximates the table's corners if the table is large enough.

        Args:
            frame (np.ndarray): The input image/frame containing the table.

        Returns:
            np.ndarray or None: The smoothed table corners if detected, otherwise None.
        """

        processed_frame = frame_to_gs(frame)
        # Detect edges
        edges = cv.Canny(processed_frame, 50, 150)
        
        # Find contours in detected edges
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Largest contour is most likely the table, so take that
            table_contour = max(contours, key=cv.contourArea)
            
            # If max contour is greater than threshold
            if cv.contourArea(table_contour) > self.table_area:
                epsilon = 0.02 * cv.arcLength(table_contour, True)
                approx = cv.approxPolyDP(table_contour, epsilon, True)

                # If number of corners are 4
                if len(approx) == 4:
                    corners = np.squeeze(approx)

                    # Check for significant change from previous frame
                    if self.is_significant_change(self.previous_corners, corners, threshold=15):
                        corners = self.smooth_corners(corners)
                        self.previous_corners = corners
                    
                    return self.previous_corners

        return None
    

    def is_significant_change(self, previous_corners, new_corners, threshold=10):
        """
        Checks if there is a significant change between the previous and current corners.

        Args:
            previous_corners (np.ndarray): The corners detected in the previous frame.
            new_corners (np.ndarray): The corners detected in the current frame.
            threshold (int, optional): The threshold distance to consider a significant change.
                                    Defaults to 10.

        Returns:
            bool: True if the change between the corners is significant, otherwise False.
        """
        if previous_corners is None:
            return True  # Accept the first detection
        
        # Calculate distances between corresponding corners
        distances = np.linalg.norm(previous_corners - new_corners, axis=1)
        return np.any(distances > threshold)

    

    def draw_edges(self, frame, table):
        """
        Draws the detected table by marking the corners and connecting them with lines.

        Args:
            frame (np.ndarray): The input frame where the table is detected.
            corners (np.ndarray): The detected corner points of the table.

        Returns:
            None
        """
        if table is not None and len(table) == 4:
            # Draw corner points
            for corner in table:
                cv.circle(frame, tuple(map(int, corner)), 5, (0, 0, 255), -1)  # Red circles
            
            # Connect corners with lines
            for i in range(4):
                start_point = tuple(map(int, table[i]))
                end_point = tuple(map(int, table[(i + 1) % 4]))  # Wrap around to the first point
                cv.line(frame, start_point, end_point, (0, 255, 0), 2)  # Green lines