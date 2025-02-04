import cv2 as cv
import numpy as np

class ShapeDetector:
    """
    A class for detecting shapes in a binary mask, specifically circles based on contour analysis.

    This class identifies contours in a binary mask, calculates their area and perimeter, 
    and checks if they correspond to circular shapes. Detected shapes are returned with 
    their center coordinates, radius, and area.
    """
    def __init__(self, config):
        """
        Initializes the ShapeDetector with configuration settings.

        Args:
            config (dict): The configuration dictionary containing detector settings, 
                           including parameters like `min_area` for shape detection.
        """
        self.config = config

    def detect(self, mask):
        """
        Detects shapes in a given binary mask using contour analysis.

        This method finds contours in the mask, calculates their area and perimeter, 
        and checks if they correspond to circular shapes based on the circularity metric.

        Args:
            mask (np.ndarray): A binary mask (image) where the shapes to be detected are white 
                               (255) on a black (0) background.

        Returns:
            list of tuples: A list of detected shapes, each represented by a tuple:
                             (x, y, radius, area), where:
                             - (x, y) is the center of the circle,
                             - radius is the radius of the circle,
                             - area is the area of the detected shape.
        """

        # Find contours in the current mask
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        detected_shapes = []

        # Check each contour
        for contour in contours:
            area = cv.contourArea(contour)

            # If contour is greater than threshold area
            if area > self.config["min_area"]:
                perimeter = cv.arcLength(contour, True)

                # Check if circle shape
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter ** 2))

                    if 0.7 < circularity <= 1.2:  # Threshold for circular shapes
                        (x, y), radius = cv.minEnclosingCircle(contour)
                        detected_shapes.append((int(x), int(y), int(radius), float(area)))

        return detected_shapes