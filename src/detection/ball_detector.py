import cv2 as cv

from detection.color_detection import ColorDetector
from detection.shape_detection import ShapeDetector
from detection.ball_classification import Classifier


class BallDetector:
    """
    A class for detecting and classifying balls based on color and shape in an image.

    The BallDetector class uses color detection, shape detection, and classification techniques
    to detect balls in an image. It combines these methods to identify balls based on their
    color, shape, and size, and returns their position, radius, and area.
    """
    def __init__(self, config):
        """
        Initializes the BallDetector with configuration settings.

        Args:
            config (dict): The configuration dictionary containing settings for color detection, 
                           shape detection, and ball classification.
        """
        self.color_detector = ColorDetector(config)
        self.shape_detector = ShapeDetector(config)
        self.ball_classifier = Classifier(config)

    def detect(self, frame):
        """
        Detects balls in the given frame using color detection, shape detection, and classification.

        This method first detects the color of the balls, then identifies their shapes, and finally
        classifies the detected shapes as balls. It returns a list of detected balls, including their
        color, position, radius, and area.

        Args:
            frame (np.ndarray): The input image/frame to be processed.

        Returns:
            list: A list of detected balls, where each ball is represented as a dictionary containing:
                  - "color": The detected color of the ball.
                  - "position": The (x, y) coordinates of the ball's center.
                  - "radius": The radius of the detected ball.
                  - "area": The area of the detected ball.
        """

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        detected_balls = []

        for color in self.color_detector.hsv_thresholds.keys():
            # Step 1: Detect color
            mask = self.color_detector.detect(hsv, color)

            # Step 2: Detect shapes
            balls = self.shape_detector.detect(mask)

            # Step 3: Classify balls
            #balls = self.ball_classifier.classify(shapes)

            # Step 4: Combine results
            for (x, y, radius, area) in balls:
                detected_balls.append({
                    "color": color,
                    "position": (x, y),
                    "radius": radius,
                    "area" : area,
                })

            # Display the mask for each color in frame
            cv.imshow(f"Color {color}", mask)

        return detected_balls
    

    def draw_detected_balls(self, stitched_frame, detected_balls):
        """
        Draws detected balls on the stitched frame.

        Args:
            stitched_frame (numpy.ndarray): The stitched image frame.
            detected_balls (list of dict): List of detected balls, where each ball is a dictionary containing:
                - "position" (tuple): The (x, y) coordinates of the ball.
                - "radius" (int): The radius of the ball.
                - "color" (str): The detected color of the ball.

        Returns:
            None
        """
        
        for ball in detected_balls:
            x, y = ball["position"]
            radius = ball["radius"]
            color_name = ball["color"]

            # Define text position
            text_position = (x - 20, y - radius - 10)

             # Draw a circle around the detected ball
            cv.circle(stitched_frame, (x, y), radius, (0, 255, 255), 2)

            # Put text with detected color above the ball
            cv.putText(stitched_frame, color_name, text_position, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)