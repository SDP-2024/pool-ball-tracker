import cv2 as cv
import time
import imutils
import logging

logger = logging.getLogger(__name__)

class Stitcher:
    """
    A class to perform image stitching on frames from two sources.
    """
    def __init__(self, config):
        """
        Initializes the Stitcher class.

        Args:
            config (dict): Configuration file
        """
        self.stitcher = cv.Stitcher().create()
        self.last_stitched_frame = None
        self.last_update_time = time.time()
        self.config = config

    def stitch_frames(self, left_frame, right_frame):
        """
        Stitches two frames together to create a panoramic image.

        Args:
            left_frame (numpy.ndarray): The left image frame.
            right_frame (numpy.ndarray): The right image frame.

        Returns:
            numpy.ndarray: The stitched output image. If stitching fails, 
                           returns the last successfully stitched frame 
                           or the left frame as a fallback.
        """

        # Update the stitched frame at a fixed interval or based on other criteria
        if time.time() - self.last_update_time > self.config["time_between_stitch"]: 
            self.last_update_time = time.time()
            self.last_stitched_frame = None  # Force recalculation of the stitched frame
            logger.info("Forcing new stitching due to time interval.")

        # Resize frames for faster processing
        left_frame = imutils.resize(left_frame, width=640, height=480)
        right_frame = imutils.resize(right_frame, width=640, height=480)

        try:
            # If no stitched frame exists, calculate one
            if self.last_stitched_frame is None:
                status, stitched_frame = self.stitcher.stitch([left_frame, right_frame])
                if status == cv.Stitcher_OK:
                    self.last_stitched_frame = stitched_frame  # Cache the stitched frame
                    logger.info("Stitching successful!")
                else:
                    logger.error(f"Stitching failed with error code {status}")
                    stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else left_frame  # Fallback
            else:
                # Reuse the last stitched frame without recalculating it
                stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else left_frame

        except cv.error as e:
            logger.error(f"OpenCV stitching error: {e}")
            stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else left_frame  # Fallback

        return stitched_frame
    
