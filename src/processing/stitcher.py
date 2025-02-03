import cv2 as cv
import time
import imutils
import logging
import numpy as np

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

    def stitch_frames(self, frame_1, frame_2):
        """
        Stitches two frames together to create a panoramic image.

        Args:
            frame_1 (numpy.ndarray): The frame from camera 1.
            frame_2 (numpy.ndarray): The frame from camera 2.

        Returns:
            numpy.ndarray: The stitched output image. If stitching fails, 
                           returns the last successfully stitched frame 
                           or the camera 1 frame as a fallback.
        """

        # Update the stitched frame at a fixed interval or based on other criteria
        if time.time() - self.last_update_time > self.config["time_between_stitch"]: 
            self.last_update_time = time.time()
            self.last_stitched_frame = None  # Force recalculation of the stitched frame
            logger.info("Forcing new stitching due to time interval.")

        # Resize frames for faster processing
        frame_1 = imutils.resize(frame_1, width=640, height=480)
        frame_2 = imutils.resize(frame_2, width=640, height=480)

        try:
            # If no stitched frame exists, calculate one
            if self.last_stitched_frame is None:
                status, stitched_frame = self.stitcher.stitch([frame_1, frame_2])
                if status == cv.Stitcher_OK:
                    self.last_stitched_frame = stitched_frame  # Cache the stitched frame
                    logger.info("Stitching successful!")
                else:
                    logger.error(f"Stitching failed with error code {status}")
                    stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else frame_1  # Fallback
            else:
                # Reuse the last stitched frame without recalculating it
                stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else frame_1

        except cv.error as e:
            logger.error(f"OpenCV stitching error: {e}")
            stitched_frame = self.last_stitched_frame if self.last_stitched_frame is not None else frame_1  # Fallback

        return stitched_frame
    

    def compute_homography(self, frame_1, frame_2):
        gray_1 = cv.cvtColor(frame_1, cv.COLOR_BGR2GRAY)
        gray_2 = cv.cvtColor(frame_2, cv.COLOR_BGR2GRAY)


        orb = cv.ORB.create()
        kp1, des1 = orb.detectAndCompute(gray_1, None)
        kp2, des2 = orb.detectAndCompute(gray_2, None)


        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched keypoints
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography matrix
        H, _ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)

        # Warp second camera's view using homography
        height, width, _ = frame_1.shape
        warped_img2 = cv.warpPerspective(frame_2, H, (width * 2, height))

        # Merge the two images into one stitched frame
        stitched_frame = warped_img2.copy()
        stitched_frame[:, :width] = frame_1  # Place first camera's view in left part

        # Save transformation for later use
        np.save("homography_matrix.npy", H)


    def warp_frame(self, H, frame_1, frame_2):
        # Warp second camera's frame
        warped_frame2 = cv.warpPerspective(frame_2, H, (frame_1.shape[1] * 2, frame_1.shape[0]))
        stitched_frame = warped_frame2.copy()
        stitched_frame[:, :frame_1.shape[1]] = frame_1  # Merge views

        # Convert to HSV for color-based tracking
        hsv = cv.cvtColor(stitched_frame, cv.COLOR_BGR2HSV)

        return hsv
