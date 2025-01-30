import cv2 as cv
import argparse
from config_manager import load_config, create_profile
from detection.ball_detector import BallDetector
from detection.table_detector import TableDetector
from processing.frame_processing import crop_to_middle
from processing.camera_adjustments import *
from imutils.video import VideoStream
from processing.stitcher import Stitcher
import time
import logging
from processing.camera_calibration import *
from processing.stitcher import Stitcher

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)



def main():
    """
    Main function for initializing and running the ball and table detection system.
    """
    
    # Load config
    profile = parse_args()
    config = load_config(profile)

    if config is None:
        logger.error("Error getting config file.")
        return

    ball_detector = BallDetector(config)
    table_detector = TableDetector(config)

    # Calibrate cameras
    mtx_left, dst_left, mtx_right, dst_right = handle_calibration(config)

    stitcher = Stitcher(config)

    left_camera, right_camera = None, None

    # Attempt to load cameras
    try:
        logger.info("Starting cameras...")
        left_camera = VideoStream(config["camera_port_1"]).start()
        logger.info("Left camera started.")

        # Check if second camera enabled
        if config["camera_port_2"] != -1:
                right_camera = VideoStream(config["camera_port_2"]).start()
                logger.info("Right camera started.")
        else:
            logger.warning("Right camera disabled. Continuing with left camera only.")
            right_camera = None
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return

    # Allow cameras to warm up
    time.sleep(2.0)

    # Create a named window with the WINDOW_NORMAL flag to allow resizing
    cv.namedWindow("Stitched Image (Cropped)", cv.WINDOW_NORMAL)


    while True:
        # Read frames
        left_frame = left_camera.read()
        right_frame = right_camera.read() if right_camera else None

        # Fix any distortion in the cameras
        left_frame, right_frame = undistort_cameras(config, left_frame, right_frame, mtx_left, dst_left, mtx_right, dst_right)

        # Check if frames are valid
        if left_frame is None:
            logger.error("Left camera frame is invalid.")
            continue 
        

        # Handle frame stitching if required
        if right_frame is None:
            stitched_frame = left_frame  # Fallback to left frame
        else:
            stitched_frame = stitcher.stitch_frames(left_frame, right_frame)

            # Crop the stitched frame to the middle area
            if stitched_frame is not None:
                stitched_frame = crop_to_middle(stitched_frame, crop_fraction=0.1)

        # Detect and draw balls to frame
        detected_balls = ball_detector.detect(stitched_frame)
        ball_detector.draw_detected_balls(stitched_frame, detected_balls)

        # Detect table and draw to frame
        table = table_detector.detect(stitched_frame)
        table_detector.draw_edges(stitched_frame, table)


        # Display frames
        cv.imshow("Left camera", left_frame)
        if right_frame is not None:
            cv.imshow("Right camera", right_frame)
        cv.imshow("Stitched Image (Cropped)", stitched_frame)

        # Exit if 'q' pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    left_camera.stop()
    if right_camera is not None:
        right_camera.stop()
    cv.destroyAllWindows()



def parse_args():
    """
    Parses command-line arguments to select a configuration profile.

    Returns:
        str: The name of the selected profile (default is 'default').
    """
    parser = argparse.ArgumentParser(description="Select a config profile to use.")
    parser.add_argument(
        "--profile",
        type=str,
        default='default',
        help="The name of the profile to use (default: `default`)"
    )

    parser.add_argument(
        "--create-profile",
        type=str,
        help="Create a new profile with the specified name."
    )
    args = parser.parse_args()

    if args.create_profile:
        create_profile(name=args.create_profile)
        return args.create_profile

    return args.profile

if __name__ == "__main__":
    main()
