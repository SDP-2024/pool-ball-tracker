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
import numpy as np

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        #logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)


# Main function
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
    mtx_1, dst_1, mtx_2, dst_2 = handle_calibration(config)

    stitcher = Stitcher(config)

    camera_1, camera_2 = None, None

    # Attempt to load cameras
    try:
        logger.info("Starting cameras...")
        camera_1 = VideoStream(config["camera_port_1"]).start()
        logger.info("Camera 1 started.")

        # Check if second camera enabled
        if config["camera_port_2"] != -1:
            camera_2 = VideoStream(config["camera_port_2"]).start()
            logger.info("Camera 2 started.")
        else:
            logger.warning("Camera 2 disabled. Continuing with camera 1 only.")
            camera_2 = None
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return

    # Allow cameras to warm up
    time.sleep(2.0)

    # Create a named window with the WINDOW_NORMAL flag to allow resizing
    cv.namedWindow("Stitched Image (Cropped)", cv.WINDOW_NORMAL)

    if camera_2 is not None:
        table_pts_cam1, table_pts_cam2 = manage_point_selection(config, camera_1, camera_2, mtx_1, dst_1, mtx_2, dst_2)
    
    # Process the frames and perform stitching
    while True:
        # Read frames
        frame_1 = camera_1.read()
        frame_2 = camera_2.read() if camera_2 else None

        # Fix any distortion in the cameras
        frame_1, frame_2 = undistort_cameras(config, frame_1, frame_2, mtx_1, dst_1, mtx_2, dst_2)

        if frame_1 is None:
            logger.error("Camera 1 frame is invalid.")
            continue 

        # Handle frame stitching if required
        if frame_2 is None:
            stitched_frame = frame_1  # Fallback to frame 1
        else:
            # Compute homography matrices
            output_size = (800, 400)
            table_rect = np.float32([[0, 0], [output_size[0], 0], [0, output_size[1]], [output_size[0], output_size[1]]])

            H1 = cv.getPerspectiveTransform(table_pts_cam1, table_rect)
            H2 = cv.getPerspectiveTransform(table_pts_cam2, table_rect)

            top_down_view1 = cv.warpPerspective(frame_1, H1, output_size)
            top_down_view2 = cv.warpPerspective(frame_2, H2, output_size)

            top_down_view1 = cv.rotate(top_down_view1, cv.ROTATE_180)

            # Stack frames vertically
            stitched_frame = np.vstack((top_down_view1, top_down_view2))

        # Detect and draw balls to frame
        detected_balls = ball_detector.detect(stitched_frame)
        ball_detector.draw_detected_balls(stitched_frame, detected_balls)

        # Detect table and draw to frame
        table = table_detector.detect(stitched_frame)
        table_detector.draw_edges(stitched_frame, table)

        # Display frames
        cv.imshow("Camera 1", frame_1)
        if frame_2 is not None:
            cv.imshow("Camera 2", frame_2)
        cv.imshow("Stitched Image (Cropped)", stitched_frame)

        # Exit if 'q' pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera_1.stop()
    if camera_2 is not None:
        camera_2.stop()
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
