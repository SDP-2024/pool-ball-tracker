import cv2
import argparse
import time
import logging
import threading
from imutils.video import VideoStream
from random import randint

from config.config_manager import load_config, create_profile
from src.processing.camera_calibration import *
from src.processing.frame_processing import *
from src.detection.detection_model import DetectionModel
from src.detection.autoencoder import AutoEncoder
from src.networking.network import Network
from src.logic.game_state import StateManager

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

    args = parse_args()

    # Load config
    profile_name = args.create_profile if args.create_profile else args.profile
    if args.create_profile:
        create_profile(name=profile_name)
    config = load_config(profile_name)
    
    if config is None:
        logger.error("Error getting config file.")
        return

    if args.collect_ae_data:
        reset_ae_data(config)
    if args.set_points:
        reset_points()
    
    # Calibrate cameras
    mtx, dst, new_mtx, roi = handle_calibration(config)
    camera = load_cameras(config)

    # Allow cameras to warm up
    time.sleep(2.0)
        
    # Read frames
    frame = camera.read()

    # Set up coordinate system for the cropped frames
    table_pts = manage_point_selection(config, camera, mtx, dst)
    stitched_frame = get_top_down_view(frame,table_pts)
    logger.info(stitched_frame.shape)

    # If networking is enabled, start the server
    network = None
    if config.get("use_networking", False):
        network = Network(config)
        network.start()
        state_manager = StateManager(config, network)
    else:
        state_manager = StateManager(config)

    detection_model = DetectionModel(config)

    autoencoder = None
    if not args.collect_ae_data:
        autoencoder = AutoEncoder(config)
    if detection_model.model is None: # Check if model loaded successfully
        return
    
    
    # Process the frames
    while True:
        # Read frames
        frame = camera.read()

        # Fix any distortion in the cameras
        frame = undistort_cameras(config, frame, mtx, dst, new_mtx, roi)

        if frame is None:
            logger.error("Camera frame is invalid.")
            continue 

        # Get top-down view of the table
        frame = get_top_down_view(frame, table_pts)

        if args.collect_ae_data: # Collect data for autoencoder
            capture_frame(config, stitched_frame)
        if args.collect_model_images:
            capture_frame_for_training(config, stitched_frame)

        drawing_frame = stitched_frame.copy()

        # Detect and draw balls to frame
        detected_balls, labels = detection_model.detect(stitched_frame)
        detection_model.draw(drawing_frame, detected_balls)

        state_manager.update(detected_balls, labels)
        
        # Detect anomalies in the frame if required
        if not args.collect_ae_data and not args.no_anomaly:
            table_only = detection_model.extract_bounding_boxes(stitched_frame, detected_balls)
            is_anomaly = autoencoder.detect_anomaly(table_only)
            if network:
                network.send_obstruction(is_anomaly)

        # Exit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    if config["use_networking"]:
        network.disconnect()


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

    parser.add_argument(
        "--collect-ae-data",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--set-points",
        action='store_true',
    )

    parser.add_argument(
        "--no-anomaly",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "--collect-model-images",
        action="store_true",
        default=False
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=False
    )

    return parser.parse_args()

def load_cameras(config):
    # Attempt to load cameras
    try:
        logger.info("Starting cameras...")
        camera = VideoStream(config["camera_port_1"]).start()
        logger.info("Camera 1 started.")
        return camera
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return
    
def capture_frame_for_training(config, frame):
    path=f"./{config['model_training_path']}"
    if not os.path.exists(path):
        os.makedirs(path)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        num = randint(0, 10000)
        filename = f"{path}train_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")

def capture_frame(config, frame):
    path = f"./{config['clean_images_path']}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        num = randint(0, 10000)
        filename = f"{path}clean_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")

def reset_ae_data(config):
    path = f"./{config['clean_images_path']}/"
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(path + file)
        logger.info("Data reset.")

def reset_points():
    path = "./config/table_points.json"
    if os.path.exists(path):
        os.remove(path)
        logger.info("Points reset.")

if __name__ == "__main__":
    main()
