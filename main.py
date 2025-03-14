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
from src.networking.video_feed import start_stream
from src.database.db_controller import DBController
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
    mtx_1, dst_1, mtx_2, dst_2 = handle_calibration(config)
    camera_1, camera_2 = load_cameras(config)

    # Allow cameras to warm up
    time.sleep(2.0)
        
    # Read frames
    frame_1 = camera_1.read()
    frame_2 = camera_2.read() if camera_2 else None

    # Set up coordinate system for the cropped frames
    if camera_2 is not None:
        table_pts_cam1, table_pts_cam2 = manage_point_selection(config, camera_1, camera_2, mtx_1, dst_1, mtx_2, dst_2)
        stitched_frame = get_top_down_view(frame_1, frame_2, table_pts_cam1, table_pts_cam2)
        logger.info(stitched_frame.shape)

    # If networking is enabled, start the server
    network = None
    if config.get("use_networking", False):
        network = Network(config)
        network.start()
        state_manager = StateManager(config, network)

    if config["video_stream"]:
        stream_thread = threading.Thread(target=start_stream)
        stream_thread.start()


    detection_model = DetectionModel(config)

    autoencoder = None
    if not args.collect_ae_data:
        autoencoder = AutoEncoder(config)
    if detection_model.model is None: # Check if model loaded successfully
        return
    
    
    # Process the frames
    while True:
        # Read frames
        frame_1 = camera_1.read()
        frame_2 = camera_2.read() if camera_2 else None

        # Fix any distortion in the cameras
        frame_1, frame_2 = undistort_cameras(config, frame_1, frame_2, mtx_1, dst_1, mtx_2, dst_2)

        if frame_1 is None:
            logger.error("Camera 1 frame is invalid.")
            continue 

        # Get top-down view of the table
        stitched_frame = frame_1 if frame_2 is None else get_top_down_view(frame_1, frame_2, table_pts_cam1, table_pts_cam2)

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
        if not args.collect_ae_data or not args.no_anomaly:
            table_only = detection_model.extract_bounding_boxes(stitched_frame, detected_balls)
            if network:
                if autoencoder.detect_anomaly(table_only):
                    network.send_obstruction(True)
                else:
                    network.send_obstruction(False)


        # Exit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera_1.stop()
    if camera_2 is not None:
        camera_2.stop()
    cv2.destroyAllWindows()
    # if config["use_db"]:
    #     db_controller.cleanup()
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
        default = False
    )

    parser.add_argument(
        "--collect-model-images",
        action="store_true",
        default=False
    )

    return parser.parse_args()

def load_cameras(config):
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

        return camera_1, camera_2
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return
    
def capture_frame_for_training(config, frame):
    path=f"./{config["model_training_path"]}"
    if not os.path.exists(path):
        os.makedirs(path)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        num = randint(0, 10000)
        filename = f"{path}train_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")

def capture_frame(config, frame):
    path = f"./{config["clean_images_path"]}"
    if not os.path.exists(path):
        os.makedirs(path)
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        num = randint(0, 10000)
        filename = f"{path}clean_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")

def reset_ae_data(config):
    path = f"./{config["clean_images_path"]}/"
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
