import cv2
import argparse
import time
import logging
from random import randint
import threading

from config.config import ConfigManager
from config.config_gui import run_config_interface
from src.processing.camera_calibration import *
from src.processing.frame_processing import get_top_down_view
from src.detection.detection_model import DetectionModel
from src.detection.autoencoder import AutoEncoder
from src.networking.network import Network
from src.logic.game_state import StateManager
from src.logic.calibration_gui import run_calibration_interface

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
def main() -> None:
    """
    Main function for running the pool-pal computer vision system
    """

    args : argparse.Namespace = parse_args()

    if args.config and args.calibrate:
        logger.error("Cannot run config and calibrate interface at the same time. Selecting config.")
        args.calibrate = False

    config = ConfigManager()
    if args.config:
        threading.Thread(target=run_config_interface, args=(config, )).start()

    
    if config is None:
        logger.error("Error getting config file.")
        return

    # Process optional arguments
    if args.collect_ae_data:
        reset_ae_data(config)
    if args.set_points:
        reset_points()
    

    # Process either static image or a live webcam feed
    if args.file is not None:
        frame : cv2.Mat = cv2.imread(args.file)
    else:
        camera : cv2.VideoCapture = load_camera(config)
        # Read frames
        ret, frame = camera.read()
        if not ret:
            logger.error("Error reading camera frame. Exiting")
            return

    # Calibrate camera and undistort
    mtx, dist, newcameramtx, roi = handle_calibration(config, frame)
    undistorted_frame = undistort_camera(config, frame, mtx, dist, newcameramtx, roi)

    # Select points and compute homography
    table_pts : np.ndarray = manage_point_selection(undistorted_frame)
    if table_pts is None:
        logger.error("Table points not selected. Exiting.")
        return
    
    table_rect : np.float32 = np.float32([
        [0, 0], 
        [config.output_width, 0], 
        [0, config.output_height], 
        [config.output_width, config.output_height]
    ])
    homography_matrix : cv2.Mat = cv2.getPerspectiveTransform(table_pts,table_rect)

    undistorted_frame : cv2.Mat = get_top_down_view(undistorted_frame,homography_matrix)

    # Initialise autoencoder
    autoencoder = None
    if not args.collect_ae_data and not args.no_anomaly:
        autoencoder = AutoEncoder(config)
        anomaly = False
        
    # If networking is enabled, start the server
    network = None
    if config.use_networking:
        network = Network(config)
        network.start()
        state_manager = StateManager(config, network, mtx, dist)
    else:
        state_manager = StateManager(config, None, mtx, dist)

    if args.calibrate:
        threading.Thread(target=run_calibration_interface, args=(config, state_manager.offset_manager)).start()
        state_manager.offset_manager.calibrating = True

    # Initialise detection model
    detection_model = DetectionModel(config)
    
    if detection_model.model is None: # Check if model loaded successfully
        return
    
    
    # Process the frames
    while True:
        # Read frame from file or webcam
        if args.file is None:
            ret, frame = camera.read()
            if frame is None:
                logger.error("Camera frame is invalid.")
                continue 
        
        # Undistort camera
        undistorted_frame : cv2.Mat = undistort_camera(config, frame, mtx, dist, newcameramtx, roi)

        # Get top-down view of the table
        undistorted_frame : cv2.Mat = get_top_down_view(undistorted_frame,homography_matrix)
        
        # Process optional arguments
        if args.collect_ae_data: 
            capture_frame(config.clean_images_path, undistorted_frame)
        if args.collect_model_images:
            capture_frame(config.model_training_path, undistorted_frame)

        # Copy frame for drawing
        drawing_frame : cv2.Mat = undistorted_frame.copy()

        # Detect and draw to frame
        detections, labels= detection_model.detect(undistorted_frame)
        detection_model.draw(drawing_frame, detections)

        # Update state
        state_manager.update(detections, labels, drawing_frame)
        
        # Detect anomalies in the frame if required
        if not args.collect_ae_data and not args.no_anomaly:
            table_only : cv2.Mat = detection_model.extract_bounding_boxes(undistorted_frame, detections)
            is_anomaly : bool = autoencoder.detect_anomaly(table_only)
            if is_anomaly:
                cv2.rectangle(drawing_frame, (config.output_width // 2 - 350, config.output_height // 2 - 50), (config.output_width // 2 + 310, config.output_height // 2 + 10), (0, 0, 0), -1)
                cv2.putText(drawing_frame, "Obstruction Detected", ((config.output_width // 2) - 350 , config.output_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow("Detection", drawing_frame)
                if network and not anomaly:
                    anomaly = True
                    network.send_obstruction("true")
                else:
                    logger.info(f"Obstruction detected: {is_anomaly}")
            # Send false if obstruction has been removed
            elif not is_anomaly and anomaly:
                anomaly = False
                network.send_obstruction("false")

        # Exit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    if args.file is None:
        camera.release()
    cv2.destroyAllWindows()
    if config.use_networking:
        network.disconnect()


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments to run the program with the flags.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action='store_true',
        default=False
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

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="The path to the image file."
    )

    parser.add_argument(
        "--calibrate",
        action="store_true",
        default=False
    )

    return parser.parse_args()


def load_camera(config) -> cv2.VideoCapture | None:
    """
    This function loads the camera from the camera port specified in the config.
    It uses the MSMF backend to allow the camera to run at it's native resolution.
    """
    try:
        logger.info("Starting camera...")
        camera = cv2.VideoCapture(config.camera_port, cv2.CAP_MSMF)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        time.sleep(2.0)  # Allow camera to warm up
        return camera
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return


def capture_frame(path : str, frame : cv2.Mat) -> None:
    """
    This function captures a the current frame and saves it to the specified path.
    """
    path : str = f"./{path}/"
    if not os.path.exists(path):
        os.makedirs(path)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        num : int = randint(0, 10000)
        filename : str = f"{path}image_{num}.jpg"
        cv2.imwrite(filename, frame)
        time.sleep(0.1)
        logger.info(f"Image {num} saved")


def reset_ae_data(config) -> None:
    """
    This deletes the clean images path so that the autoencoder can be made with fresh images.
    """
    path : str = f"./{config.clean_images_path}/"
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(path + file)
        logger.info("Data reset.")


def reset_points() -> None:
    """
    This deletes the table points so that new points can be selected.
    """
    path : str = "./config/table_points.json"
    if os.path.exists(path):
        os.remove(path)
        logger.info("Points reset.")

if __name__ == "__main__":
    main()
