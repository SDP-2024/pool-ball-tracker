import glob
import cv2
import numpy as np
import os
import logging
import json
from imutils.video import VideoStream, WebcamVideoStream

logger = logging.getLogger(__name__)

def calibrate_camera(path, chessboard_size=(8, 5)):
    """
    Calibrates a camera using a set of images of a chessboard pattern.

    Args:
        path (str): Path to the images used for calibration.
        chessboard_size (tuple, optional): The number of internal corners in the chessboard pattern 
                                           as (columns, rows). Defaults to (8, 5).

    Returns:
        tuple: A tuple containing:
            - mtx (np.ndarray): The camera matrix.
            - dist (np.ndarray): The distortion coefficients.
    """
    
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in image plane

    # Prepare a grid of points for the chessboard pattern
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Get all image files
    images = glob.glob(os.path.join(path, "*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {path}")

    for img_path in images:
        img = cv2.imread(img_path)  # Read the image
        if img is None:
            logger.error(f"Could not read {img_path}")
            continue  # Skip unreadable images

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    # Ensure calibration data exists
    if not obj_points or not img_points:
        raise ValueError("No chessboard corners found in any image. Check the images or chessboard size.")

    # Calibrate the camera
    ret, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist




def handle_calibration(config):
    """
    Handles the calibration of the cameras.

    Args:
        config (dict): Configuration dictionary containing calibration parameters.

    Returns:
        tuple: Contains the camera matrices, distortion coefficients, optimized matrices, and ROI.
               (mtx, dst, new_mtx, roi)
    """

    mtx, dst = None, None
    if not config.get("calibrate_camera", False):
        return mtx, dst
    calibration_folder = config.get("calibration_folder", [])
    if len(calibration_folder) < 1:
        logger.error("No calibration folder supplied. Please check config.yaml.")
        return mtx, dst
    
    def calibrate_and_log(folder):
        folder_path = os.path.abspath(folder)
        logger.info(f"Calibrating camera using {folder_path}")
        return calibrate_camera(folder_path)

    mtx, dst = calibrate_and_log(calibration_folder[0])
    return mtx, dst



def undistort_camera(config, frame, mtx, dst):
    """
    Undistorts the input frames using the camera calibration parameters.

    Args:
        config (dict): Configuration dictionary containing calibration settings.
        frame (numpy.ndarray): The frame from camera.
        mtx (numpy.ndarray): Camera matrix for camera.
        dst (numpy.ndarray): Distortion coefficients for camera.

    Returns:
        tuple: The undistorted frames.
    """
    def undistort_frame(frame, mtx, dst):
        if frame is None or mtx is None or dst is None:
            return frame
        h, w = frame.shape[:2]
        new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dst, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dst, None, new_camera_mtx)
        return undistorted_frame

    if config.get("calibrate_cameras", False):
        frame = undistort_frame(frame, mtx, dst)

    return frame




def select_points(event, x, y, flags, param):
    table_pts = param  # Get the list from param

    if event == cv2.EVENT_LBUTTONDOWN:
        table_pts.append((x, y))
        logger.info(f"Point selected: {(x, y)}")

        if len(table_pts) == 4:
            logger.info("All points selected!")
            cv2.setMouseCallback("Point Selection", lambda *args: None)  # Disable further callbacks


def load_table_points(file_path="config/table_points.json"):
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)
    
    try:
        table_pts = np.array(data["table_pts"], dtype=np.float32)
    except KeyError as e:
        logger.error(f"Key error: {e}. Points need to be selected manually.")
        return None

    return table_pts



def save_table_points(table_pts, file_path="config/table_points.json"):
    data = {
        "table_pts": [list(pt) for pt in table_pts],
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Table points saved to {file_path}")


def manage_point_selection(config, camera, mtx, dst):
    table_pts = load_table_points()

    if table_pts is None:
        table_pts = []

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=table_pts)

        logger.info("Select 4 points for Camera (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")

        while len(table_pts) < 4:
            if isinstance(camera, WebcamVideoStream):
                frame = camera.read()
                if frame is None:
                    logger.error("Failed to grab frame from Camera")
                    return None
            else:
                frame = camera
            frame = undistort_camera(config, frame, mtx, dst)
            
            display_frame = frame.copy()

            for pt in table_pts:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow("Point Selection", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Point selection aborted by user.")
                return None

        save_table_points(table_pts)
        cv2.destroyWindow("Point Selection")

    return np.array(table_pts, dtype=np.float32)