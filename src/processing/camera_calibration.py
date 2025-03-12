import glob
import cv2
import numpy as np
import os
import logging
import json

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
        config (dict): Configuration dictionary containing calibration parameters,
                       including "calibrate_cameras" and "calibration_folders".

    Returns:
        tuple: Contains the camera matrices and distortion coefficients for both cameras.
               (mtx_1, dst_1)
    """

    mtx_1, dst_1 = None, None
    
    if not config.get("calibrate_cameras", False):
        return mtx_1, dst_1

    calibration_folders = config.get("calibration_folders", [])
    if len(calibration_folders) < 1:
        logger.error("No calibration folders supplied. Please check config.yaml.")
        return mtx_1, dst_1
    
    
    def calibrate_and_log(camera_index, folder):
        folder_path = os.path.abspath(folder)
        logger.info(f"Calibrating camera {camera_index}")
        return calibrate_camera(folder_path)

    mtx_1, dst_1 = calibrate_and_log(1, calibration_folders[0])

    return mtx_1, dst_1


def undistort_cameras(config, frame_1, mtx_1, dst_1):
    """
    Undistorts the input frames using the camera calibration parameters.

    Args:
        config (dict): Configuration dictionary containing calibration settings.
        frame_1 (numpy.ndarray): The frame from camera 1.
        mtx_1 (numpy.ndarray): Camera matrix for camera 1.
        dst_1 (numpy.ndarray): Distortion coefficients for camera 1.

    Returns:
        tuple: The undistorted frames.
    """
    def undistort_frame(frame, mtx, dst):
        if frame is None or mtx is None or dst is None:
            return frame
        h, w = frame.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dst, None, new_camera_mtx)
        x, y, w, h = roi
        return undistorted_frame[y:y+h, x:x+w]

    if config.get("calibrate_cameras", False):
        frame_1 = undistort_frame(frame_1, mtx_1, dst_1)

    return frame_1


def select_points(event, x, y, param):
    table_pts_cam1, selected_cam = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if selected_cam[0] == 1:
            table_pts_cam1.append((x, y))
        logger.info(f"Point selected: {(x, y)} for Camera {selected_cam[0]}")

        if len(table_pts_cam1) == 4:
            logger.info("All points selected!")
            cv2.setMouseCallback("Point Selection", lambda *args: None)  # Disable further callbacks


def load_table_points(file_path="config/table_points.json"):
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None

    with open(file_path, "r") as f:
        data = json.load(f)
    
    try:
        table_pts_cam1 = np.array(data["table_pts_cam1"], dtype=np.float32)
    except KeyError as e:
        logger.error(f"Key error: {e}. Points need to be selected manually.")
        return None

    return table_pts_cam1


def save_table_points(table_pts_cam1, file_path="config/table_points.json"):
    data = {
        "table_pts_cam1": [list(pt) for pt in table_pts_cam1],
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Table points saved to {file_path}")


def manage_point_selection(camera_1):
    table_pts_cam1 = load_table_points()

    if table_pts_cam1 is None:
        table_pts_cam1 = []
        selected_cam = [1]

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=(table_pts_cam1, selected_cam))

        logger.info("Select 4 points for Camera (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")

        while len(table_pts_cam1) < 4:
            frame_1 = camera_1.read()
            if frame_1 is None:
                logger.error("Failed to grab frame from Camera 1")
                return None, None
            

            display_frame = frame_1.copy()

            for pt in table_pts_cam1:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow("Point Selection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Point selection aborted by user.")
                break

        save_table_points(table_pts_cam1)
        cv2.destroyWindow("Point Selection")

    return np.array(table_pts_cam1, dtype=np.float32)