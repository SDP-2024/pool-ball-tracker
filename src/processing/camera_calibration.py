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
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist


def handle_calibration(config):
    """
    Handles the calibration of the cameras.

    Args:
        config (dict): Configuration dictionary containing calibration parameters,
                       including "calibrate_cameras" and "calibration_folders".

    Returns:
        tuple: Contains the camera matrices and distortion coefficients for both cameras.
               (mtx_1, dst_1, mtx_2, dst_2)
    """

    mtx_1, dst_1, mtx_2, dst_2 = None, None, None, None
    
    if not config.get("calibrate_cameras", False):
        return mtx_1, dst_1, mtx_2, dst_2

    calibration_folders = config.get("calibration_folders", [])
    if len(calibration_folders) < 1:
        logger.error("No calibration folders supplied. Please check config.yaml.")
        return mtx_1, dst_1, mtx_2, dst_2
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    def calibrate_and_log(camera_index, folder):
        folder_path = os.path.abspath(os.path.join(script_dir, "../../config/calibration/", folder))
        logger.info(f"Calibrating camera {camera_index}")
        return calibrate_camera(folder_path)

    mtx_1, dst_1 = calibrate_and_log(1, calibration_folders[0])

    if config.get("camera_port_2", -1) != -1:
        if len(calibration_folders) < 2:
            logger.error("Could not calibrate camera 2. Please check calibration folders are correctly supplied.")
            return mtx_1, dst_1, mtx_2, dst_2

        mtx_2, dst_2 = calibrate_and_log(2, calibration_folders[1])

    return mtx_1, dst_1, mtx_2, dst_2


def undistort_cameras(config, frame_1, frame_2, mtx_1, dst_1, mtx_2, dst_2):
    """
    Undistorts the input frames using the camera calibration parameters.

    Args:
        config (dict): Configuration dictionary containing calibration settings.
        frame_1 (numpy.ndarray): The frame from camera 1.
        frame_2 (numpy.ndarray): The frame from camera 2.
        mtx_1 (numpy.ndarray): Camera matrix for camera 1.
        dst_1 (numpy.ndarray): Distortion coefficients for camera 1.
        mtx_2 (numpy.ndarray): Camera matrix for camera 2.
        dst_2 (numpy.ndarray): Distortion coefficients for camera 2.

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
        frame_2 = undistort_frame(frame_2, mtx_2, dst_2)

    return frame_1, frame_2


def select_points(event, x, y, flags, param):
    table_pts_cam1, table_pts_cam2, selected_cam = param

    if event == cv2.EVENT_LBUTTONDOWN:
        if selected_cam[0] == 1:
            table_pts_cam1.append((x, y))
        else:
            table_pts_cam2.append((x, y))

        logger.info(f"Point selected: {(x, y)} for Camera {selected_cam[0]}")

        if len(table_pts_cam1) == 4 and selected_cam[0] == 1:
            logger.info("Switching to Camera 2. Select 4 points.")
            selected_cam[0] = 2
        elif len(table_pts_cam2) == 4:
            logger.info("All points selected!")
            cv2.setMouseCallback("Point Selection", lambda *args: None)  # Disable further callbacks


def load_table_points(file_path="config/table_points.json"):
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None, None

    with open(file_path, "r") as f:
        data = json.load(f)
    
    try:
        table_pts_cam1 = np.array(data["table_pts_cam1"], dtype=np.float32)
        table_pts_cam2 = np.array(data["table_pts_cam2"], dtype=np.float32)
    except KeyError as e:
        logger.error(f"Key error: {e}. Points need to be selected manually.")
        return None, None

    return table_pts_cam1, table_pts_cam2


def save_table_points(table_pts_cam1, table_pts_cam2, file_path="config/table_points.json"):
    data = {
        "table_pts_cam1": [list(pt) for pt in table_pts_cam1],
        "table_pts_cam2": [list(pt) for pt in table_pts_cam2]
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Table points saved to {file_path}")


def manage_point_selection(config, camera_1, camera_2, mtx_1, dst_1, mtx_2, dst_2):
    table_pts_cam1, table_pts_cam2 = load_table_points()

    if table_pts_cam1 is None or table_pts_cam2 is None:
        table_pts_cam1, table_pts_cam2 = [], []
        selected_cam = [1]

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=(table_pts_cam1, table_pts_cam2, selected_cam))

        logger.info("Select 4 points for Camera 1 (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")

        while len(table_pts_cam1) < 4 or len(table_pts_cam2) < 4:
            ret_1, frame_1 = camera_1.read()
            ret_2, frame_2 = camera_2.read() if camera_2 else (False, None)

            if not ret_1 or (camera_2 and not ret_2):
                logger.error("Failed to read from one or both cameras.")
                break

            frame_1, frame_2 = undistort_cameras(config, frame_1, frame_2, mtx_1, dst_1, mtx_2, dst_2)

            display_frame = frame_1.copy() if selected_cam[0] == 1 else frame_2.copy()

            for pt in (table_pts_cam1 if selected_cam[0] == 1 else table_pts_cam2):
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow("Point Selection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Point selection aborted by user.")
                break

        save_table_points(table_pts_cam1, table_pts_cam2)
        cv2.destroyWindow("Point Selection")

    return np.array(table_pts_cam1, dtype=np.float32), np.array(table_pts_cam2, dtype=np.float32)