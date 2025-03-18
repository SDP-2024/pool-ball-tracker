import glob
import cv2
import numpy as np
import os
import logging
import json

logger = logging.getLogger(__name__)

def calibrate_camera(path, chessboard_size=(8, 5), square_size=2.9):
    """
    Calibrates a wide-angle camera using a set of chessboard images.

    Args:
        path (str): Path to the images used for calibration.
        chessboard_size (tuple): The number of internal corners in the chessboard pattern (columns, rows).
        square_size (float): Real-world size of a square on the chessboard (optional, default is 1.0).

    Returns:
        tuple: A tuple containing:
            - mtx (np.ndarray): The camera matrix.
            - dist (np.ndarray): The distortion coefficients.
    """

    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Prepare object points (chessboard pattern in real-world space)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    images = glob.glob(os.path.join(path, "*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images found in {path}")

    for img_path in images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            refined_corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1),
                                               (cv2.TERM_CRITERIA_EPS +
                                                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            imgpoints.append(refined_corners)
            objpoints.append(objp)

    if not objpoints or not imgpoints:
        raise ValueError("No valid chessboard corners found. Check images or chessboard size.")

    h, w = gray.shape[:2]
    mtx = np.eye(3, dtype=np.float32)
    dist = np.zeros((4, 1), dtype=np.float32)

    # Calibrate camera using standard distortion model (5 parameters)
    ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (w, h), None, None)

    # Compute reprojection error, good: < 0.5 pixels, bad: > 1 pixel
    total_error = 0
    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
        total_error += error

    reprojection_error = total_error / len(objpoints)
    logger.info(f"Reprojection Error: {reprojection_error:.4f}")

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
        frame (numpy.ndarray): The frame from the camera.
        mtx (numpy.ndarray): Original camera matrix.
        dst (numpy.ndarray): Distortion coefficients.

    Returns:
        numpy.ndarray: The undistorted frame.
    """
    
    def undistort_frame(frame, mtx, dst):
        if frame is None or mtx is None or dst is None:
            return frame
        undistorted_frame = cv2.fisheye.undistortImage(frame, mtx, dst)

        return undistorted_frame

    if config.get("calibrate_camera", False):
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


def manage_point_selection(camera):
    table_pts = load_table_points()

    if table_pts is None:
        table_pts = []

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=table_pts)

        logger.info("Select 4 points for Camera (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")

        while len(table_pts) < 4:
            frame = camera.read()
            if frame is None:
                logger.error("Failed to grab frame from Camera")
                return None

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