import glob
import cv2 as cv
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def calibrate_camera(path, chessboard_size=(8, 5)):
    """
    Calibrates a camera using a set of images of a chessboard pattern.

    Args:
        path (str): Path to the images used for calibration.
        chessboard_size (tuple, optional): The number of internal corners in the chessboard pattern 
                                           as (columns, rows). Defaults to (9, 6).

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
    images = glob.glob(path + "/*.jpg")
    if not images:
        raise FileNotFoundError(f"No images found in {path}")

    gray = None

    for img_path in images:
        img = cv.imread(img_path)  # Read the image
        if img is None:
            logger.error(f"Could not read {img_path}")
            continue  # Skip unreadable images

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            img_points.append(corners)
            obj_points.append(objp)

    # Ensure calibration data exists
    if not obj_points or not img_points:
        raise ValueError("No chessboard corners found in any image. Check the images or chessboard size.")

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist


def handle_calibration(config):
    """
    Handles the calibration of the left and right cameras.

    Args:
        config (dict): Configuration dictionary containing calibration parameters,
                       including "calibrate_cameras" and "calibration_folders".

    Returns:
        tuple: Contains the camera matrices and distortion coefficients for both cameras.
               (mtx_left, dst_left, mtx_right, dst_right)
    """

    mtx_left, dst_left, mtx_right, dst_right = None, None, None, None
    
    if config["calibrate_cameras"]:
        if len(config["calibration_folders"]) < 1:
            logger.error("No calibration folders supplied. Please check config.yaml.")
            return mtx_left, dst_left, mtx_right, dst_right
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cam1_calibration_dir = os.path.join(script_dir, "../../config/calibration/" + config["calibration_folders"][0])  # Relative path
        cam1_calibration_dir = os.path.abspath(cam1_calibration_dir)
        
        logger.info("Calibrating left camera")
        mtx_left, dst_left = calibrate_camera(cam1_calibration_dir)

        if config["camera_port_2"] != -1:
            if len(config["calibration_folders"]) < 2:
                logger.error("Could not calibrate right camera. Please check calibration folders are correctly supplied.")
                return mtx_left, dst_left, mtx_right, dst_right

            cam2_calibration_dir = os.path.join(script_dir, "../../config/calibration/" + config["calibration_folders"][1])
            cam2_calibration_dir = os.path.abspath(cam2_calibration_dir)
            logger.info("Calibrating right camera")
            mtx_right, dst_right = calibrate_camera(cam2_calibration_dir)

    return mtx_left, dst_left, mtx_right, dst_right


def undistort_cameras(config, left_frame, right_frame, mtx_left, dst_left, mtx_right, dst_right):
    """
    Undistorts the input frames using the camera calibration parameters.

    Args:
        config (dict): Configuration dictionary containing calibration settings.
        left_frame (numpy.ndarray): The left image frame.
        right_frame (numpy.ndarray): The right image frame.
        mtx_left (numpy.ndarray): Camera matrix for the left camera.
        dst_left (numpy.ndarray): Distortion coefficients for the left camera.
        mtx_right (numpy.ndarray): Camera matrix for the right camera.
        dst_right (numpy.ndarray): Distortion coefficients for the right camera.

    Returns:
        tuple: The undistorted left and right frames.
    """
    if config["calibrate_cameras"]:
        if mtx_left is None or dst_left is None:
            logger.error("Could not undistort left camera. (Calibration enabled but no folders supplied?)")
            return left_frame, right_frame
        

        h_left,  w_left = left_frame.shape[:2]
        newcameramtx_left, roi_left = cv.getOptimalNewCameraMatrix(mtx_left, dst_left, (w_left,h_left), 1, (w_left,h_left))
        # undistort
        left_frame = cv.undistort(left_frame, mtx_left, dst_left, None, newcameramtx_left)
        
        # crop the image
        x, y, w, h = roi_left
        left_frame = left_frame[y:y+h, x:x+w]

        if right_frame is not None:
            if mtx_right is None or dst_right is None:
                logger.error("Could not undistort right camera. (Calibration enabled but no folders supplied?)")
                return left_frame, right_frame

            h_right,  w_right = right_frame.shape[:2]
            newcameramtx_right, roi_right = cv.getOptimalNewCameraMatrix(mtx_right, dst_right, (w_right, h_right), 1, (w_right, h_right))
            # undistort
            right_frame = cv.undistort(right_frame, mtx_right, dst_right, None, newcameramtx_right)
            
            # crop the image
            x, y, w, h = roi_right
            right_frame = right_frame[y:y+h, x:x+w]

    return left_frame, right_frame