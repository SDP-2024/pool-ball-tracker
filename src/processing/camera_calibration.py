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
    
    if config["calibrate_cameras"]:
        if len(config["calibration_folders"]) < 1:
            logger.error("No calibration folders supplied. Please check config.yaml.")
            return mtx_1, dst_1, mtx_2, dst_2
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cam_1_calibration_folder = os.path.join(script_dir, "../../config/calibration/" + config["calibration_folders"][0])  # Relative path
        cam_1_calibration_folder = os.path.abspath(cam_1_calibration_folder)
        
        logger.info("Calibrating camera 1")
        mtx_1, dst_1 = calibrate_camera(cam_1_calibration_folder)

        if config["camera_port_2"] != -1:
            if len(config["calibration_folders"]) < 2:
                logger.error("Could not calibrate camera 2. Please check calibration folders are correctly supplied.")
                return mtx_1, dst_1, mtx_2, dst_2

            cam_2_calibration_folder = os.path.join(script_dir, "../../config/calibration/" + config["calibration_folders"][1])
            cam_2_calibration_folder = os.path.abspath(cam_2_calibration_folder)
            logger.info("Calibrating camera 2")
            mtx_2, dst_2 = calibrate_camera(cam_2_calibration_folder)

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
    if config["calibrate_cameras"]:
        if mtx_1 is None or dst_1 is None:
            logger.error("Could not undistort camera 1. (Calibration enabled but no folders supplied?)")
            return frame_1, frame_2
        

        h_1,  w_1 = frame_1.shape[:2]
        new_camera_mtx_1, roi_1 = cv2.getOptimalNewCameraMatrix(mtx_1, dst_1, (w_1,h_1), 1, (w_1,h_1))
        # undistort
        frame_1 = cv2.undistort(frame_1, mtx_1, dst_1, None, new_camera_mtx_1)
        
        # crop the image
        x, y, w, h = roi_1
        frame_1 = frame_1[y:y+h, x:x+w]

        if frame_2 is not None:
            if mtx_2 is None or dst_2 is None:
                logger.error("Could not undistort camera 2. (Calibration enabled but no folders supplied?)")
                return frame_1, frame_2

            h_2,  w_2 = frame_2.shape[:2]
            new_camera_mtx_2, roi_2 = cv2.getOptimalNewCameraMatrix(mtx_2, dst_2, (w_2, h_2), 1, (w_2, h_2))
            # undistort
            frame_2 = cv2.undistort(frame_2, mtx_2, dst_2, None, new_camera_mtx_2)
            
            # crop the image
            x, y, w, h = roi_2
            frame_2 = frame_2[y:y+h, x:x+w]

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


def load_table_points(file_path="config/table_points.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return np.array(data["table_pts_cam1"], dtype=np.float32), np.array(data["table_pts_cam2"], dtype=np.float32)
    except FileNotFoundError:
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None, None


def save_table_points(table_pts_cam1, table_pts_cam2, file_path="config/table_points.json"):
    data = {
        "table_pts_cam1": np.array(table_pts_cam1, dtype=np.float32).tolist(),
        "table_pts_cam2": np.array(table_pts_cam2, dtype=np.float32).tolist()
    }
    with open(file_path, "w") as f:
        json.dump(data, f)
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
            frame_1 = camera_1.read()
            frame_2 = camera_2.read() if camera_2 else None

            frame_1, frame_2 = undistort_cameras(config, frame_1, frame_2, mtx_1, dst_1, mtx_2, dst_2)

            display_frame = frame_1.copy() if selected_cam[0] == 1 else frame_2.copy()

            for pt in (table_pts_cam1 if selected_cam[0] == 1 else table_pts_cam2):
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow("Point Selection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        save_table_points(table_pts_cam1, table_pts_cam2)

    return np.array(table_pts_cam1, dtype=np.float32), np.array(table_pts_cam2, dtype=np.float32)