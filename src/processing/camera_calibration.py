import glob
import cv2
from cv2 import aruco
import numpy as np
import os
import logging
import json
from imutils.video import VideoStream, WebcamVideoStream

logger = logging.getLogger(__name__)

def calibrate_camera(path, frame):
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

    if os.path.exists("./camera_calibration.json"):
        logger.info("Camera calibration already exists. Skipping calibration.")
        with open("./camera_calibration.json", "r") as json_file:
            data = json.load(json_file)
        mtx = np.array(data["mtx"])
        dist = np.array(data["dist"])
        logging.info(f"mtx: {mtx}, dist: {dist}")
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        return mtx, dist, newcameramtx, roi
    
    logger.info("Calibrating camera...")
    # Define Charuco board
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((6, 8), 34, 27, ARUCO_DICT)


    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    all_charuco_ids = []
    all_charuco_corners = []
    images = glob.glob(os.path.join(path, "*.jpg"))

    # Loop over images and extraction of corners
    for image_file in images:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = image.shape
        image_copy = image.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        
        if len(marker_ids) > 0: # If at least one marker is detected
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)

    # Calibrate camera with extracted information
    result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)

    OUTPUT_JSON = "./camera_calibration.json"

    data = {"mtx": mtx.tolist(), "dist": dist.tolist()}

    with open(OUTPUT_JSON, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logging.info("Calibration data saved to camera_calibration.json")
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return mtx, dist, newcameramtx, roi




def handle_calibration(config, frame):
    """
    Handles the calibration of the cameras.

    Args:
        config (dict): Configuration dictionary containing calibration parameters.

    Returns:
        tuple: Contains the camera matrices, distortion coefficients, optimized matrices, and ROI.
               (mtx, dst, new_mtx, roi)
    """

    mtx, dist, newcameramtx, roi = None, None, None, None
    if not config.get("calibrate_camera", False):
        return mtx, dist, newcameramtx, roi
    calibration_folder = config.get("calibration_folder", [])
    if len(calibration_folder) < 1:
        logger.error("No calibration folder supplied. Please check config.yaml.")
        return mtx, dist, newcameramtx, roi

    mtx, dist, newcameramtx, roi = calibrate_camera(os.path.abspath((calibration_folder[0])), frame)
    return mtx, dist, newcameramtx, roi



def undistort_camera(config, frame, mtx, dist, newcameramtx, roi):
    """
    Undistorts the input frames using the camera calibration parameters.

    Args:
        config (dict): Configuration dictionary containing calibration settings.
        frame (numpy.ndarray): The frame from camera.
        mtx (numpy.ndarray): Camera matrix for camera.
        dist (numpy.ndarray): Distortion coefficients for camera.
        newcameramtx (numpy.ndarray): New camera matrix for camera.
        roi (tuple): Region of interest after undistortion.

    Returns:
        numpy.ndarray: The undistorted frame.
    """
    if config.get("calibrate_camera", False):
        logger.info(f"{frame.shape}")
        if frame is None or mtx is None or dist is None:
            logging.warning("Frame, mtx, or dist is none. Cannot undistort.")
            return frame
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop the image to the ROI
        x, y, w, h = roi
        cropped_frame = undistorted_frame[y:y+h, x:x+w]
        logger.info("Undistorted frame.")
        return cropped_frame

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


def manage_point_selection(config, frame, mtx, dist, newcameramtx, roi):
    table_pts = load_table_points()

    if table_pts is None:
        table_pts = []

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=table_pts)

        logger.info("Select 4 points for Camera (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")

        while len(table_pts) < 4:
            #undistorted_frame = undistort_camera(config, frame, mtx, dist, newcameramtx, roi)
            
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