import glob
import cv2
from cv2 import aruco
import numpy as np
import os
import logging
import json

logger = logging.getLogger(__name__)

def calibrate_camera(in_path : str, frame : cv2.Mat, out_path : str ="./src/processing/camera_calibration.json") -> tuple[np.ndarray, np.ndarray, cv2.Mat, tuple]:
    """
    Calibrates a camera using a set of images of a Charuco board pattern.

    Args:
        path (str): Path to the images used for calibration.
        frame (numpy.ndarray): A frame from the camera to get the image size.

    Returns:
        tuple: A tuple containing:
            - mtx (np.ndarray): The camera matrix.
            - dist (np.ndarray): The distortion coefficients.
            - newcameramtx (np.ndarray): The optimized camera matrix.
            - roi (tuple): The region of interest after undistortion.
    """

    # Loads calibration information if it already exists
    if os.path.exists(out_path):
        logger.info("Camera calibration already exists. Skipping calibration.")
        with open(out_path, "r") as json_file:
            data = json.load(json_file)
        mtx : np.ndarray= np.array(data["mtx"])
        dist : np.ndarray= np.array(data["dist"])
        h,  w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        return mtx, dist, newcameramtx, roi
    
    logger.info("Calibration data not found, calibrating camera.")

    # Define Charuco board
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    board = aruco.CharucoBoard((6, 8), 34, 27, ARUCO_DICT)

    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    # Get images
    images : list = glob.glob(os.path.join(in_path, "*.jpg"))
    image : cv2.Mat = cv2.imread(images[0])
    image : cv2.Mat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgSize = image.shape

    # Extract charuco corners and ids from images
    all_charuco_ids, all_charuco_corners = get_charuco_corners_and_ids(images, detector, board)

    # Calibrate camera with extracted information
    _, mtx, dist, _, _ = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)

    data : dict = {"mtx": mtx.tolist(), "dist": dist.tolist()}

    with open(out_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    logging.info("Calibration data saved to camera_calibration.json")
    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    return mtx, dist, newcameramtx, roi


def get_charuco_corners_and_ids(images, detector, board)  -> tuple[list[cv2.Mat], list[cv2.Mat]]:
    """
    Helper function to extract all the charuco ids and corners from the images
    """
    all_charuco_ids : list[cv2.Mat] = []
    all_charuco_corners: list[cv2.Mat] = []
    # Loop over images and extraction of corners
    for image_file in images:
        image : cv2.Mat = cv2.imread(image_file)
        image : cv2.Mat = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = detector.detectMarkers(image)
        
        # Ensure enough markers are detected
        if len(marker_ids) > 0:
            _, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)

    return all_charuco_ids, all_charuco_corners




def handle_calibration(config, frame : cv2.Mat) -> tuple:
    """
    Handles the calibration of the camera.

    Args:
        config (dict): Configuration dictionary containing calibration parameters.
        frame (numpy.ndarray): A frame from the camera to get the image size.

    Returns:
        tuple: Contains the camera matrices, distortion coefficients, optimized matrices, and ROI.
               (mtx, dist, newcameramtx, roi)
    """

    mtx, dist, newcameramtx, roi = None, None, None, None
    if not config.calibrate_camera:
        return mtx, dist, newcameramtx, roi
    calibration_folder = config.calibration_folder
    if len(calibration_folder) < 1:
        logger.error("No calibration folder supplied. Please check config.yaml.")
        return mtx, dist, newcameramtx, roi

    mtx, dist, newcameramtx, roi = calibrate_camera(os.path.abspath((calibration_folder)), frame)
    return mtx, dist, newcameramtx, roi



def undistort_camera(config, frame : cv2.Mat, mtx : np.ndarray, dist : np.ndarray, newcameramtx : np.ndarray, roi : tuple) -> cv2.Mat:
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
    if config.calibrate_camera:
        if frame is None or mtx is None or dist is None:
            logging.warning("Frame, mtx, or dist is none. Cannot undistort.")
            return frame
        undistorted_frame : cv2.Mat = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # Crop the image to the ROI
        x, y, w, h = roi
        cropped_frame : cv2.Mat = undistorted_frame[y:y+h, x:x+w]
        return cropped_frame

    return frame




def select_points(event, x : int, y : int, _, param) -> None:
    """
    Function to select the points on the frame, a callback from manage_point_selection
    """

    table_pts : list = param  # Get the list from param

    if event == cv2.EVENT_LBUTTONDOWN and len(table_pts) < 4:
        table_pts.append((x, y))
        logger.info(f"Point selected: {(x, y)}")


def load_table_points(file_path : str ="config/table_points.json") -> None | np.ndarray:
    """
    Loads the table points from the saved file.

    Args:
        file_path (str, optional): The file to load the points from. Defaults to "config/table_points.json".

    Returns:
        numpy.array: An array of the saved points
    """
    if not os.path.exists(file_path):
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None

    with open(file_path, "r") as f:
        data : dict = json.load(f)
    
    try:
        table_pts : np.ndarray = np.array(data["table_pts"], dtype=np.float32)
    except KeyError as e:
        logger.error(f"Key error: {e}. Points need to be selected manually.")
        return None

    return table_pts



def save_table_points(table_pts : list, file_path : str ="config/table_points.json") -> None:
    """
    Save the table points to a file so they can be loaded next time the program is run

    Args:
        table_pts (list): The table points as a list to be saved
        file_path (str, optional): The optional path for saving the points to. Defaults to "config/table_points.json".
    """
    data : dict = {
        "table_pts": [list(pt) for pt in table_pts],
    }
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Table points saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving table points: {e}")


def manage_point_selection(frame : cv2.Mat) -> np.ndarray:
    """
    Manages the point selection for cropping the view of the pool table.

    Args:
        frame (numpy.ndarray): The static frame to be cropped

    Returns:
        numpy.array: The array of table points
    """
    sorted_pts : np.ndarray = load_table_points()

    if sorted_pts is None:
        table_pts : list = []

        cv2.namedWindow("Point Selection")
        cv2.setMouseCallback("Point Selection", select_points, param=table_pts)

        logger.info("Select 4 points for Camera")

        while True:
                
            display_frame : cv2.Mat = frame.copy()

            for pt in table_pts:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
                cv2.putText(display_frame, f"{pt}", (pt[0] + 10, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw lines between all other points
            if len(table_pts) < 4:
                for i in range(1, len(table_pts)):
                    cv2.line(display_frame, table_pts[i - 1], table_pts[i], (0, 0, 255), 2)

            # Complete the rectangle        
            if len(table_pts) == 4:
                sorted_pts = sort_points(table_pts)
                cv2.line(display_frame, sorted_pts[0], sorted_pts[1], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[1], sorted_pts[3], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[3], sorted_pts[2], (0, 0, 255), 2)
                cv2.line(display_frame, sorted_pts[2], sorted_pts[0], (0, 0, 255), 2)
                cv2.putText(display_frame, "Press Enter to confirm points", (frame.shape[1] // 2 - 150, frame.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Point Selection", display_frame)
            key : int = cv2.waitKey(1) & 0xFF
            
            # Press enter key to confirm points if all are selected
            if (key == ord('\n') or key == ord('\r')) and len(table_pts) == 4:
                break
            # Remove the most recent point if backspace pressed
            if key == ord('\b') and len(table_pts) > 0:
                logger.info(f"Point {table_pts[-1]} removed.")
                table_pts.pop()
                
            if key == ord('q'):
                logger.info("Point selection aborted by user.")
                return None
            
        # Sort the table points in to the correct order
        sorted_pts : list = sort_points(table_pts)
        save_table_points(sorted_pts)
        cv2.destroyWindow("Point Selection")

    return np.array(sorted_pts, dtype=np.float32)


def sort_points(table_pts : list) -> list:
    """
    Sorts the points in the order: top-left, top-right, bottom-left, bottom-right.

    Args:
        table_pts (list): List of points to be sorted.

    Returns:
        list: Sorted list of points.
    """
    # Sort by y-coordinate (top to bottom)
    table_pts : list = sorted(table_pts, key=lambda x: x[1])
    top_pts : list= sorted(table_pts[:2], key=lambda x: x[0])
    bottom_pts : list = sorted(table_pts[2:], key=lambda x: x[0])
    sorted_pts : list = [top_pts[0], top_pts[1], bottom_pts[0], bottom_pts[1]]
    
    return sorted_pts
