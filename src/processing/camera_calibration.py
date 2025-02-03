import glob
import cv2 as cv
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


# Select points for table corners
def select_points(event, x, y, flags, param):
    global table_pts_cam1, table_pts_cam2, selected_cam

    if event == cv.EVENT_LBUTTONDOWN:
        if selected_cam == 1:
            table_pts_cam1.append((x, y))
        else:
            table_pts_cam2.append((x, y))

        print(f"Point selected: {(x, y)} for Camera {selected_cam}")

        if len(table_pts_cam1) == 4 and selected_cam == 1:
            print("Switching to Camera 2. Select 4 points.")
            selected_cam = 2
        elif len(table_pts_cam2) == 4:
            print("All points selected!")


# Load previously saved points
def load_table_points(file_path="config/table_points.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        table_pts_cam1 = np.array(data["table_pts_cam1"], dtype=np.float32)
        table_pts_cam2 = np.array(data["table_pts_cam2"], dtype=np.float32)
        return table_pts_cam1, table_pts_cam2
    except FileNotFoundError:
        logger.warning(f"{file_path} not found. Points need to be selected manually.")
        return None, None
    


# Store selected points
def save_table_points(table_pts_cam1, table_pts_cam2, file_path="config/table_points.json"):
    # Ensure points are numpy arrays
    table_pts_cam1 = np.array(table_pts_cam1, dtype=np.float32)
    table_pts_cam2 = np.array(table_pts_cam2, dtype=np.float32)

    data = {
        "table_pts_cam1": table_pts_cam1.tolist(),
        "table_pts_cam2": table_pts_cam2.tolist()
    }
    with open(file_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Table points saved to {file_path}")


def manage_point_selection(config, left_camera, right_camera, mtx_left, dst_left, mtx_right, dst_right):
    table_pts_cam1, table_pts_cam2 = load_table_points()

    if table_pts_cam1 is None or table_pts_cam2 is None:
        selected_cam = 1  # Start with camera 1 for selection
        table_pts_cam1, table_pts_cam2 = [], []

        # Set mouse callback to select points
        cv.setMouseCallback("Stitched Image (Cropped)", select_points)

        logger.info("Select the 4 points for Camera 1 (Top-Left, Top-Right, Bottom-Left, Bottom-Right)")
        while len(table_pts_cam1) < 4 or len(table_pts_cam2) < 4:
            # Read frames
            left_frame = left_camera.read()
            right_frame = right_camera.read() if right_camera else None

            # Fix any distortion in the cameras
            left_frame, right_frame = undistort_cameras(config, left_frame, right_frame, mtx_left, dst_left, mtx_right, dst_right)

            # Display the frame with the selected points
            if selected_cam == 1:
                display_frame = left_frame.copy()
            else:
                display_frame = right_frame.copy()

            for pt in (table_pts_cam1 if selected_cam == 1 else table_pts_cam2):
                cv.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv.imshow("Stitched Image (Cropped)", display_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit selection
                break

        save_table_points(table_pts_cam1, table_pts_cam2)
        return table_pts_cam1, table_pts_cam2
    return table_pts_cam1, table_pts_cam2