import cv2
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class OffsetManager:
    def __init__(self, config, mtx, dist):
        self.config = config
        # Calibration parameters
        self.calibrating = False
        self.camera_matrix = mtx
        self.dist = dist
        self.calibration_mode = self.config.calibration_mode

        self.grid_size = 100
        self.selected_cell = (0,0)
        self.selected_cell_values = (0,0)
        self.saved_grid = {} # Saved grids indexed by the grid size

        self.x_scaling_factor = 0
        self.y_scaling_factor = 0
        self.mirror_scaling = False

        self.x_linear = 0
        self.y_linear = 0
        self.mirror_linear = False

        self.matrix_correction_factor = 0

        self.parameters_path = "./src/logic/calibration_parameters.json"
        self.load_all_parameters()


    def update(self, frame, middlex, middley):
        if self.calibration_mode == 0:
            if self.calibrating:
                frame = self._handle_grid(frame)
            corrected_middlex, corrected_middley = self._correct_position_grid(middlex, middley)
        elif self.calibration_mode == 1:
            corrected_middlex, corrected_middley = self._correct_position_matrix(middlex, middley)
        elif self.calibration_mode == 2:
            corrected_middlex, corrected_middley = self._correct_position_scaling(middlex, middley)
        elif self.calibration_mode == 3:
            corrected_middlex, corrected_middley = self._correct_position_linear(middlex, middley)
        else:
            corrected_middlex, corrected_middley = middlex, middley
        return int(corrected_middlex), int(corrected_middley)
    

    def _correct_position_scaling(self, middlex, middley):
        """
        Scale the coordinates by a constant scaling factor
        """
        if not self.mirror_scaling:
            corrected_x = middlex + (middlex * self.x_scaling_factor)
            corrected_y = middley + (middley * self.y_scaling_factor)

            return int(corrected_x), int(corrected_y)
        else:
            if middlex <= self.config.output_width // 2:
                corrected_x = middlex + ((self.config.output_width // 2 - middlex) * self.x_scaling_factor)
            else:
                corrected_x = middlex - ((middlex - self.config.output_width // 2) * self.x_scaling_factor)
            corrected_y = middley + (middley * self.y_scaling_factor)

            return int(corrected_x), int(corrected_y)


    def _correct_position_grid(self, middlex, middley):
        """
        Corrects ball position using scaling factor based on the coordinates

        Args:
            middlex (int): X-coordinate of the detected ball.
            middley (int): Y-coordinate of the detected ball.

        Returns:
            tuple: Corrected (x, y) coordinates.
        """
        distance_from_center_x = self.config.output_width / 2 - middlex

        offset_400 = (((self.config.output_width/2) + middlex) * 0.025)
        offset_200 = (((self.config.output_width/2) + middlex) * 0.023)
        offset_100 = (((self.config.output_width/2) + middlex) * 0.017)
        
        if distance_from_center_x < -400:
            corrected_x = int(middlex + offset_400)
        elif distance_from_center_x >= -400 and distance_from_center_x < -200:
            corrected_x = int(middlex + offset_200)
        elif distance_from_center_x >= -200 and distance_from_center_x < -100:
            corrected_x = int(middlex + offset_100)
        elif distance_from_center_x >= -100 and distance_from_center_x < 100:
            corrected_x = middlex
        elif distance_from_center_x >= 100 and distance_from_center_x < 200:
            corrected_x = int(middlex - offset_100)
        elif distance_from_center_x >= 200 and distance_from_center_x < 400:
            corrected_x = int(middlex - offset_200)
        elif distance_from_center_x >= 400:
            corrected_x = int(middlex - offset_400)
        else:
            corrected_x = middlex

        if middley < 100:
            corrected_y = middley
        elif middley >= 100 and middley < 200:
            corrected_y = middley + (middley * 0.015)
        elif middley >= 200 and middley < 400:
            corrected_y = middley + (middley * 0.025)
        elif middley >= 400:
            corrected_y = middley + (middley * 0.035)
        else:
            corrected_y = middley

        return int(corrected_x), int(corrected_y)
    

    def _handle_grid(self, frame):
        """
        This function handles the grid.
        It detects which cell is selected, and works with the calibration tool to track offsets per cell.
        """
        cv2.setMouseCallback("Detection", self._select_cell)
        frame = self._draw_grid(frame)

        if self.selected_cell is not None:
            top_left = (self.selected_cell[0] * self.grid_size, self.selected_cell[1] * self.grid_size)
            bottom_right = (top_left[0] + self.grid_size, top_left[1] + self.grid_size)
            frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            if f"{self.grid_size}" in self.saved_grid and self.selected_cell in self.saved_grid[f"{self.grid_size}"]:
                self.selected_cell_values = (self.saved_grid[f"{self.grid_size}"][self.selected_cell]['x'], self.saved_grid[f"{self.grid_size}"][self.selected_cell]['y'])
            else:
                self.selected_cell_values = (0,0)
        
        # Save the current state of the grid
        if f"{self.grid_size}" not in self.saved_grid:
            self.saved_grid[f"{self.grid_size}"] = {}
        self.saved_grid[f"{self.grid_size}"][self.selected_cell] = {
            'x': self.selected_cell_values[0],
            'y': self.selected_cell_values[1]
        }

        return frame

    def _select_cell(self, event, x, y, _, param):
        """
        Callback function for selecting a cell in the grid.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.info(f"Point selected: {(x, y)}")
            self.selected_cell = self._get_cell(x, y)

    def _get_cell(self, x, y):
        return (x // self.grid_size, y // self.grid_size)
    
    def _draw_grid(self, frame):
        """
        Draws a grid to screen with the required cell size.
        """
        h, w, _ = frame.shape
        # Vertical lines
        for x in range(0, w, self.grid_size):
            cv2.line(frame, (x, 0), (x, h), (0, 0, 0), 1)
        # Horizontal lines
        for y in range(0, h, self.grid_size):
            cv2.line(frame, (0, y), (w, y), (0, 0, 0), 1)

        return frame
    

    def _correct_position_matrix(self, middlex, middley):
        """
        Corrects ball position using camera calibration and non-linear distortion correction.
        """
        src_points = np.array([[[middlex, middley]]], dtype=np.float32)

        # Undistort the point using the camera matrix and distortion coefficients
        undistorted_points = cv2.undistortPoints(
            src_points, self.camera_matrix, self.dist, P=self.camera_matrix
        )
        corrected_x, corrected_y = undistorted_points[0][0]

        # Get the optical center
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1,2]

        # Compute the distance from the center along the X and Y axes
        dx = corrected_x - cx
        dy = corrected_y - cy

        # Radial distortion is proportional to the distance from the center
        distance = np.sqrt(dx**2 + dy**2)

        correction_factor = 1 + self.matrix_correction_factor * (distance / max(cx, cy))

        corrected_x = cx + dx * correction_factor
        corrected_y = cy + dy * correction_factor

        return int(corrected_x), int(corrected_y)


    def _correct_position_linear(self, middlex, middley):
        """
        Simply offsets the ball coordinates by a fixed amount
        """
        if not self.mirror_linear:
            corrected_x = middlex + self.x_linear
            corrected_y = middley + self.y_linear
            return int(corrected_x), int(corrected_y)
        else:
            if middlex <= self.config.output_width // 2:
                corrected_x = middlex  + self.x_linear
            else:
                corrected_x = middlex - self.x_linear
            corrected_y = middley + self.y_linear

            return int(corrected_x), int(corrected_y)
        

    def save_all_parameters(self):
        """
        Saves all of the parameters for the calibration settings to a json file.
        """
        data = {
            0: {"grid": {
                "grid_size": self.grid_size,
                #"saved_grid": {str(k): v for k, v in self.saved_grid.items()},
            }},
            1: {"matrix_correction_factor": self.matrix_correction_factor},
            2: {"scaling": {
                "x_scaling_factor" : self.x_scaling_factor,
                "y_scaling_factor" : self.y_scaling_factor,
                "mirror_scaling": self.mirror_scaling,
            }},
            3: {"linear": {
                "x_linear": self.x_linear,
                "y_linear": self.y_linear,
                "mirror_linear": self.mirror_linear,
            }},
        }
        try:
            os.makedirs(os.path.dirname(self.parameters_path), exist_ok=True)
            with open(self.parameters_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Parameters saved to {self.parameters_path}")
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")


    def load_all_parameters(self):
        """
        Loads all of the calibration settings into the state manager.
        """
        if not os.path.exists(self.parameters_path):
            logger.warning(f"{self.parameters_path} not found. Continuing with default parameters.")
            return

        try:
            with open(self.parameters_path, "r") as f:
                data = json.load(f)

            # Validate the structure of the loaded data
            if not all(key in data for key in ["0", "1", "2", "3"]):
                logger.error["Incorrect parameters format"]
                return

            self.grid_size = data["0"]["grid"]["grid_size"]
            #self.saved_grid = {eval(k): v for k, v in data["0"]["grid"]["saved_grid"].items()}
            self.matrix_correction_factor = data["1"]["matrix_correction_factor"]
            self.x_scaling_factor = data["2"]["scaling"]["x_scaling_factor"]
            self.y_scaling_factor = data["2"]["scaling"]["y_scaling_factor"]
            self.mirror_scaling = data["2"]["scaling"]["mirror_scaling"]
            self.x_linear = data["3"]["linear"]["x_linear"]
            self.y_linear = data["3"]["linear"]["y_linear"]
            self.mirror_linear = data["3"]["linear"]["mirror_linear"]
            
            logger.info("Successfully loaded all parameters.")
        
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {self.parameters_path}: {e}")
            return
        except ValueError as e:
            logger.error(f"Error validating data from {self.parameters_path}: {e}")
            return
        except Exception as e:
            logger.error(f"Unexpected error loading parameters: {e}")
            return


