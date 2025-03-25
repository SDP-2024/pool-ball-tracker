import time
import logging
import cv2
import numpy as np
import os
import json
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class StateManager():
    """
    This class is in charge of managing the current state of the game.
    It monitors the current and previous state to detect when the balls have stopped moving.
    It also sends the current state to the network if it is available.
    """
    #selected_cell_changed = pyqtSignal(tuple)
    def __init__(self, config, network=None, mtx=None, dist=None):
        #super().__init__()
        self.config = config
        self.previous_state = None
        self.network = network
        self.time_between_updates = self.config["network_update_interval"]
        self.time_since_last_update = time.time() - self.time_between_updates
        self.end_of_turn = False
        self.not_moved_counter = 0

        # Calibration parameters
        self.calibrating = False
        self.camera_matrix = mtx
        self.dist = dist
        self.calibration_mode = self.config["calibration_mode"]

        self.grid_size = 100
        #self._selected_cell = (0,0)
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


    def update(self, data, labels, frame):
        """
        Main update function for the game state.
        It gets the current ball positions and compares it to the previous state.
        If it thinks that there has been no movement between state, then it won't send another update.
        If the balls go from moving to not moving, then it would suggest that the turn is finished.

        Args:
            data (tuple): A tuple containing all the detected objects
            labels (dict): A dictionary of the object labels
        """
        current_time = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates: return

        balls = {}

        num_balls = 0
        self.not_moved_counter = 0

        if data is None or len(data) == 0 or data[0].boxes is None:
            boxes = []
        else:
            boxes = data[0].boxes

        # If calibration enabled then draw grid
        if self.calibrating and self.calibration_mode == 0:
            frame = self._handle_grid(frame)

        # Process detected balls
        for ball in boxes:
            classname, middlex, middley = self._get_ball_info(ball, labels)

            # Ignore arm and hole
            if classname == "arm" or classname == "hole":
                continue

            num_balls += 1

            if self.calibration_mode == 0:
                middlex, middley = self._correct_ball_position_grid(middlex, middley)
            elif self.calibration_mode == 1:
                middlex, middley = self._correct_ball_position_matrix(middlex, middley)
            elif self.calibration_mode == 2:
                middlex, middley = self._correct_ball_position_scaling(middlex, middley)
            elif self.calibration_mode == 3:
                middlex, middley = self._correct_ball_position_linear(middlex, middley)

            cv2.circle(frame, (middlex, middley), 4, (0, 255, 0), -1)
            cv2.imshow("Detection", frame)
            # Check if this ball is close to a previous position
            if self.previous_state and classname in self.previous_state:
                for prev_ball in self.previous_state[classname]:
                    if self._has_moved(prev_ball, middlex, middley):
                        self.not_moved_counter += 1
                        prev_ball["x"] = middlex
                        prev_ball["y"] = middley
                        break

            if classname not in balls:
                balls[classname] = []
            balls[classname].append({"x": middlex, "y": middley})

        # Only update the state if there are new positions
        if self.not_moved_counter == num_balls:
            logger.debug("No significant ball movement detected. Skipping state update.")
            self.previous_state = balls

            # If balls stopped moving detected, end the turn. Only send once
            self._handle_end_of_turn()
            return

        # Update the socket with the new state
        self._update_and_send_balls(balls, current_time)


    def _correct_ball_position_scaling(self, middlex, middley):
        """
        Scale the coordinates by a constant scaling factor
        """
        if not self.mirror_scaling:
            corrected_x = middlex + (middlex * self.x_scaling_factor)
            corrected_y = middley + (middley * self.y_scaling_factor)

            return self._coords_clamped(corrected_x, corrected_y)
        else:
            if middlex <= self.config["output_width"] // 2:
                corrected_x = middlex + ((self.config["output_width"] // 2 - middlex) * self.x_scaling_factor)
            else:
                corrected_x = middlex - ((middlex - self.config["output_width"] // 2) * self.x_scaling_factor)
            corrected_y = middley + (middley * self.y_scaling_factor)

            return self._coords_clamped(corrected_x, corrected_y)


    def _correct_ball_position_grid(self, middlex, middley):
        """
        Corrects ball position using scaling factor based on the coordinates

        Args:
            middlex (int): X-coordinate of the detected ball.
            middley (int): Y-coordinate of the detected ball.

        Returns:
            tuple: Corrected (x, y) coordinates.
        """
        distance_from_center_x = self.config["output_width"] / 2 - middlex

        offset_400 = (((self.config["output_width"]/2) + middlex) * 0.025)
        offset_200 = (((self.config["output_width"]/2) + middlex) * 0.023)
        offset_100 = (((self.config["output_width"]/2) + middlex) * 0.017)
        
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

        return self._coords_clamped(corrected_x, corrected_y)
    

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
    
    # @property
    # def selected_cell(self):
    #     return self._selected_cell
    
    # @selected_cell.setter
    # def selected_cell(self, new_value):
    #     if self._selected_cell != new_value:
    #         self._selected_cell = new_value
    #         self.selected_cell_changed.emit(new_value)

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
    

    def _correct_ball_position_matrix(self, middlex, middley):
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

        return self._coords_clamped(corrected_x, corrected_y)


    def _correct_ball_position_linear(self, middlex, middley):
        """
        Simply offsets the ball coordinates by a fixed amount
        """
        if not self.mirror_linear:
            corrected_x = middlex + self.x_linear
            corrected_y = middley + self.y_linear
            return self._coords_clamped(corrected_x, corrected_y)
        else:
            if middlex <= self.config["output_width"] // 2:
                corrected_x = middlex  + self.x_linear
            else:
                corrected_x = middlex - self.x_linear
            corrected_y = middley + self.y_linear

            return self._coords_clamped(corrected_x, corrected_y)


    def _get_ball_info(self, ball, labels):
        """
        Gets the important info of the ball that is passed to it
        """
        xyxy_tensor = ball.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
        classidx = int(ball.cls.item())
        classname = labels[classidx]
        _middlex = int((xmin + xmax) // 2)
        _middley = int((ymin + ymax) // 2)

        # Clamp coordinates to boundaries
        middlex, middley = self._coords_clamped(_middlex, _middley)

        return classname, middlex, middley
    

    def _coords_clamped(self, _middlex, _middley):
        """
        Clamps the coordinates to the boundaries of the table
        """
        middlex = self.config["output_width"] if _middlex > self.config["output_width"] else _middlex
        middley = self.config["output_height"] if _middley > self.config["output_height"] else _middley
        middlex = 0 if middlex < 0 else middlex
        middley = 0 if middley < 0 else middley
        return int(middlex), int(middley)
    

    def _has_moved(self, prev_ball, middlex, middley):
        """
        Checks if the ball is close to a previous position
        """
        dx = abs(prev_ball["x"] - middlex)
        dy = abs(prev_ball["y"] - middley)
        return dx <= self.config["position_threshold"] and dy <= self.config["position_threshold"]
    

    def _handle_end_of_turn(self):
        """
        Sends message if the end of turn is detected
        """
        if not self.end_of_turn:
            self.end_of_turn = True
            if self.network:
                self.network.send_end_of_turn("true")
            else:
                logger.info("No movement detected, turn ended.")


    def _update_and_send_balls(self, balls, current_time):
        """
        Sends the balls if new positions are detected
        """
        if balls:
            self.previous_state = balls
            self.time_since_last_update = current_time
            self.end_of_turn = False
            if self.network:
                self.network.send_balls({"balls": balls})
            else:
                logger.info("Sending balls: %s", balls)


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
