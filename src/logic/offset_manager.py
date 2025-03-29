import cv2
import numpy as np
import logging
import os
import json

logger = logging.getLogger(__name__)

class OffsetManager:
    """
    This class handles the offset of the ball with the purpose of translating the detected coordinates to real-world coordinates.
    """
    def __init__(self, config, mtx : np.ndarray | None, dist : np.ndarray | None):
        self.config = config
        # Calibration parameters
        self.calibrating = False
        self.camera_matrix = mtx
        self.dist = dist
        self.calibration_mode = self.config.calibration_mode

        self.grid_size : int = 100
        self.selected_cell : tuple[int, int] = (0,0)
        self.selected_cell_values : tuple[int, int] = (0,0)
        self.saved_grid : dict = {} # Saved grids indexed by the grid size

        self.x_scaling_factor : float = 0
        self.y_scaling_factor : float = 0
        self.mirror_scaling : bool = False

        self.x_linear : int = 0
        self.y_linear : int = 0
        self.mirror_linear : bool = False

        self.matrix_correction_factor : float = 0

        self.parameters_path : str = "./src/logic/calibration_parameters.json"
        self.load_all_parameters()


    def update(self, frame : cv2.Mat, middlex : int, middley : int) -> tuple[int, int]:
        """
        This is the main update function for the ball. This sets the offset based on the calibration mode.
        Grid mode: A different offset per cell in the grid of the table. Grid size can be changed.
        Matrix mode: Uses the calculated camera matrix to adjust the offset based on distortion coefficients.
        Scaling mode: Adjusts the offset based on the x and y coordinates. Can be mirrored around the camera.
        Linear mode: A constant offset on the x and y regardless of the position on the table.
        """
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
    

    def _correct_position_scaling(self, middlex : int, middley : int) -> tuple[int, int]:
        """
        Scale the coordinates by a constant scaling factor
        """
        if not self.mirror_scaling:
            corrected_x : int = middlex + (middlex * self.x_scaling_factor)
            corrected_y : int = middley + (middley * self.y_scaling_factor)

            return int(corrected_x), int(corrected_y)
        else:
            if middlex <= self.config.output_width // 2:
                corrected_x : int = middlex + ((self.config.output_width // 2 - middlex) * self.x_scaling_factor)
            else:
                corrected_x : int = middlex - ((middlex - self.config.output_width // 2) * self.x_scaling_factor)
            corrected_y : int = middley + (middley * self.y_scaling_factor)

            return int(corrected_x), int(corrected_y)


    def _correct_position_grid(self, middlex : int, middley : int) -> tuple[int, int]:
        """
        Corrects ball position using an offset for a grid cell.
        """
        cell : tuple[int, int] = self._get_cell(middlex, middley)
        grid : dict = self.saved_grid.get(str(self.grid_size), {})
        
        # Convert cell to string format if that's how it's stored
        cell_key = str(cell) if any(isinstance(k, str) for k in grid.keys()) else cell
        
        if cell_key in grid:
            offsets = grid[cell_key]
            corrected_x : int = middlex + offsets['x']
            corrected_y : int = middley + offsets['y']
        else:
            corrected_x : int = middlex
            corrected_y : int = middley
        
        return int(corrected_x), int(corrected_y)


    def _handle_grid(self, frame : cv2.Mat) -> cv2.Mat:
        """
        This function handles the grid.
        It detects which cell is selected, and works with the calibration tool to track offsets per cell.
        """
        cv2.setMouseCallback("Detection", self._select_cell)
        frame : cv2.Mat = self._draw_grid(frame)
        frame : cv2.Mat = self._highlight_edited_cells(frame)

        if self.selected_cell is not None:
            top_left, bottom_right = self._get_cell_boundaries(self.selected_cell)
            frame : cv2.Mat = cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)
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
    
    
    def _highlight_edited_cells(self, frame : cv2.Mat) -> cv2.Mat:
        """
        This helper function highlights the cells on the grid that have offsets set.
        This makes it easier to tell what cells still need to be calibrated.
        """
        current_grid : dict = self.saved_grid.get(str(self.grid_size), {})
        
        for cell, offsets in current_grid.items():
            if offsets["x"] != 0 or offsets["y"] != 0:
                top_left, bottom_right = self._get_cell_boundaries(cell)
                frame : cv2.Mat = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        return frame
    

    def _get_cell_boundaries(self, cell : tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        This gets the boundaries of a cell for drawing it to the screen.
        """
        top_left : tuple[int, int] = (cell[0] * self.grid_size, cell[1] * self.grid_size)
        bottom_right : tuple[int, int] = (top_left[0] + self.grid_size, top_left[1] + self.grid_size)
        return top_left, bottom_right


    def _select_cell(self, event, x, y, _, param) -> None:
        """
        Handles the current cell selected, loads the offset if one has been set.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            new_cell : tuple[int, int] = self._get_cell(x, y)
            logger.info(f"Selected cell: {new_cell}")
            if new_cell != self.selected_cell:
                self.selected_cell = new_cell
                
                # Get current grid size
                current_grid : dict = self.saved_grid.get(self.grid_size, {})
                
                # Load offsets for this cell if they exist
                if new_cell in current_grid:
                    self.selected_cell_values = (
                        current_grid[new_cell]['x'],
                        current_grid[new_cell]['y']
                    )
                else:
                    self.selected_cell_values = (0, 0)
                
                # Update GUI if available
                if hasattr(self, 'gui') and self.gui:
                    self.gui.update_cell_info()


    def _get_cell(self, x : int, y : int) -> tuple[int, int]:
        """
        Helper function to get the current cell on the grid.
        """
        return (x // self.grid_size, y // self.grid_size)
    

    def _draw_grid(self, frame : cv2.Mat) -> cv2.Mat:
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
    

    def _correct_position_matrix(self, middlex : int, middley : int) -> tuple[int, int]:
        """
        Corrects ball position using camera calibration and non-linear distortion correction.
        """
        src_points : np.ndarray = np.array([[[middlex, middley]]], dtype=np.float32)

        # Undistort the point using the camera matrix and distortion coefficients
        undistorted_points : cv2.Mat = cv2.undistortPoints(
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


    def _correct_position_linear(self, middlex : int, middley : int) -> tuple[int, int]:
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
        

    def save_all_parameters(self) -> None:
        """
        Saves all of the parameters for the calibration settings to a json file.
        """

        serializable_grid : dict = {
            str(grid_size): {
                str(cell): offsets 
                for cell, offsets in cells.items()
                if offsets['x'] != 0 or offsets['y'] != 0  # Only include non-zero offsets
            }
            for grid_size, cells in self.saved_grid.items()
            if cells
        }

        data : dict = {
            0: {"grid": {
                "grid_size": self.grid_size,
                "saved_grid": serializable_grid,
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


    def load_all_parameters(self) -> None:
        """
        Loads all of the calibration settings into the state manager.
        """
        if not os.path.exists(self.parameters_path):
            logger.warning(f"{self.parameters_path} not found. Continuing with default parameters.")
            return

        try:
            with open(self.parameters_path, "r") as f:
                data = json.load(f)

            # Validate basic structure
            if not all(str(k) in data for k in range(4)):
                logger.error("Invalid parameters format")
                return

            # Load grid parameters
            grid_data : dict = data["0"]["grid"]
            self.grid_size : dict = grid_data["grid_size"]
            
            # Initialize saved_grid
            self.saved_grid : dict = {}
            
            if "saved_grid" in grid_data:
                for grid_size_str, cells in grid_data["saved_grid"].items():
                    try:
                        grid_size = grid_size_str
                        self.saved_grid[grid_size] = {}
                        
                        for cell_str, offsets in cells.items():
                            # Parse cell coordinates from string "(x,y)"
                            cell_str = cell_str.strip("()")
                            x, y = map(int, cell_str.split(','))
                            cell = (x, y)
                            
                            # Store the offsets
                            self.saved_grid[grid_size][cell] = {
                                'x': int(offsets['x']),
                                'y': int(offsets['y'])
                            }
                            logger.debug(f"Loaded cell {cell} with offsets {offsets}")
                    except Exception as e:
                        logger.error(f"Error loading grid {grid_size_str}: {e}")
                        continue
           
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


    def set_gui_reference(self, gui):
        """
        Set a reference to the GUI for callbacks
        """
        self.gui = gui


