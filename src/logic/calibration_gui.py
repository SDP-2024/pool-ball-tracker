from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget, QSlider, QCheckBox
from PyQt6.QtCore import Qt
import sys
import logging

logger = logging.getLogger(__name__)

class CalibrationInterface(QWidget):
    """
    This calibration interface allows for the real-time modification of calibration settings.
    These settings produce an offset from the detected coordinates to the "real" coordinates.
    The aim is to find settings that allow the gantry to reliably move to the "real" coordinates.
    """
    def __init__(self, config, offset_manager):
        super().__init__()
        self.config = config
        self.offset_manager = offset_manager
        self.setWindowTitle("Calibration Interface")
        self.resize(300, 200)
        self.layout = QVBoxLayout()

        self.grid_label = None
        self.grid_slider = None
        self.cell_label = None
        self.cell_x_offset_slider = None
        self.cell_x_offset_label = None
        self.cell_y_offset_slider = None
        self.cell_y_offset_label = None

        self.x_scaling_slider = None
        self.x_scaling_label = None
        self.y_scaling_slider = None
        self.y_scaling_label = None
        self.scaling_mirror = None

        self.x_linear_slider = None
        self.x_linear_label = None
        self.y_linear_slider = None
        self.y_linear_label = None
        self.linear_mirror = None

        self.matrix_correction_slider = None
        self.matrix_correction_label = None
        self._setup()

    def _setup(self):
        """
        Setup the buttons to switch to the different modes, and the save button.
        """
        grid_mode_button = QPushButton("Grid Mode")
        grid_mode_button.clicked.connect(self.grid_mode)
        self.layout.addWidget(grid_mode_button)

        matrix_mode_button = QPushButton("Matrix Mode")
        matrix_mode_button.clicked.connect(self.matrix_mode)
        self.layout.addWidget(matrix_mode_button)

        scaling_mode_button = QPushButton("Scaling Mode")
        scaling_mode_button.clicked.connect(self.scaling_mode)
        self.layout.addWidget(scaling_mode_button)

        linear_mode_button = QPushButton("Linear Mode")
        linear_mode_button.clicked.connect(self.linear_mode)
        self.layout.addWidget(linear_mode_button)

        save_button = QPushButton("Save All")
        save_button.clicked.connect(self.offset_manager.save_all_parameters)
        save_button.setFixedWidth(100)
        self.layout.addWidget(save_button)
        

        self.setLayout(self.layout)
    

    def grid_mode(self):
        """
        The main function for the grid mode.
        This draws a grid to the frame, where the user can then select a cell.
        Each cell can be assigned its own offset.
        """
        self.destroy_all()
        logger.info("Grid Mode enabled.")
        self.offset_manager.calibration_mode = 0

        self.cell_label = QLabel(f"Selected Cell: {self.offset_manager.selected_cell}")
        self.layout.addWidget(self.cell_label)

        if not self.grid_slider:
            self.grid_slider = QSlider(Qt.Orientation.Horizontal)
            self.grid_slider.setRange(50, 250)
            self.grid_slider.setValue(self.offset_manager.grid_size)

            self.grid_label = QLabel(f"Grid Size: {self.grid_slider.value()}")

            self.grid_slider.valueChanged.connect(self._update_grid_size)

            if self.offset_manager.selected_cell is not None:
                self.cell_x_offset_slider = QSlider(Qt.Orientation.Horizontal)
                self.cell_x_offset_slider.setRange(-50, 50)
                self.cell_x_offset_slider.setValue(self.offset_manager.selected_cell_values[0])
                self.cell_x_offset_label = QLabel(f"X Offset: {self.cell_x_offset_slider.value()}")

                self.cell_y_offset_slider = QSlider(Qt.Orientation.Horizontal)
                self.cell_y_offset_slider.setRange(-50, 50)
                self.cell_y_offset_slider.setValue(self.offset_manager.selected_cell_values[1])
                self.cell_y_offset_label = QLabel(f"Y Offset: {self.cell_y_offset_slider.value()}")

                self.cell_x_offset_slider.valueChanged.connect(self._update_grid_x_offset)
                self.cell_y_offset_slider.valueChanged.connect(self._update_grid_y_offset)
            


            self.layout.addWidget(self.grid_label)
            self.layout.addWidget(self.grid_slider)
            self.layout.addWidget(self.cell_x_offset_label)
            self.layout.addWidget(self.cell_x_offset_slider)
            self.layout.addWidget(self.cell_y_offset_label)
            self.layout.addWidget(self.cell_y_offset_slider)

    def update_cell_info(self):
        """Update the cell label and offset sliders when a new cell is selected"""
        if not hasattr(self.offset_manager, 'selected_cell'):
            return

        # Update cell label
        if self.cell_label:
            self.cell_label.setText(f"Selected Cell: {self.offset_manager.selected_cell}")

        # Get the saved offsets for this cell and grid size
        grid_size = str(self.offset_manager.grid_size)
        cell = self.offset_manager.selected_cell
        
        # Default to (0, 0) if no offsets exist
        x_offset = 0
        y_offset = 0
        
        # Check if offsets exist in saved grid
        if (grid_size in self.offset_manager.saved_grid and 
            cell in self.offset_manager.saved_grid[grid_size]):
            saved_values = self.offset_manager.saved_grid[grid_size][cell]
            x_offset = saved_values.get('x', 0)
            y_offset = saved_values.get('y', 0)
        
        # Update the offset manager's current values
        self.offset_manager.selected_cell_values = (x_offset, y_offset)
        
        # Update sliders if they exist
        if self.cell_x_offset_slider:
            self.cell_x_offset_slider.setValue(x_offset)
        if self.cell_y_offset_slider:
            self.cell_y_offset_slider.setValue(y_offset)
        
        # Update labels if they exist
        if self.cell_x_offset_label:
            self.cell_x_offset_label.setText(f"X Offset: {x_offset}")
        if self.cell_y_offset_label:
            self.cell_y_offset_label.setText(f"Y Offset: {y_offset}")



    
    def _update_grid_size(self, value):
        """
        Updates the state with the grid size.
        """
        if self.grid_label:
            self.grid_label.setText(f"Grid Size: {value}")
        self.offset_manager.grid_size = value


    def _update_grid_x_offset(self, value):
        """
        Updates the offset of the cell on the x-axis
        """
        if self.cell_x_offset_label:
            self.cell_x_offset_label.setText(f"X Offset: {value}")
        
        self.offset_manager.selected_cell_values = (value, self.offset_manager.selected_cell_values[1])
        grid_size = str(self.offset_manager.grid_size)
        
        # Initialize grid size entry if needed
        if grid_size not in self.offset_manager.saved_grid:
            self.offset_manager.saved_grid[grid_size] = {}

        # Only store if not default (0,0) value
        if value != 0 or self.offset_manager.selected_cell_values[1] != 0:
            self.offset_manager.saved_grid[grid_size][self.offset_manager.selected_cell] = {
                'x': value,
                'y': self.offset_manager.selected_cell_values[1]
            }
        else:
            if self.offset_manager.selected_cell in self.offset_manager.saved_grid[grid_size]:
                del self.offset_manager.saved_grid[grid_size][self.offset_manager.selected_cell]


    def _update_grid_y_offset(self, value):
        """
        Update the offset of the cell on the y-axis
        """
        if self.cell_y_offset_label:
            self.cell_y_offset_label.setText(f"Y Offset: {value}")
        
        self.offset_manager.selected_cell_values = (self.offset_manager.selected_cell_values[0], value)
        grid_size = str(self.offset_manager.grid_size)
        
        # Initialize grid size entry if needed
        if grid_size not in self.offset_manager.saved_grid:
            self.offset_manager.saved_grid[grid_size] = {}
        
        # Only store if not default (0,0) value
        if self.offset_manager.selected_cell_values[0] != 0 or value != 0:
            self.offset_manager.saved_grid[grid_size][self.offset_manager.selected_cell] = {
                'x': self.offset_manager.selected_cell_values[0],
                'y': value
            }
        else:
            if self.offset_manager.selected_cell in self.offset_manager.saved_grid[grid_size]:
                del self.offset_manager.saved_grid[grid_size][self.offset_manager.selected_cell]


    def matrix_mode(self):
        """
        This mode uses the camera calibration matrix and distortion coefficients.
        It aims to follow the radial distortion of the camera.
        """
        self.destroy_all()
        logger.info("Matrix Mode enabled.")
        self.offset_manager.calibration_mode = 1
        if not self.matrix_correction_slider:
            self.matrix_correction_slider = QSlider(Qt.Orientation.Horizontal)
            self.matrix_correction_slider.setRange(-1000, 1000)
            self.matrix_correction_slider.setValue(int(self.offset_manager.matrix_correction_factor * 1000))

            self.matrix_correction_label = QLabel(f"Matrix Correction Factor: {self.matrix_correction_slider.value()/1000}")

            self.matrix_correction_slider.valueChanged.connect(self._update_matrix_correction_factor)

            self.layout.addWidget(self.matrix_correction_label)
            self.layout.addWidget(self.matrix_correction_slider)


    def _update_matrix_correction_factor(self, value):
        """
        Updates the correction factor.
        """
        value = value / 1000
        if self.matrix_correction_label:
            self.matrix_correction_label.setText(f"Matrix Correction Factor: {value}")
        self.offset_manager.matrix_correction_factor = value


    def scaling_mode(self):
        """
        This mode allows the offset to be scaled according to the position on the frame.
        A detection further from the origin will result in a larger offset.
        This can be set to mirror around the middle.
        Points further from the middle will have a greater offset.
        """
        self.destroy_all()
        logger.info("Scaling Mode enabled.")
        self.offset_manager.calibration_mode = 2
        if not self.x_scaling_slider and not self.y_scaling_slider:
            self.scaling_mirror = QCheckBox("Mirror around center")
            self.scaling_mirror.stateChanged.connect(self._update_scaling_mirror)
            self.scaling_mirror.setChecked(self.offset_manager.mirror_scaling)
            self.x_scaling_slider = QSlider(Qt.Orientation.Horizontal)
            self.x_scaling_slider.setRange(-200, 200)
            self.x_scaling_slider.setValue(int(self.offset_manager.x_scaling_factor*1000))

            self.y_scaling_slider = QSlider(Qt.Orientation.Horizontal)
            self.y_scaling_slider.setRange(-200, 200)
            self.y_scaling_slider.setValue(int(self.offset_manager.y_scaling_factor*1000))

            self.x_scaling_label = QLabel(f"X Scaling Factor: {self.x_scaling_slider.value()/1000}")
            self.y_scaling_label = QLabel(f"Y Scaling Factor: {self.y_scaling_slider.value()/1000}")

            self.x_scaling_slider.valueChanged.connect(self._update_scaling_value_x)
            self.y_scaling_slider.valueChanged.connect(self._update_scaling_value_y)

            self.layout.addWidget(self.x_scaling_label)
            self.layout.addWidget(self.x_scaling_slider)
            self.layout.addWidget(self.y_scaling_label)
            self.layout.addWidget(self.y_scaling_slider)
            self.layout.addWidget(self.scaling_mirror)


    def _update_scaling_value_x(self, value):
        """
        Update scaling factor in x axis.
        """
        value = value / 1000
        if self.x_scaling_label:
            self.x_scaling_label.setText(f"X Scaling Factor: {value}")
        self.offset_manager.x_scaling_factor = value


    def _update_scaling_value_y(self, value):
        """
        Update scaling factor in y axis.
        """
        value = value / 1000
        if self.y_scaling_label:
            self.y_scaling_label.setText(f"Y Scaling Factor: {value}")
        self.offset_manager.y_scaling_factor = value


    def _update_scaling_mirror(self, state):
        """
        Enable or disable mirroring around middle.
        """
        if state == 2:
            self.offset_manager.mirror_scaling = True
        else:
            self.offset_manager.mirror_scaling = False


    def linear_mode(self):
        """
        This mode simply applies a linear, and constant offset to each detection.
        This can also be made to mirror around the middle.
        """
        self.destroy_all()
        logger.info("Linear Mode enabled.")
        self.offset_manager.calibration_mode = 3
        if not self.x_linear_slider and not self.y_linear_slider:
            self.linear_mirror = QCheckBox("Mirror around center")
            self.linear_mirror.stateChanged.connect(self._update_linear_mirror)
            self.linear_mirror.setChecked(self.offset_manager.mirror_linear)
            self.x_linear_slider = QSlider(Qt.Orientation.Horizontal)
            self.x_linear_slider.setRange(-50, 50)
            self.x_linear_slider.setValue(int(self.offset_manager.x_linear))

            self.y_linear_slider = QSlider(Qt.Orientation.Horizontal)
            self.y_linear_slider.setRange(-50, 50)
            self.y_linear_slider.setValue(int(self.offset_manager.y_linear))

            self.x_linear_label = QLabel(f"X Offset Factor: {self.x_linear_slider.value()}")
            self.y_linear_label = QLabel(f"Y Offset Factor: {self.y_linear_slider.value()}")

            self.x_linear_slider.valueChanged.connect(self._update_linear_value_x)
            self.y_linear_slider.valueChanged.connect(self._update_linear_value_y)

            self.layout.addWidget(self.x_linear_label)
            self.layout.addWidget(self.x_linear_slider)
            self.layout.addWidget(self.y_linear_label)
            self.layout.addWidget(self.y_linear_slider)
            self.layout.addWidget(self.linear_mirror)


    def _update_linear_value_x(self, value):
        """
        Update offset on x axis.
        """
        if self.x_linear_label:
            self.x_linear_label.setText(f"X Offset: {value}")
        self.offset_manager.x_linear = value


    def _update_linear_value_y(self, value):
        """
        Update offset on y axis.
        """
        if self.y_linear_label:
            self.y_linear_label.setText(f"Y Offset: {value}")
        self.offset_manager.y_linear = value


    def _update_linear_mirror(self, state):
        """
        Enable or disable mirroring around the middle.
        """
        if state == 2:
            self.offset_manager.mirror_linear = True
        else:
            self.offset_manager.mirror_linear = False


    def destroy_all(self):
        """
        Remove widgets from the layout and delete them.
        """
    
        if self.grid_label:
            self.layout.removeWidget(self.grid_label)
            self.grid_label.deleteLater()
        if self.grid_slider:
            self.layout.removeWidget(self.grid_slider)
            self.grid_slider.deleteLater()
        if self.cell_label:
            self.layout.removeWidget(self.cell_label)
            self.cell_label.deleteLater()
        if self.cell_x_offset_slider:
            self.layout.removeWidget(self.cell_x_offset_slider)
            self.cell_x_offset_slider.deleteLater()
        if self.cell_x_offset_label:
            self.layout.removeWidget(self.cell_x_offset_label)
            self.cell_x_offset_label.deleteLater()
        if self.cell_y_offset_slider:
            self.layout.removeWidget(self.cell_y_offset_slider)
            self.cell_y_offset_slider.deleteLater()
        if self.cell_y_offset_label:
            self.layout.removeWidget(self.cell_y_offset_label)
            self.cell_y_offset_label.deleteLater()
        
        if self.x_scaling_slider:
            self.layout.removeWidget(self.x_scaling_slider)
            self.x_scaling_slider.deleteLater()
        if self.x_scaling_label:
            self.layout.removeWidget(self.x_scaling_label)
            self.x_scaling_label.deleteLater()
        if self.y_scaling_slider:
            self.layout.removeWidget(self.y_scaling_slider)
            self.y_scaling_slider.deleteLater()
        if self.y_scaling_label:
            self.layout.removeWidget(self.y_scaling_label)
            self.y_scaling_label.deleteLater()
        if self.scaling_mirror:
            self.layout.removeWidget(self.scaling_mirror)
            self.scaling_mirror.deleteLater()

        if self.x_linear_slider:
            self.layout.removeWidget(self.x_linear_slider)
            self.x_linear_slider.deleteLater()
        if self.x_linear_label:
            self.layout.removeWidget(self.x_linear_label)
            self.x_linear_label.deleteLater()
        if self.y_linear_slider:
            self.layout.removeWidget(self.y_linear_slider)
            self.y_linear_slider.deleteLater()
        if self.y_linear_label:
            self.layout.removeWidget(self.y_linear_label)
            self.y_linear_label.deleteLater()
        if self.linear_mirror:
            self.layout.removeWidget(self.linear_mirror)
            self.linear_mirror.deleteLater()

        if self.matrix_correction_slider:
            self.layout.removeWidget(self.matrix_correction_slider)
            self.matrix_correction_slider.deleteLater()
        if self.matrix_correction_label:
            self.layout.removeWidget(self.matrix_correction_label)
            self.matrix_correction_label.deleteLater()

        # Set references to None
        self.grid_label = None
        self.grid_slider = None
        self.cell_label = None
        self.cell_x_offset_slider = None
        self.cell_x_offset_label = None
        self.cell_y_offset_slider = None
        self.cell_y_offset_label = None
        self.x_scaling_slider = None
        self.x_scaling_label = None
        self.y_scaling_slider = None
        self.y_scaling_label = None
        self.scaling_mirror = None
        self.x_linear_slider = None
        self.x_linear_label = None
        self.y_linear_slider = None
        self.y_linear_label = None
        self.linear_mirror = None
        self.matrix_correction_slider = None
        self.matrix_correction_label = None


def run_calibration_interface(config, offset_manager):
    """
    Entry point for calibration interface.
    This is run on its own thread.
    """
    app = QApplication(sys.argv)
    window = CalibrationInterface(config, offset_manager)
    offset_manager.set_gui_reference(window)
    window.show()
    app.exec()

        