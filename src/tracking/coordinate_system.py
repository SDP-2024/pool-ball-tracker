import numpy as np

class Coordinate_System:
    def __init__(self, config, width, height):
        """
        Initialize the Coordinate_System class.

        Args:
            config (dict): Configuration dictionary containing system parameters.
            width (int): Width of the table in pixels.
            height (int): Height of the table in pixels.
        """
        self.config = config
        self.table_width_pixels = width
        self.table_height_pixels = height
        self.table_width_mm = self.config["table_width_mm"]
        self.table_height_mm = self.config["table_height_mm"]
        self.width_mm_per_pixel = self.table_width_mm/self.table_width_pixels
        self.height_mm_per_pixel = self.table_height_mm/self.table_height_pixels
        self.stepper_degrees_per_step = self.config["stepper_degrees_per_step"]
        self.pulley_diameter = self.config["pulley_diameter_mm"]
        self.steps_per_rev = 360.0/self.stepper_degrees_per_step
        self.linear_distance_per_step = (np.pi * self.pulley_diameter)/self.steps_per_rev

    def translate_position_to_stepper_commands(self, detected_balls, labels):
        """
        Translate detected ball positions to stepper motor commands.

        Args:
            detected_balls (list): List of detected ball objects with bounding boxes.
            labels (list): List of class labels corresponding to detected objects.

        Returns:
            tuple: A tuple containing the x and y stepper motor commands.
        """
        for ball in detected_balls[0].boxes:
            if ball.conf.item() < self.config["conf_threshold"]:
                continue

            xyxy = ball.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classname = labels[int(ball.cls.item())]

            if classname == "white":
                middlex = (xmin + xmax) / 2
                middley = (ymin + ymax) / 2

                x_distance_mm = middlex * self.width_mm_per_pixel
                y_distance_mm = middley * self.height_mm_per_pixel

                x_step = x_distance_mm / self.linear_distance_per_step
                y_step = y_distance_mm / self.linear_distance_per_step

                return x_step, y_step