import numpy as np

class Coordinate_System:
    def __init__(self, config, width, height):
        self.config = config
        self.table_width_pixels = width
        self.table_height_pixels = height
        self.table_width_mm = self.config["table_width_mm"]
        self.table_height_mm = self.config["table_height_mm"]
        self.width_mm_per_pixel = self.table_width_mm/self.table_width_pixels
        self.height_mm_per_pixel = self.table_height_mm/self.table_height_pixels
        self.stepper_degrees_per_step = self.config["stepper_degrees_per_step"]
        self.pulley_diameter = self.config["pulley_diameter"]
        self.steps_per_rev = 360.0/self.stepper_degrees_per_step
        self.linear_distance_per_step = (np.pi * self.pulley_diameter)/self.steps_per_rev

    def translate_position_to_stepper_commands(self, detected_balls):
        for ball in detected_balls:
            if ball["color"] == "white":
                x,y = ball["position"]
                x_distance_mm = x * self.width_mm_per_pixel
                x_step = x_distance_mm / self.linear_distance_per_step
                y_distance_mm = y * self.height_mm_per_pixel
                y_step = y_distance_mm / self.linear_distance_per_step

                return (x_step, y_step)