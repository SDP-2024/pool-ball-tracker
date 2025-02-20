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
        self.pulley_diameter = self.config["pulley_diameter_mm"]
        self.steps_per_rev = 360.0/self.stepper_degrees_per_step
        self.linear_distance_per_step = (np.pi * self.pulley_diameter)/self.steps_per_rev

    def translate_position_to_stepper_commands(self, detected_balls, labels):
        boxes = detected_balls[0].boxes
        for ball in boxes:
            xyxy_tensor = ball.xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
            classidx = int(ball.cls.item())
            classname = labels[classidx]
            middlex = (xmin + xmax) / 2
            middley = (ymin + ymax) / 2
            if classname == "white":
                x_distance_mm = middlex * self.width_mm_per_pixel
                x_step = x_distance_mm / self.linear_distance_per_step
                y_distance_mm = middley * self.height_mm_per_pixel
                y_step = y_distance_mm / self.linear_distance_per_step

                return (x_step, y_step)