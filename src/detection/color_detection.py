import yaml
import cv2 as cv
import numpy as np
from utils import get_limits

class ColorDetector:
    """
    A class for detecting specific colors in an image based on HSV color thresholds.

    The ColorDetector class loads predefined HSV color thresholds from a configuration file 
    and uses them to generate masks for detecting the specified color in an image.
    """
    def __init__(self, config):
        """
        Initializes the ColorDetector with configuration settings.

        Args:
            config (dict): The configuration dictionary containing detector settings, 
                           including the profile name and color thresholds.
        """
        self.config = config
        self.profile = self.config["profile"]
        self.hsv_thresholds = self._load_color_ranges(filepath="config/colors.yaml", profile=self.profile)

    def detect(self, frame, color):
        """
        Detects the specified color in the given frame using the pre-loaded HSV thresholds.

        This method generates a binary mask where the detected color is white (255) and all other colors are black (0).
        The mask is then processed using morphological operations to remove noise.

        Args:
            frame (np.ndarray): The input image/frame in which to detect the color.
            color (str): The name of the color to detect (e.g., "red", "blue", "green").

        Returns:
            np.ndarray: A binary mask where the detected color is white (255) and all other regions are black (0).
        """
        lower_hsv, upper_hsv = get_limits(color, self.hsv_thresholds)
        mask = cv.inRange(frame, lower_hsv, upper_hsv)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask


    def _load_color_ranges(self, filepath, profile="default"):
        """
        Loads the HSV color ranges from a configuration file for a given profile.

        This function reads a YAML file containing predefined color ranges and returns the 
        thresholds for a specific profile. If the profile is not found, a ValueError is raised.

        Args:
            filepath (str): The path to the configuration file containing the color ranges.
            profile (str, optional): The profile name whose color ranges are to be loaded. Defaults to "default".

        Raises:
            ValueError: If the specified profile is not found in the configuration file.

        Returns:
            dict: A dictionary containing the HSV color thresholds for the specified profile.
        """
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        if "profiles" in data and profile in data["profiles"]:
            return data["profiles"][profile]
        else:
            raise ValueError(f"Profile '{profile}' not found in {filepath}.")