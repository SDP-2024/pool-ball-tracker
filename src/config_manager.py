import os
import yaml
from cerberus import Validator

import logging

logger = logging.getLogger(__name__)

# Schema definition
schema = {
    "profiles": {
    "type": "dict",
    "allow_unknown": True,
    "schema": {  # Define the structure for each profile
        "camera_port_1" : {"type": "integer", "min": 0},
        "camera_port_2" : {"type": "integer", "min": -1},
        "calibrate_cameras": {"type": "boolean"},
        "calibration_folders": {
            "type" : "list", 
            "schema": {"type" : "string", "min" : 0}
            },
        "time_between_stitch" : {"type" : "float", "min": 0.01},
        "camera_width" : {"type": "integer", "min": 0},
        "camera_height" : {"type": "integer", "min": 0},
        "output_width" : {"type": "integer", "min" : 0},
        "output_height" : {"type": "integer", "min": 0},
        "detector": {
            "type": "dict",
            "schema": {
                "profile": {"type": "string"},
                "radius": {"type": "integer", "min": 1, "max": 1000},
                "table_area": {"type" : "integer", "min": 1},
            },
        },
        "tracking": {
            "type": "dict",
            "schema": {
                "buffer_size": {"type": "integer", "min": 1, "max": 1000},
                "max_distance": {"type": "integer", "min": 1, "max": 1000},
                "min_area": {"type": "integer", "min": 1, "max": 100000},
                "circle_outline_color": {
                    "type": "list",
                    "schema": {"type": "integer", "min": 0, "max": 255},
                    "minlength": 3,
                    "maxlength": 3,
                },
                "circle_thickness": {"type": "integer", "min": 1, "max": 10},
                "radius_line_color": {
                    "type": "list",
                    "schema": {"type": "integer", "min": 0, "max": 255},
                    "minlength": 3,
                    "maxlength": 3,
                },
                "radius_line_thickness": {"type": "integer", "min": 1, "max": 10},
                "center_point_color": {
                    "type": "list",
                    "schema": {"type": "integer", "min": 0, "max": 255},
                    "minlength": 3,
                    "maxlength": 3,
                },
                "center_point_radius": {"type": "integer", "min": 1, "max": 50},
                "font_color": {
                    "type": "list",
                    "schema": {"type": "integer", "min": 0, "max": 255},
                    "minlength": 3,
                    "maxlength": 3,
                },
                "font_scale": {"type": "float", "min": 0.1, "max": 5.0},
                "font_thickness": {"type": "integer", "min": 1, "max": 10},
                "tracking_line_color": {
                    "type": "list",
                    "schema": {"type": "integer", "min": 0, "max": 255},
                    "minlength": 3,
                    "maxlength": 3,
                },
            },
        },
    },
}

}


def load_config(profile, path="../config"):
    """
    Loads the configuration for the specified profile from a YAML configuration file.

    This function reads the configuration file, validates it, and returns the settings
    for the specified profile. If the profile is not found, it falls back to the default
    profile or creates a new default configuration if the file is missing.

    Args:
        profile (str): The name of the profile to load from the configuration file.
        path (str, optional): The path to the directory containing the configuration file.
                              Defaults to "../config", which looks for the config.yaml
                              file in the parent directory.

    Returns:
        dict: The configuration settings for the specified profile, or the default profile 
              if the specified profile is not found.
    """

    # Get the script's directory (src directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If path is default (../config), go up one level and look into the config folder
    if path == "../config":
        yaml_path = os.path.join(script_dir, path, "config.yaml")
    else:
        # If another path is provided, use it directly
        yaml_path = os.path.join(path) 
    
    # Check if config.yaml exists
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, 'r') as yaml_file:
                data = yaml.safe_load(yaml_file)
                if verify_config(data):
                    if "profiles" in data and profile in data["profiles"]:
                        logger.info(f"Got config for profile: {profile}")
                        return data["profiles"][profile]
                    elif "profiles" in data and "default" in data["profiles"]:
                        logger.warning("Profile not found. Using default profile.")
                        return data["profiles"]["default"]
                    else:
                        logger.warning("No profiles found in config.yaml. Using default profile.")
                        return create_default_config(yaml_path)
            
        except Exception as e:
            logger.error(f"Error reading config.yaml: {e}")
    else:
        logger.warning("config.yaml not found. Creating a new one.")
        return create_default_config(yaml_path)


def verify_config(data):
    """
    Validates the loaded configuration data using a predefined schema.

    Args:
        data (dict): The configuration data to validate.

    Returns:
        bool: True if the configuration is valid, False otherwise.
    """
    validator = Validator(schema)
    if validator.validate(data):
        logger.info("Config validated.")
        return True
    else:
        logger.error(f"Validation errors: {validator.errors}")
        return False


def create_default_config(yaml_path):
    """
    Creates and saves a default configuration to the specified YAML file.

    Args:
        yaml_path (str): The path to the config.yaml file where the default profile will be saved.

    Returns:
        dict: The default configuration settings.
    """
    profiles = {
        "default": {
            "camera_port_1" : 0,
            "camera_port_2" : 0,
            "calibrate_cameras": False,
            "calibration_folders": [],
            "time_between_stitch": 1.0,
            "camera_width": 1280,
            "camera_height": 720,
            "output_width" : 1920,
            "output_height": 1080,
            "detector": {
                "profile": "default",
                "radius": 50,
                "table_area" : 1000,
            },
            "tracking": {
                "buffer_size": 64,
                "max_distance": 50,
                "min_area": 1000,
                "circle_outline_color": [0, 255, 255],
                "circle_thickness": 2,
                "radius_line_color": [255, 0, 0],
                "radius_line_thickness": 2,
                "center_point_color": [0, 0, 255],
                "center_point_radius": 5,
                "font_color": [255, 255, 255],
                "font_scale": 0.5,
                "font_thickness": 2,
                "tracking_line_color": [0, 0, 255],
            },
        }
    }
    try:
        # Save the profiles to the config.yaml file
        with open(yaml_path, 'w') as yaml_file:
            yaml.dump({"profiles": profiles}, yaml_file, default_flow_style=False)
        logger.info("config.yaml created with default profile.")
        return profiles["default"]
    except Exception as e:
        logger.error(f"Error writing config.yaml: {e}")
        return None

