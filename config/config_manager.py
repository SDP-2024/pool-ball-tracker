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
        "camera_port" : {"type": "integer", "min": 0},
        "calibrate_camera": {"type": "boolean"},
        "calibration_folder": {"type" : "string", "min" : 0},
        "font_color": {
            "type": "list",
            "schema": {"type": "integer", "min": 0, "max": 255},
            "minlength": 3,
            "maxlength": 3,
        },
        "font_scale": {"type": "float", "min": 0.1, "max": 5.0},
        "font_thickness": {"type": "integer", "min": 1, "max": 10},
        "use_networking" : {"type": "boolean"},
        "detection_model_path" : {"type": "string", "min" : 0},
        "conf_threshold": {"type": "float", "min": 0, "max": 1.0},
        "clean_images_path": {"type": "string", "min" : 0},
        "model_training_path" : {"type": "string", "min": 0},
        "anomaly_threshold": {"type": "float", "min": 0, "max": 1.0},
        "autoencoder_model_path": {"type": "string", "min" : 0},
        "network_update_interval": {"type": "float", "min": 0.01},
        "position_threshold": {"type": "integer", "min": 0},
        "poolpal_url" : {"type": "string", "min": 0},
        "poolpal_subdomain": {"type": "string", "min": 0},
        "output_width": {"type": "integer", "min": 0},
        "output_height": {"type": "integer", "min": 0},
        "hole_threshold": {"type": "integer", "min": 0},
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
                        return create_profile(yaml_path)
            
        except Exception as e:
            logger.error(f"Error reading config.yaml: {e}")
    else:
        logger.warning("config.yaml not found. Creating a new one.")
        return create_profile(yaml_path)


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


def create_profile(path="../config", name="default"):
    """
    Creates and saves a default configuration to the specified YAML file.

    Args:
        yaml_path (str): The path to the config.yaml file where the default profile will be saved.

    Returns:
        dict: The default configuration settings.
    """
    # Get the script's directory (src directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If path is default (../config), go up one level and look into the config folder
    if path == "../config":
        yaml_path = os.path.join(script_dir, path, "config.yaml")
    else:
        # If another path is provided, use it directly
        yaml_path = os.path.join(path) 

    profiles = {}

    # Load existing config if available
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as yaml_file:
                data = yaml.safe_load(yaml_file) or {}
                profiles = data.get("profiles", {})
        except Exception as e:
            logger.error(f"Error reading config.yaml: {e}")
            return None

    profiles[name] = {
            "camera_port" : 0,
            "calibrate_camera": False,
            "calibration_folder": "",
            "font_color": [255, 255, 255],
            "font_scale": 0.5,
            "font_thickness": 2,
            "use_networking" : False,
            "detection_model_path" : "/model",
            "conf_threshold": 0.5,
            "clean_images_path": "/clean_images",
            "model_training_images": "/training_images",
            "anomaly_threshold": 0.01,
            "autoencoder_model_path": "/model/autoencoder_model.keras",
            "network_update_interval": 0.1,
            "position_threshold": 10,
            "poolpal_url": "https://pool-pal.serveo.net",
            "poolpal_subdomain": "pool-pal",
            "output_width": 1200,
            "output_height": 600,
            "hole_threshold": 30,
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

