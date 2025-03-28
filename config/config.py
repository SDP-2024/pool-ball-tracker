import os
import logging
import yaml
from cerberus import Validator

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    This class is for the configuration variables.
    This is intended to allow all variables to be modified in real-time.
    This will aid in the grid-mode calibration functionality
    """
    def __init__(self):

        self.schema = {
            "camera_port" : {"type": "integer", "min": 0,"default": 0},
            "calibrate_camera": {"type": "boolean", "default": False},
            "calibration_folder": {"type" : "string", "min" : 0, "default": ""},
            "font_color": {
                "type": "list",
                "schema": {"type": "integer", "min": 0, "max": 255, "default": 255},
                "minlength": 3,
                "maxlength": 3,
            },
            "font_scale": {"type": "float", "min": 0.1, "max": 5.0, "default": 1},
            "font_thickness": {"type": "integer", "min": 1, "max": 10, "default": 1},
            "use_networking" : {"type": "boolean", "default": False},
            "detection_model_path" : {"type": "string", "min" : 0, "default": "/model/model.pt"},
            "conf_threshold": {"type": "float", "min": 0, "max": 1.0, "default": 0.5},
            "clean_images_path": {"type": "string", "min" : 0, "default": ""},
            "model_training_path" : {"type": "string", "min": 0, "default": ""},
            "anomaly_threshold": {"type": "float", "min": 0, "max": 1.0, "default": 0.1},
            "autoencoder_model_path": {"type": "string", "min" : 0, "default": ""},
            "network_update_interval": {"type": "float", "min": 0.01, "default": 0.5},
            "position_threshold": {"type": "integer", "min": 0, "max": 100, "default": 3},
            "poolpal_url" : {"type": "string", "min": 0, "default": ""},
            "poolpal_subdomain": {"type": "string", "min": 0, "default": ""},
            "output_width": {"type": "integer", "min": 640, "max": 4096, "default": 1200},
            "output_height": {"type": "integer", "min": 480, "max": 2160,"default": 600},
            "hole_threshold": {"type": "integer", "min": 0, "max": 100, "default": 30},
            "calibration_mode": {"type": "integer", "min": -1, "max": 3, "default": -1}
            }
        # Define member variable for all config options
        for key in self.schema.keys():
            setattr(self, key, None)

        
        self.load_config()


    def load_config(self, path="../config"):
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
                    if self.verify_config(data):
                        return self.set_all_values(data)
                
            except Exception as e:
                logger.error(f"Error reading config.yaml: {e}")
        else:
            logger.warning("config.yaml not found. Creating a new one.")
            self.create_config()
        

    def verify_config(self, data):
        """
        Validates the loaded configuration data using a predefined schema.

        Args:
            data (dict): The configuration data to validate.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        validator = Validator(self.schema)
        if validator.validate(data):
            logger.info("Config validated.")
            return True
        else:
            logger.error(f"Validation errors: {validator.errors}")
            return False
        

    def create_config(self):
        """
        Creates and saves a default configuration to the specified YAML file.
        Returns:
            dict: The default configuration settings.
        """

        config = {key: value.get('default') for key, value in self.schema.items()}
        
        self.save_config(config)
        self.set_all_values(config)


    def save_config(self, config=None):
        yaml_path = self._get_yaml_path()
        if config is None:
            config = self.get_all_values()
        try:
            # Save the profiles to the config.yaml file
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)
            logger.info("config.yaml saved.")
            return config
        except Exception as e:
            logger.error(f"Error writing config.yaml: {e}")
            return None
        
    def set_all_values(self, data):
        for key in self.schema.keys():
            setattr(self, key, data[key])


    def get_all_values(self):
        """
        Retrieves all configuration values from the class attributes.

        Returns:
            dict: A dictionary containing all configuration values.
        """
        config_values = {}
        for key in self.schema.keys():
            config_values[key] = getattr(self, key)
        return config_values

        
    def _get_yaml_path(self, path="../config"):
        # Get the script's directory (src directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If path is default (../config), go up one level and look into the config folder
        if path == "../config":
            yaml_path = os.path.join(script_dir, path, "config.yaml")
        else:
            # If another path is provided, use it directly
            yaml_path = os.path.join(path)
        return yaml_path
        

    