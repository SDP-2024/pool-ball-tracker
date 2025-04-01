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
    def __init__(self) -> None:

        self.schema : dict = {
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
            "anomaly_buffer_size": {"type": "integer", "min": 1, "max": 50, "default": 6},
            "autoencoder_model_path": {"type": "string", "min" : 0, "default": ""},
            "network_update_interval": {"type": "float", "min": 0.01, "default": 0.5},
            "position_threshold": {"type": "integer", "min": 0, "max": 100, "default": 3},
            "poolpal_url" : {"type": "string", "min": 0, "default": ""},
            "poolpal_subdomain": {"type": "string", "min": 0, "default": ""},
            "output_width": {"type": "integer", "min": 640, "max": 4096, "default": 1200},
            "output_height": {"type": "integer", "min": 480, "max": 2160,"default": 600},
            "hole_threshold": {"type": "integer", "min": 0, "max": 100, "default": 30},
            "calibration_mode": {"type": "integer", "min": -1, "max": 3, "default": -1},
            "ball_area" : {"type": "integer", "min": 1000, "max": 10000, "default": 3000},
            "use_hidden_balls" : {"type" : "boolean", "default" : True}
            }
        # Define member variable for all config options
        for key in self.schema.keys():
            setattr(self, key, None)

        
        self.load_config()


    def load_config(self, path : str ="../config") -> None:
        """
        Loads the configuration profile for the program.
        It will create a new config file if one does not exist at the specified path.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
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
        

    def verify_config(self, data : dict) -> bool:
        """
        Validates the loaded configuration data using the schema defined in self.schema
        """
        validator = Validator(self.schema)
        if validator.validate(data):
            logger.info("Config validated.")
            return True
        else:
            logger.error(f"Validation errors: {validator.errors}")
            return False
        

    def create_config(self) -> None:
        """
        Creates and saved a new configuration file. It also loads the default values into the config instance.
        """

        config : dict = {key: value.get('default') for key, value in self.schema.items()}
        
        self.save_config(config)
        self.set_all_values(config)


    def save_config(self, config : dict =None):
        """
        Saves the config file to the specified path.
        """
        yaml_path : str = self._get_yaml_path()
        if config is None:
            config : dict = self.get_all_values()
        try:
            # Save the profiles to the config.yaml file
            with open(yaml_path, 'w') as yaml_file:
                yaml.dump(config, yaml_file, default_flow_style=False)
            logger.info("config.yaml saved.")
            return config
        except Exception as e:
            logger.error(f"Error writing config.yaml: {e}")
            return None
        
    def set_all_values(self, data : dict) -> None:
        """
        Sets the attributes within the config class for every item in the config file.
        """
        for key in self.schema.keys():
            setattr(self, key, data[key])


    def get_all_values(self) -> dict:
        """
        Retrieves all configuration values from the class attributes.
        """
        config_values = {}
        for key in self.schema.keys():
            config_values[key] = getattr(self, key)
        return config_values

        
    def _get_yaml_path(self, path : str ="../config") -> str:
        """
        Gets the full yaml path of the configuration file.
        """
        script_dir : str = os.path.dirname(os.path.abspath(__file__))
        
        if path == "../config":
            yaml_path : str = os.path.join(script_dir, path, "config.yaml")
        else:
            # If another path is provided, use it directly
            yaml_path : str = os.path.join(path)
        return yaml_path
        

    