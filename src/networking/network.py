import requests
import time
import logging

logger = logging.getLogger(__name__)

class Network:
    def __init__(self, config, app):
        self.config = config
        self.esp_ip = config["esp_ip"]
        self.update_url = f"http://{self.esp_ip}/{config['update_endpoint']}"
        self.ready_url = f"http://{self.esp_ip}/{config['ready_endpoint']}"
        self.app = app
        self.time_between_poll = self.config["poll_interval"]
        self.time_since_last_poll = time.time() - self.time_between_poll 

    def setup(self):
        self.app.run(host="0.0.0.0", port=self.config["port"])

    def send(self, command):
        payload = {'x' : command[0], 'y': command[1]}
        try:
            response = requests.post(self.update_url, json=payload)
            if response.status_code == 200:
                logger.info(f"Coordinates sent: ({payload['x']}, {payload['y']})")
            else:
                logger.error(f"Failed to send data: {response.status_code}")

        except Exception as e:
            logger.error(f"Error {e}")
        

    def poll_ready(self):
        current_time = time.time()
        if current_time - self.time_since_last_poll >= self.time_between_poll:
            try:
                response = requests.get(self.ready_url)
                if response.status_code == 200 and response.text.strip().lower() == "true":
                    logger.info("Ready")
                    self.time_since_last_poll = time.time()
                    return True
                else:
                    logger.info("Not ready")
                    self.time_since_last_poll = time.time()
                    return False
            except Exception as e:
                logger.error(f"Error checking readiness: {e}")
                return False
            
        else:
            return False
