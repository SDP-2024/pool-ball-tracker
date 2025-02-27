import time
import logging
import serial

logger = logging.getLogger(__name__)

class Network:
    def __init__(self, config, app):
        self.config = config
        self.pin = config["arduino_pin"]
        self.arduino = serial.Serial(self.pin, 9600, timeout=1)
        self.time_between_poll = self.config["poll_interval"]
        self.time_since_last_poll = time.time() - self.time_between_poll 

    def send(self, command):
        self.arduino.write(f"{command}\n".encode())
        
    def poll_ready(self):
        current_time = time.time()
        if current_time - self.time_since_last_poll < self.time_between_poll:
            return False

        try:
            self.arduino.write("STATUS\n".encode())
            response = self.arduino.readline().decode().strip()
            if response.status_code == 200 and response.text.strip().lower() == "ready":
                logger.info("Ready")
                self.time_since_last_poll = current_time
                return True
            else:
                logger.info("Not ready")
                self.time_since_last_poll = current_time
                return False
        except Exception as e:
            logger.error(f"Error checking readiness: {e}")
            self.time_since_last_poll = current_time
            return False
