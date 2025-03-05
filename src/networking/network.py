import socketio
import time
import threading
import logging

logger = logging.getLogger(__name__)

# TODO: If cannot connect, then keep trying until connection is made (to prevent needing to restart program if connection lost)
class Network:
    def __init__(self, config):
        self.config = config
        self.sio = socketio.Client()

        @self.sio.event
        def connect():
            logger.info("Connected to server.")
            self.sio.emit("join", "ballPositions")
            self.sio.emit("join", "obstructionDetected")

    def connect(self):
        try:
            self.sio.connect(self.config["poolpal_url"], wait=False)
        except Exception as e:
            logger.error(f"Connection failed: {e}")

    def send_balls(self, balls):
        try:
            logger.info("Sending balls: %s", balls)
            self.sio.emit("ballPositions", balls)
        except Exception as e:
            logger.error("Failed to send ballPositions")
            pass

    def send_obstruction(self, obstruction_detected):
        try:
            self.sio.emit("obstructionDetected", obstruction_detected)
        except Exception as e:
            logger.error("Failed to send obstructionDetected")
            pass

    def disconnect(self):
        self.sio.disconnect()

    def start(self):
        threading.Thread(target=self.connect, daemon=True).start()
