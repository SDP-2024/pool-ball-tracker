import socketio
import time
import threading
import logging

logger = logging.getLogger(__name__)

class Network:
    def __init__(self, config):
        self.config = config
        self.sio = socketio.Client()
        self.disconnect_counter = 0

        @self.sio.event
        def connect():
            logger.info("Connected to server.")
            self.sio.emit("join", "ballPositions")
            self.sio.emit("join", "obstructionDetected")

        @self.sio.event
        def disconnect():
            logger.warning("Disconnected from server.")
            self.reconnect()

    def reconnect(self):
        while True:
            try:
                logger.info("Attempting to reconnect...")
                self.sio.connect(self.config["poolpal_url"], wait=False)
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(3)

    def connect(self):
        try:
            self.sio.connect(self.config["poolpal_url"], wait=False)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.reconnect()

    def send_balls(self, balls):
        try:
            logger.info("Sending balls: %s", balls)
            self.sio.emit("ballPositions", balls)
            self.disconnect_counter = 0
        except Exception as e:
            self.disconnect_counter += 1
            logger.error("Failed to send ballPositions")
            if self.disconnect_counter >= 10:
                self.reconnect()
            pass

    def send_obstruction(self, obstruction_detected):
        try:
            self.sio.emit("obstructionDetected", obstruction_detected)
            self.disconnect_counter = 0
        except Exception as e:
            self.disconnect_counter += 1
            logger.error("Failed to send obstructionDetected")
            if self.disconnect_counter >= 10:
                self.reconnect()
            pass

    def disconnect(self):
        self.sio.disconnect()

    def start(self):
        threading.Thread(target=self.connect, daemon=True).start()
