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
            self.sio.emit("join", "endOfTurn")

        @self.sio.event
        def disconnect():
            logger.warning("Disconnected from server.")
            self.reconnect()

    def _reconnect(self):
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
            logger.info(f"Sending balls: {balls}")
            self.sio.emit("ballPositions", balls)
            self.disconnect_counter = 0
        except Exception as e:
            self._handle_error(e, "ballsPositions")
            pass
    
    def send_end_of_turn(self, end_of_turn):
        try:
            logger.info(f"Sending end of turn: {end_of_turn}")
            self.sio.emit("endOfTurn", end_of_turn)
            self.disconnect_counter = 0
        except Exception as e:
            self._handle_error(e, "endOfTurn")
            pass

    def send_obstruction(self, obstruction_detected):
        try:
            logger.info(f"Sending obstruction detected: {obstruction_detected}")
            self.sio.emit("obstructionDetected", obstruction_detected)
            self.disconnect_counter = 0
        except Exception as e:
            self._handle_error(e, "obstructionDetected")
            pass

    def disconnect(self):
        self.sio.disconnect()

    def start(self):
        threading.Thread(target=self.connect, daemon=True).start()

    def reconnect(self):
        threading.Thread(target=self._reconnect, daemon=True).start()

    def _check_reconnect(self):
        if self.disconnect_counter >= 10:
                self.reconnect()

    def _handle_error(self, e, name):
        self.disconnect_counter += 1
        logger.error(f"Failed to send {name}: {e}")
        self._check_reconnect()