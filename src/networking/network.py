import socketio
import time
import threading
import logging

logger = logging.getLogger(__name__)

class Network:
    """
    This class is in charge of the network connection to the pub/sub server.
    It sends the ball positions, obstruction detected, and end of turn messages to the server.
    It will track failed messages and attempt to reconnect if necessary.
    """
    def __init__(self, config):
        self.config = config
        self.sio = socketio.Client()
        self.positions_requested = False

        @self.sio.event
        def connect():
            logger.info("Connected to server.")
            self.sio.emit("join", "ballPositions")
            self.sio.emit("join", "obstructionDetected")
            self.sio.emit("join", "endOfTurn")
            self.sio.emit("join", "requestPositions")
            self.sio.emit("join", "correctedPositions")

        @self.sio.event
        def disconnect():
            logger.warning("Disconnected from server.")
            self.reconnect()

        @self.sio.on("requestPositions")
        def handle_request_positions(data):
            self._handle_request_positions(data)

    def _reconnect(self):
        while True:
            try:
                logger.info("Attempting to reconnect...")
                self.sio.connect(self.config.poolpal_url, wait=False)
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(3)

    def connect(self):
        try:
            self.sio.connect(self.config.poolpal_url, wait=False)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.reconnect()

    def _handle_request_positions(self, data):
        self.positions_requested = True

    def send_balls(self, balls):
        try:
            logger.info(f"Sending balls: {balls}")
            self.sio.emit("ballPositions", balls)
        except Exception as e:
            self._handle_error(e, "ballsPositions")
            pass

    def send_corrected_white_ball(self, ball):
        try:
            logger.info(f"Sending ball: {ball}")
            self.sio.emit("correctedPositions", ball)
        except Exception as e:
            self._handle_error(e, "correctedPositions")
            pass
    
    def send_end_of_turn(self, end_of_turn):
        try:
            logger.info(f"Sending end of turn: {end_of_turn}")
            self.sio.emit("endOfTurn", end_of_turn)
        except Exception as e:
            self._handle_error(e, "endOfTurn")
            pass

    def send_obstruction(self, obstruction_detected):
        try:
            logger.info(f"Sending obstruction detected: {obstruction_detected}")
            self.sio.emit("obstructionDetected", obstruction_detected)
        except Exception as e:
            self._handle_error(e, "obstructionDetected")
            pass

    def disconnect(self):
        self.sio.disconnect()

    def start(self):
        threading.Thread(target=self.connect, daemon=True).start()

    def reconnect(self):
        threading.Thread(target=self._reconnect, daemon=True).start()

    def _handle_error(self, e, name):
        logger.error(f"Failed to send {name}: {e}")