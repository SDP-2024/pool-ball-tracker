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
    def __init__(self, config : dict) -> None:
        self.config = config
        self.sio = socketio.Client()
        self.positions_requested : bool = False
        self.finished_move : bool = False
        self.gantry_moving : bool = False
        self.finished_move_counter : int = 0
        self.finished_hit : bool = False
        self.moving_to_origin : bool = False

        @self.sio.event
        def connect() -> None:
            logger.info("Connected to server.")
            self.sio.emit("join", "ballPositions")
            self.sio.emit("join", "obstructionDetected")
            self.sio.emit("join", "endOfTurn")
            self.sio.emit("join", "requestPositions")
            self.sio.emit("join", "correctedPositions")
            self.sio.emit("join", "finishedMove")
            self.sio.emit("join", "finishedHit")
            self.sio.emit("join", "move")


        @self.sio.event
        def disconnect() -> None:
            logger.warning("Disconnected from server.")
            self.reconnect()


        @self.sio.on("requestPositions")
        def handle_request_positions(data) -> None:
            self._handle_request_positions(data)

        @self.sio.on("finishedMove")
        def handle_finished_move(data) -> None:
            self._handle_finished_move(data)

        @self.sio.on("finishedHit")
        def handle_finished_hit(data) -> None:
            self._handle_finished_hit(data)

        @self.sio.on("move")
        def handle_move(data) -> None:
            self._handle_move(data)


    def _reconnect(self) -> None:
        while True:
            try:
                logger.info("Attempting to reconnect...")
                self.sio.connect(self.config.poolpal_url, wait=False)
                break
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
                time.sleep(3)


    def connect(self) -> None:
        try:
            self.sio.connect(self.config.poolpal_url, wait=False)
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.reconnect()


    def _handle_request_positions(self, data) -> None:
        self.positions_requested = True

    def _handle_finished_move(self, data) -> None:
        """
        This handles the move commands. "finishedMove" is emitted when the gantry completes an instruction.
        One "finishedMove" signal means that the gantry has moved to the ball. If finishedMove and finishedHit then the gantry is moving back to origin.
        This signals to keep track of the balls that are in the origin position.
        Two "finishedMove" signals mean the gantry has moved to the ball and back to origin.
        This is required for disabling the obstruction detection during movement.
        """
        self.finished_move_counter += 1
        logger.info("Finished move")
        self.finished_move = True
        self.moving_to_origin = False
        self.gantry_moving = False


    def _handle_finished_hit(self, data) -> None:
        """
        This is used for tracking when the hitting mechanism has hit the ball.
        This is for tracking the origin point for balls that may be hidden by the hitting mechanism
        """
        self.finished_hit = True
        logger.info("Hit finished, moving back to origin.")

    def _handle_move(self, data) -> None:
        self.gantry_moving = True
        if (int(data['x']) == 0 and int(data['y'] == 0)):
            self.moving_to_origin = True
        logger.info("Gantry moving.")


    def send_balls(self, balls : dict) -> None:
        try:
            logger.info(f"Sending balls: {balls}")
            self.sio.emit("ballPositions", balls)
        except Exception as e:
            self._handle_error(e, "ballsPositions")
            pass


    def send_corrected_white_ball(self, ball : dict) -> None:
        try:
            logger.info(f"Sending ball: {ball}")
            self.sio.emit("correctedPositions", ball)
        except Exception as e:
            self._handle_error(e, "correctedPositions")
            pass
    

    def send_end_of_turn(self, end_of_turn : str) -> None:
        try:
            logger.info(f"Sending end of turn: {end_of_turn}")
            self.sio.emit("endOfTurn", end_of_turn)
        except Exception as e:
            self._handle_error(e, "endOfTurn")
            pass


    def send_obstruction(self, obstruction_detected : str) -> None:
        try:
            logger.info(f"Sending obstruction detected: {obstruction_detected}")
            self.sio.emit("obstructionDetected", obstruction_detected)
        except Exception as e:
            self._handle_error(e, "obstructionDetected")
            pass


    def disconnect(self) -> None:
        self.sio.disconnect()


    def start(self) -> None:
        threading.Thread(target=self.connect, daemon=True).start()


    def reconnect(self) -> None:
        threading.Thread(target=self._reconnect, daemon=True).start()


    def _handle_error(self, e, name) -> None:
        logger.error(f"Failed to send {name}: {e}")