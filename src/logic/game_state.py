import time
import logging
from tkinter import FALSE
import cv2

from src.logic.offset_manager import OffsetManager

logger = logging.getLogger(__name__)

class StateManager():
    """
    This class is in charge of managing the current state of the game.
    It monitors the current and previous state to detect when the balls have stopped moving.
    It also sends the current state to the network if it is available.
    """
    def __init__(self, config, network=None, mtx=None, dist=None):
        self.config = config
        self.previous_state = None
        self.network = network
        self.time_between_updates : float = self.config.network_update_interval
        self.time_since_last_update : float = time.time() - self.time_between_updates
        self.end_of_turn : bool = False
        self.not_moved_counter : int = 0
        self.hidden_state = None

        self.offset_manager = OffsetManager(config, mtx, dist)


    def update(self, data : tuple, labels : dict, frame : cv2.Mat) -> None:
        """
        Main update function for the game state.
        It gets the current ball positions and compares it to the previous state.
        If it thinks that there has been no movement between state, then it won't send another update.
        If the balls go from moving to not moving, then it would suggest that the turn is finished.

        Args:
            data (tuple): A tuple containing all the detected objects
            labels (dict): A dictionary of the object labels
        """
        # If the network has a received a request for positions, force positions to be sent.
        if self.network and self.network.positions_requested:
            self.previous_state = None
            self.network.positions_requested = False

        current_time : float = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates: return

        balls : dict = {}
        corrected_white_ball : dict = {}

        num_balls : int = 0
        self.not_moved_counter : int = 0

        boxes : list = []
        if data is not None or len(data) != 0 or data[0].boxes is not None:
            boxes = data[0].boxes

        # If gantry is moving then save the previous state and can begin adding balls that are revealed
        if self.network.gantry_moving:
            self.hidden_state = self.previous_state if self.previous_state is not None else None
            self.network.gantry_moving = False
            for ball in boxes:
                classname, middlex, middley = self._get_ball_info(ball, labels)
                if classname == "arm" or classname == "hole":
                    continue
                if classname == "white":
                    corrected_white_middlex, corrected_white_middley = self.offset_manager.update(frame, middlex, middley)
                    corrected_white_middlex, corrected_white_middley = self._coords_clamped(corrected_white_middlex, corrected_white_middley)
                    cv2.circle(frame, (corrected_white_middlex, corrected_white_middley), 4, (0, 255, 0), -1)
                    corrected_white_ball.update({"x": corrected_white_middlex, "y": corrected_white_middley})

                cv2.imshow("Detection", frame)
                if self.hidden_state and classname in self.hidden_state:
                    for saved_ball in self.hidden_state[classname]:
                        if not self._has_not_moved(saved_ball, middlex, middley):
                            if classname not in self.hidden_state:
                                    self.hidden_state[classname] = []
                            self.hidden_state[classname].append({'x': middlex, 'y': middley})


        # Process detected balls
        else:
            for ball in boxes:
                classname, middlex, middley = self._get_ball_info(ball, labels)

                # Ignore arm and hole
                if classname == "arm" or classname == "hole":
                    continue

                num_balls += 1

                if classname == "white":
                    corrected_white_middlex, corrected_white_middley = self.offset_manager.update(frame, middlex, middley)
                    corrected_white_middlex, corrected_white_middley = self._coords_clamped(corrected_white_middlex, corrected_white_middley)
                    cv2.circle(frame, (corrected_white_middlex, corrected_white_middley), 4, (0, 255, 0), -1)
                    corrected_white_ball.update({"x": corrected_white_middlex, "y": corrected_white_middley})

                cv2.imshow("Detection", frame)
                # Check if this ball is close to a previous position
                if self.previous_state and classname in self.previous_state:
                    for prev_ball in self.previous_state[classname]:
                        if self._has_not_moved(prev_ball, middlex, middley):
                            self.not_moved_counter += 1
                            prev_ball["x"] = middlex
                            prev_ball["y"] = middley
                            break

                if classname not in balls:
                    balls[classname] = []
                balls[classname].append({"x": middlex, "y": middley})

        # Only update the state if there are new positions, only if not currently tracking the hiddens state
        if self.not_moved_counter == num_balls and self.hidden_state is None:
            logger.debug("No significant ball movement detected. Skipping state update.")
            self.previous_state = balls

            # If balls stopped moving detected, end the turn. Only send once
            self._handle_end_of_turn()
            return
        
        # Handle the sending of the hidden state if the gantry is back at origin and reset
        if self.hidden_state is not None and self.network.finished_move:
            balls = self.hidden_state
            self.hidden_state = None

        # Update the socket with the new state
        self._update_and_send_balls(balls, corrected_white_ball, current_time)


    def _get_ball_info(self, ball, labels : dict) -> tuple[str, int, int]:
        """
        Gets the important info of the ball that is passed to it
        """
        xyxy_tensor = ball.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
        classidx : int = int(ball.cls.item())
        classname : str = labels[classidx]
        _middlex : int = int((xmin + xmax) // 2)
        _middley : int = int((ymin + ymax) // 2)

        # Clamp coordinates to boundaries
        middlex, middley = self._coords_clamped(_middlex, _middley)

        return classname, middlex, middley
    

    def _coords_clamped(self, _middlex : int, _middley : int) -> tuple[int, int]:
        """
        Clamps the coordinates to the boundaries of the table
        """
        middlex : int = self.config.output_width if _middlex > self.config.output_width else _middlex
        middley : int = self.config.output_height if _middley > self.config.output_height else _middley
        middlex : int = 0 if middlex < 0 else middlex
        middley : int = 0 if middley < 0 else middley
        return int(middlex), int(middley)
    

    def _has_not_moved(self, prev_ball, middlex : int, middley : int) -> bool:
        """
        Checks if the ball is close to a previous position
        """
        dx : int = abs(prev_ball["x"] - middlex)
        dy : int = abs(prev_ball["y"] - middley)
        return dx <= self.config.position_threshold and dy <= self.config.position_threshold
    

    def _handle_end_of_turn(self) -> None:
        """
        Sends message if the end of turn is detected
        """
        if not self.end_of_turn:
            self.end_of_turn = True
            if self.network:
                self.network.send_end_of_turn("true")
            else:
                logger.info("No movement detected, turn ended.")


    def _update_and_send_balls(self, balls, white_ball, current_time : float) -> None:
        """
        Sends the balls if new positions are detected
        """
        if balls:
            self.previous_state = balls
            self.time_since_last_update = current_time
            self.end_of_turn = False
            if self.network:
                self.network.send_balls({"balls": balls})
            else:
                logger.info(f"Sending balls: {balls}")
        self._send_white_ball(white_ball, current_time)


    def _send_white_ball(self, ball, current_time : float) -> None:
        if ball:
            self.time_since_last_update = current_time
            if self.network:
                self.network.send_corrected_white_ball(ball)
            else:
                logger.info(f"Sending corrected white ball: {ball}")

