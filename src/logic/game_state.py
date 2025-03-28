import time
import logging
import cv2
import numpy as np
import os
import json

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
        self.time_between_updates = self.config.network_update_interval
        self.time_since_last_update = time.time() - self.time_between_updates
        self.end_of_turn = False
        self.not_moved_counter = 0

        self.offset_manager = OffsetManager(config, mtx, dist)


    def update(self, data, labels, frame):
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

        current_time = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates: return

        balls = {}
        corrected_white_ball = {}

        num_balls = 0
        self.not_moved_counter = 0

        if data is None or len(data) == 0 or data[0].boxes is None:
            boxes = []
        else:
            boxes = data[0].boxes


        # Process detected balls
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
                    if self._has_moved(prev_ball, middlex, middley):
                        self.not_moved_counter += 1
                        prev_ball["x"] = middlex
                        prev_ball["y"] = middley
                        break

            if classname not in balls:
                balls[classname] = []
            balls[classname].append({"x": middlex, "y": middley})

        # Only update the state if there are new positions
        if self.not_moved_counter == num_balls:
            logger.debug("No significant ball movement detected. Skipping state update.")
            self.previous_state = balls

            # If balls stopped moving detected, end the turn. Only send once
            self._handle_end_of_turn()
            return

        # Update the socket with the new state
        self._update_and_send_balls(balls, corrected_white_ball, current_time)


    def _get_ball_info(self, ball, labels):
        """
        Gets the important info of the ball that is passed to it
        """
        xyxy_tensor = ball.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
        classidx = int(ball.cls.item())
        classname = labels[classidx]
        _middlex = int((xmin + xmax) // 2)
        _middley = int((ymin + ymax) // 2)

        # Clamp coordinates to boundaries
        middlex, middley = self._coords_clamped(_middlex, _middley)

        return classname, middlex, middley
    

    def _coords_clamped(self, _middlex, _middley):
        """
        Clamps the coordinates to the boundaries of the table
        """
        middlex = self.config.output_width if _middlex > self.config.output_width else _middlex
        middley = self.config.output_height if _middley > self.config.output_height else _middley
        middlex = 0 if middlex < 0 else middlex
        middley = 0 if middley < 0 else middley
        return int(middlex), int(middley)
    

    def _has_moved(self, prev_ball, middlex, middley):
        """
        Checks if the ball is close to a previous position
        """
        dx = abs(prev_ball["x"] - middlex)
        dy = abs(prev_ball["y"] - middley)
        return dx <= self.config.position_threshold and dy <= self.config.position_threshold
    

    def _handle_end_of_turn(self):
        """
        Sends message if the end of turn is detected
        """
        if not self.end_of_turn:
            self.end_of_turn = True
            if self.network:
                self.network.send_end_of_turn("true")
            else:
                logger.info("No movement detected, turn ended.")


    def _update_and_send_balls(self, balls, white_ball, current_time):
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

    def _send_white_ball(self, ball, current_time):
        if ball:
            self.time_since_last_update = current_time
            if self.network:
                self.network.send_corrected_white_ball(ball)
            else:
                logger.info(f"Sending corrected white ball: {ball}")

