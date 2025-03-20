import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, config, network=None):
        self.config = config
        self.previous_state = None
        self.network = network
        self.time_between_updates = self.config["network_update_interval"]
        self.time_since_last_update = time.time() - self.time_between_updates
        self.end_of_turn = False
        self.origin_set = False
        self.origin_offset = (0,0)
        self.holes_found = False
        self.holes = []
        self.corners = []

    # TODO: Handle balls that are missed for a few frames by detection
    def update(self, data, labels):
        current_time = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates:
            logger.debug("Skipping update: Too soon since last update.")
            return

        balls = {}
        position_threshold = self.config["position_threshold"]

        not_moved_counter = 0
        num_balls = 0

        if data is None or len(data) == 0 or data[0].boxes is None:
            boxes = []
        else:
            boxes = data[0].boxes

        # Process detected balls
        for ball in boxes:
            xyxy_tensor = ball.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
            classidx = int(ball.cls.item())
            classname = labels[classidx]

            # Check if the origin is set and if not then check if all holes are found
            # If all holes are found then identify the corners
            # Then set the origin to the top left corner
            if not self.origin_set and classname == "hole":
                middlex = int((xmin + xmax) // 2)
                middley = int((ymin + ymax) // 2)
                if not self.holes_found:
                    self.holes.append((middlex, middley))
                    if len(self.holes) == 6:
                        self.holes_found = True
                        self.order_holes()
                        self.corners = np.array([self.holes[0], self.holes[2], self.holes[3], self.holes[5]], dtype=np.float32)
                        logger.info(f"Ordered corners: {self.corners}")
                elif (middlex - 100) < 0 and (middley - 100) < 0:
                    self.origin_offset = (middlex, middley)
                    logger.info(f"Origin set to: {self.origin_offset}")
                    self.origin_set = True


            # Ignore arm and hole
            if classname == "arm" or classname == "hole":
                continue

            num_balls += 1

            middlex = int((xmin + xmax) // 2) - self.origin_offset[0]
            middley = int((ymin + ymax) // 2) - self.origin_offset[1]

            # Check if this ball is close to a previous position
            if self.previous_state and classname in self.previous_state:
                for prev_ball in self.previous_state[classname]:
                    dx = abs(prev_ball["x"] - middlex)
                    dy = abs(prev_ball["y"] - middley)
                    if dx <= position_threshold and dy <= position_threshold:
                        not_moved_counter += 1
                        prev_ball["x"] = middlex
                        prev_ball["y"] = middley
                        break  # Match found

            if classname not in balls:
                balls[classname] = []
            balls[classname].append({"x": middlex, "y": middley})

        # Only update the state if there are new positions
        if not_moved_counter == num_balls:
            logger.debug("No significant ball movement detected. Skipping state update.")
            self.previous_state = balls
            # If balls stopped moving detected, end the turn. Only send once
            if not self.end_of_turn:
                self.end_of_turn = True
                if self.network:
                    self.network.send_end_of_turn("true")
            return

        # Update the socket with the new state
        if balls:
            self.previous_state = balls
            self.time_since_last_update = current_time
            self.end_of_turn = False
            logger.info("Sending balls: %s", balls)

            if self.network:
                self.network.send_balls({"balls": balls})


    def order_holes(self):
        """
        Orders the holes in the following order:
        1 2 3
        4 5 6
        """
        if len(self.holes) != 6:
            logger.error("Holes not found.")
            return

        # Sort by y first
        self.holes.sort(key=lambda x: x[1])
        # Sort by x for each row
        self.holes[:3] = sorted(self.holes[:3], key=lambda x: x[0])
        self.holes[3:] = sorted(self.holes[3:], key=lambda x: x[0])

        logger.info("Ordered holes: %s", self.holes)
