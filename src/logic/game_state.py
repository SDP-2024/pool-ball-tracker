import time
import logging

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, config, network=None):
        self.config = config
        self.previous_state = None
        self.network = network
        self.time_between_updates = self.config["network_update_interval"]
        self.time_since_last_update = time.time() - self.time_between_updates

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

        # Process detected balls
        for ball in data[0].boxes:
            xyxy_tensor = ball.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
            classidx = int(ball.cls.item())
            classname = labels[classidx]

            # Ignore arm
            if classname == "arm":
                continue

            num_balls += 1

            middley = int((xmin + xmax) // 2)
            middlex = int((ymin + ymax) // 2)

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
            return

        # Update the socket with the new state
        if balls:
            self.previous_state = balls
            self.time_since_last_update = current_time

            if self.network:
                self.network.send_balls({"balls": balls})
            else:
                logger.warning("Network is not initialized. Cannot send ball positions.")
