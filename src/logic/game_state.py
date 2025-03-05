import time
import logging

logger = logging.getLogger(__name__)

class StateManager:
    def __init__(self, config, network):
        self.config = config
        self.previous_state = None
        self.network = network
        self.time_between_updates = self.config["network_update_interval"]
        self.time_since_last_update = time.time() - self.time_between_updates

    def update(self, data, labels):
        current_time = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates:
            logger.debug("Skipping update: Too soon since last update.")
            return

        balls = {}
        position_threshold = self.config["position_threshold"]

        # Initialize missing_frames for each ball in the previous state
        if self.previous_state:
            for color, prev_balls in self.previous_state.items():
                for prev_ball in prev_balls:
                    prev_ball["missing_frames"] = prev_ball.get("missing_frames", 0) + 1

        # Process detected balls
        for ball in data[0].boxes:
            xyxy_tensor = ball.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
            classidx = int(ball.cls.item())
            classname = labels[classidx]
            conf = ball.conf.item()

            if conf > self.config["conf_threshold"]:
                middley = int((xmin + xmax) // 2)
                middlex = int((ymin + ymax) // 2)

                # Check if this ball is close to a previous position
                is_new_position = True
                if self.previous_state:
                    for prev_ball in self.previous_state.get(classname, []):
                        dx = abs(prev_ball["x"] - middley)
                        dy = abs(prev_ball["y"] - middlex)
                        if dx <= position_threshold and dy <= position_threshold:
                            is_new_position = False
                            # prev_ball["x"] = middley
                            # prev_ball["y"] = middlex
                            middley = prev_ball["x"]
                            middlex = prev_ball["y"]
                            break  # No need to check further, we found a match

                if is_new_position:
                    if classname not in balls:
                        balls[classname] = []
                    balls[classname].append({"x": middlex, "y": middley})

        # Update the database with the new state
        if balls:
            #logger.info("Updating state with detected balls: %s", balls)
            self.previous_state = balls
            self.time_since_last_update = current_time

            if self.network:  # Ensure network is not None
                self.network.send_balls({"balls": balls})
            else:
                logger.warning("Network is not initialized. Cannot send ball positions.")
