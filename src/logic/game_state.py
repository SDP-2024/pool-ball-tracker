import time
class StateManager:
    def __init__(self, config, db_controller):
        self.config = config
        self.previous_state = None
        self.db_controller = db_controller
        self.time_between_updates = self.config["db_update_interval"]
        self.time_since_last_update = time.time() - self.time_between_updates
    # TODO: Move db updating to state manager so frequency can be reduced, and if balls havent moved enough, stop updating.
    # TODO: Implement game logic so that the turn can be decided.
    def update(self, data, labels):
        current_time = time.time()
        if current_time - self.time_since_last_update < self.time_between_updates:
            return
        balls = {}
        for ball in data[0].boxes:
            xyxy_tensor = ball.xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = map(int, xyxy.astype(int))
            classidx = int(ball.cls.item())
            classname = labels[classidx]
            conf = ball.conf.item()

            if conf > self.config["conf_threshold"]:
                middlex = int((xmin + xmax) // 2)
                middley = int((ymin + ymax) // 2)

                if classname not in balls:
                    balls[classname] = []
                balls[classname].append({"x": middlex, "y": middley})

        self.previous_state = balls
        self.db_controller.update({"balls": balls})
        self.time_since_last_update = current_time
