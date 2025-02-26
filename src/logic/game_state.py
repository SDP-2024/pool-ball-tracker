class StateManager:
    def __init__(self, config):
        self.config = config

    def update(self, data, labels):
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
                balls[classname] = {"x": middlex, "y": middley}

        game_state = {"balls": balls}
        return game_state