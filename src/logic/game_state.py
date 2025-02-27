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

                if classname not in balls:
                    balls[classname] = []
                balls[classname].append({"x": middlex, "y": middley})

        self.previous_state = balls
        return {"balls": balls}
