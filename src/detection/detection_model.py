from inference_sdk import InferenceHTTPClient
import cv2
import logging

logger = logging.getLogger(__name__)

class DetectionModel:
    def __init__(self, config):
        self.config = config
        self.client = InferenceHTTPClient("http://localhost:9001")

    def detect(self, frame):
        results =self.client.infer(frame)

        return results.get("predictions", [])

    def draw(self, frame, predictions):
        for prediction in predictions:
            x1, y1, x2, y2 = prediction["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, prediction["label"], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            

