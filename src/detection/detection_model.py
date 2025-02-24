import cv2
import logging
import os
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

class DetectionModel:
    def __init__(self, config):
        """
        Initializes the DetectionModel with the given configuration.

        Args:
            config (dict): Configuration dictionary containing model path, confidence threshold, font scale, and font thickness.
        """
        self.config = config
        self.model_path = config["detection_model_path"]
        self.model = self.load_model()
        self.labels = self.model.names
        self.bbox_colors = [(0,0,0), (255,0,0), (255,255,255), (255,255,0)]
        self.count = 0
        self.track_history = defaultdict(lambda: [])


    def load_model(self):
        # Load model if present
        if (not os.path.exists(self.model_path)):
            return None
        else:
            return YOLO(self.model_path, task='detect')
        

    def detect(self, frame):
        """
        Detect objects in the given frame.

        Args:
            frame (numpy.ndarray): The input image frame in which to detect objects.

        Returns:
            tuple: A tuple containing the detection results and the labels.
        """
        self.count = 0
        results = self.model(frame, verbose=False)
        return results, self.labels
    

    def draw(self, frame, detected_balls):
        """
        Draw bounding boxes and labels on the frame for detected objects.

        Args:
            frame (numpy.ndarray): The input image frame on which to draw.
            detected_balls (tuple): The detection results containing bounding boxes and labels.
        """
        boxes = detected_balls[0].boxes
        for ball in boxes:
            xyxy_tensor = ball.xyxy.cpu() # Detections in Tensor format in CPU memory
            xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
            xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int
            classidx = int(ball.cls.item())
            classname = self.labels[classidx]
            
            conf = ball.conf.item()

            if conf > self.config["conf_threshold"]:
                color = self.bbox_colors[classidx % len(self.bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{classname}: {int(conf*100)}%"
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], self.config["font_thickness"])
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], (0, 0, 0), self.config["font_thickness"]) # Draw label text

                self.count += 1

        cv2.putText(frame, f'Number of objects: {self.count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1) # Draw count of objects
        cv2.imshow("Detection", frame)


    def extract_bounding_boxes(self, frame, balls):
        """
        Extract bounding boxes from the detected balls and create a mask for anomaly detection.

        Args:
            frame (numpy.ndarray): The input image frame from which to extract bounding boxes.
            balls (tuple): The detection results containing bounding boxes.

        Returns:
            numpy.ndarray: The frame with the detected objects inpainted.
        """
        detected_balls = []
        for ball in balls:
            for box in ball.boxes.xyxy:
                xmin, ymin, xmax, ymax = map(int, box)
                detected_balls.append((xmin, ymin, xmax, ymax))

        mask = np.zeros_like(frame[:, :, 0])
        for (xmin, ymin, xmax, ymax) in detected_balls:
            mask[ymin:ymax, xmin:xmax] = 255

        table_only = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        return table_only # All regions to be checked for anomalies


            

