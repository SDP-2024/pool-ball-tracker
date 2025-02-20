from calendar import c
import cv2
import logging
import os
from collections import defaultdict
import numpy as np

from ultralytics import YOLO

logger = logging.getLogger(__name__)

class DetectionModel:
    def __init__(self, config):
        self.config = config
        self.model_path = config["model_path"]
        self.model = self.load_model()
        self.labels = self.model.names
        self.bbox_colors = [(0,0,0), (255,0,0), (255,255,255), (255,255,0)]
        self.count = 0
        self.track_history = defaultdict(lambda: [])


    def load_model(self):
        if (not os.path.exists(self.model_path)):
            logger.error("Model path does not exist.")
            return None
        else:
            return YOLO(self.model_path, task='detect')
        
    def detect(self, frame):
        self.count = 0
        results = self.model(frame, verbose=False)
        return results, self.labels
    
    # def track(self, frame):
    #     results = self.model.track(frame, persist=True)

    #     # Get the boxes and track IDs
    #     boxes = results[0].boxes.xywh.cpu()
    #     track_ids = results[0].boxes.id.int().cpu().tolist()

    #     # Visualize the results on the frame
    #     annotated_frame = results[0].plot()

    #     # Plot the tracks
    #     for box, track_id in zip(boxes, track_ids):
    #         x, y, w, h = box
    #         track = self.track_history[track_id]
    #         track.append((float(x), float(y)))  # x, y center point
    #         if len(track) > 30:  # retain 90 tracks for 90 frames
    #             track.pop(0)

    #         # Draw the tracking lines
    #         points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    #         cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

    #     # Display the annotated frame
    #     cv2.imshow("YOLO11 Tracking", annotated_frame)
    
    def draw(self, frame, detected_balls):
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

                # Basic example: count the number of objects in the image
                self.count += 1

        cv2.putText(frame, f'Number of objects: {self.count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1) # Draw total number of detected objects
        cv2.imshow("Detection", frame)


            

