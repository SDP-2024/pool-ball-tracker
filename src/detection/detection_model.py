import cv2
import logging
import os
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger(__name__)

class DetectionModel:
    """
    This class contains all the functions that are related to the detection of objects on the pool table.
    It filters, and draws the results to a window.
    """
    def __init__(self, config):
        self.config = config
        self.model_path = config["detection_model_path"]
        self.model = self.load_model()
        self.labels = self.model.names
        self.bbox_colors = [(0,0,255), (0,0,0), (0, 255, 0), (255,0,0), (255,255,255), (255,255,0)]
        self.total_objects = 0
        self.total_balls = 0
        self.track_history = defaultdict(lambda: [])


    def load_model(self):
        """
        Loads the model if it is present
        """
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
            tuple: A tuple containing the filtered detection results and the labels.
        """
        results = self.model(frame, verbose=False)
        
        # Filter the detected results
        all_results = []
        
        if results is None or len(results) == 0 or results[0].boxes is None:
            return None, None
        
        for result in results:
            for r in result.boxes:
                classidx = int(r.cls.item())
                classname = self.labels[classidx]
                all_results.append((r, classname))

        # Sort results by confidence
        all_results.sort(key=lambda x: x[0].conf.item(), reverse=True)

        filtered_results = self._filter_results(all_results)
        
        # Create a new result object with filtered boxes
        filtered_results_struct = results[0]
        filtered_results_struct.boxes = filtered_results
        
        return (filtered_results_struct, ), self.labels
    

    def _filter_results(self, all_results):
        """
        Helper function to filter the results to the detections with the highest confidence.
        It also ensures that no extra objects can be detected.
        """
        filtered_results = []
        counts = defaultdict(int)
        self.total_balls = 0
        for result, classname in all_results:
            if classname == "white" and counts["white"] < 1:
                counts["white"] += 1
                filtered_results.append(result)
                self.total_balls += 1
            elif classname == "black" and counts["black"] < 1:
                counts["black"] += 1
                filtered_results.append(result)
                self.total_balls += 1
            elif classname == "red" and counts["red"] < 7:
                counts["red"] += 1
                filtered_results.append(result)
                self.total_balls += 1
            elif classname == "yellow" and counts["yellow"] < 7:
                counts["yellow"] += 1
                filtered_results.append(result)
                self.total_balls += 1
            elif classname == "hole" and counts["hole"] < 6:
                counts["hole"] += 1
                filtered_results.append(result)
            elif classname == "arm":
                filtered_results.append(result)

        return filtered_results
    

    def draw(self, frame, filtered_results):
        """
        Draw bounding boxes and labels on the frame for detected objects.

        Args:
            frame (numpy.ndarray): The input image frame on which to draw.
            filtered_results (tuple): The detection results containing bounding boxes and labels.
        """
        if len(filtered_results) == 0 or filtered_results[0].boxes is None:
            boxes = []
        else:
            boxes = filtered_results[0].boxes
        self.total_objects = 0
        for result in boxes:
            classname, color, xmin, ymin, xmax, ymax = self._get_result_info(result)
            
            conf = result.conf.item()

            if conf > self.config["conf_threshold"]:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f"{classname}: {int(conf*100)}%"
                labelSize, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], self.config["font_thickness"])
                label_ymin = max(ymin, labelSize[1] + 10)
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], self.config["font_color"], self.config["font_thickness"])

                self.total_objects += 1
                
        cv2.putText(frame, f'Total number of objects: {self.total_objects}', (60,40), cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], self.config["font_color"], self.config["font_thickness"]) # Draw count of objects
        cv2.putText(frame, f'Total number of balls: {self.total_balls}', (60,60), cv2.FONT_HERSHEY_SIMPLEX, self.config["font_scale"], self.config["font_color"], self.config["font_thickness"]) # Draw count of objects
        cv2.imshow("Detection", frame)

    
    def _get_result_info(self, result):
        """
        Helper function to get the important result information
        """

        xyxy_tensor = result.xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)
        classidx = int(result.cls.item())
        classname = self.labels[classidx]
        color = self.bbox_colors[classidx % len(self.bbox_colors)]

        return classname, color, xmin, ymin, xmax, ymax


    def extract_bounding_boxes(self, frame, results):
        """
        Extract bounding boxes from the results and create a mask for anomaly detection.

        Args:
            frame (numpy.ndarray): The input image frame from which to extract bounding boxes.
            results (tuple): The detection results containing bounding boxes.

        Returns:
            numpy.ndarray: The frame with the detected objects inpainted.
        """
        
        bounding_boxes = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy().squeeze()
                xmin, ymin, xmax, ymax = map(int, xyxy)
                bounding_boxes.append((xmin, ymin, xmax, ymax))

        mask = np.zeros_like(frame[:, :, 0])
        for (xmin, ymin, xmax, ymax) in bounding_boxes:
            mask[ymin:ymax, xmin:xmax] = 255

        table_only = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        return table_only # All regions to be checked for anomalies


            

