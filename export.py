from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("model/ball_detection_model.pt")

# Export the model to NCNN format
model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("./ball_detection_ncnn_model")
