# **Pool Ball Detection - Wide Angle Camera**

## **Overview**

This project performs real time pool ball detection using computer vision. It is capable of communicating with a rail system, and detecting obstructions on the table.

## **Features**

- **Ball detection** - Performs ball detection using pre-trained computer vision model.
- **Camera calibration** - Calibrates camera to reduce distortion.
- **Obstruction detection** - Uses an autoencoder to detect obstructions on the table for increased safety.
- **Configuration** - Key features can be modified with an intuative configuration file, with support for multiple profiles.

---

## **Installation**

### **Prerequisites**

Ensure you have Python **3.8+** installed. Then, install dependencies:

```bash
pip install -r requirements.txt
```

---

## **Usage**

### **Running the Project**

To start the program, simply run:

```bash
python main.py
```

- This will create a default configuration file and profile.

### **Configuration**

Modify `config.yaml` to adjust settings such as:

- Camera port
- Confidence threshold
- Position threshold
- Update interval
- Use networking

**Important settings**

- `camera_port` **Must** be set to a valid port (typically 0 or 1)
- `calibrate_camera` Default to `false`, if setting to `true` then camera calibration photos must be provided.
- `calibration_folder` Provide the folder name of the calibration photos. Must be placed within `/config/calibration/`.
  - Example: `calibration_folder : ["/folder_cam"]`
- `detection_model_path` Provide the path of a ball detection model if you wish to use your own.

Optionally create a new configuration profile with:

```bash
python main.py --create-profile [name]
```

### **Running with arguments**

There are optional arguments for running the program.

- `--set-points` - Set points for camera to crop
- `--no-anomaly` - Disable autoencoder for anomaly detection
- `--collect-model-images` - Collect images when holding `s` for use in model training.
- `--collect-ae-data` - Collect images when holding `s` for use in the autoencoder.

### **Running with Custom Config**

To test with a different config profile:

```bash
python main.py --profile [name]
```

## **License**

This project is licensed under the **MIT License**.
