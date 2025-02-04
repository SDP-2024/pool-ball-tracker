# **Pool Ball Detection**

## **📌 Overview**

This project performs real-time **pool ball and table detection** using OpenCV. It supports camera calibration, image undistortion, and real-time processing of frames from two cameras.

## **🚀 Features**

- **Camera Calibration & Undistortion**: Uses pre-calibrated parameters to correct lens distortions.
- **Ball Detection**: Detects and highlights balls in the stitched image, identifying their colors.
- **Table Detection**: Detects a pool table and highlights the perimeter.
- **Configuration via YAML**: Adjust settings dynamically without modifying the code.

---

## **📦 Installation**

### **🔧 Prerequisites**

Ensure you have Python **3.8+** installed. Then, install dependencies:

```bash
pip install -r requirements.txt
```

---

## **⚙️ Usage**

### **1️⃣ Running the Project**

To start the program, simply run:

```bash
python src/main.py
```

- This will create a default configuration file and profile.

### **2️⃣ Configuration**

Modify `config.yaml` to adjust settings such as:

- Camera ports
- Color profile
- Size parameters

**Important settings**

- `camera_port_1` **Must** be set to a valid port (typically 0 or 1)
- `camera_port_2` Optional port, set to `-1` if you only wish to use one camera
- `calibrate_cameras` Default to `false`, if setting to `true` then camera calibration photos must be provided.
- `calibration_folders` Provide the folder name of the calibration photos. Must be placed within `/config/calibration/`.
  - Example: `calibration_folders : ["/folder_cam1", "/folder_cam2"]`
- `profile` Name of the color profile you wish to use.

Optionally create a new configuration profile with:

```bash
python src/main.py --create-profile [name]
```

**Modifying colors**

Colors can be modified by running:

```bash
python config/adjust_colors.py
```

This will launch a window which will allow you to adjust and save the HSV values for different colors in `colors.yaml`.

Color profiles can also be created to allow you to save common values under different lighting conditions.

---

## **📁 Project Structure**

```
📂 project_root/
│-- 📂 config/                  # Configuration files
|   |-- 📂 calibration/         # Folder for camera calibration photos
|   |-- config.yaml             # Configuration file
│-- 📂 src/                     # Source code
|   |-- 📂 detection/           # Detection code
│   │-- 📂 processing/          # Frame processing code, including stitching
│   │-- 📂 tracking/            # Ball tracking code
│   │-- main.py                 # Entry point of the project
│-- requirements.txt            # Dependencies
│-- README.md                   # Project documentation
```

---

## **🛠️ Development**

### **👨‍💻 Running with Custom Config**

To test with a different config profile:

```bash
python src/main.py --profile [name]
```

---

## **🙌 Contributing**

1. Fork the repo.
2. Create a new branch (`feature-branch`).
3. Commit changes (`git commit -m "Added new feature"`).
4. Push to branch (`git push origin feature-branch`).
5. Open a pull request.

---

## **📜 License**

This project is licensed under the **MIT License**.
