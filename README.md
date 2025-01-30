# **Pool Ball Detection**

## **📌 Overview**
This project performs **pool ball and table detection** with **real-time stereo video stitching** using OpenCV. It supports camera calibration, image undistortion, and real-time processing of frames from two cameras.

## **🚀 Features**
- **Stereo Image Stitching**: Merges left and right frames into a seamless panoramic image.
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
- Camera calibration parameters
- Frame stitching frequency
- Image resolution and processing thresholds

Optionally create a new configuration profile with:
```bash
python src/main.py --create-profile [name]
```

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
