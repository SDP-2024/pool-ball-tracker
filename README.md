# **Pool Ball Detection - Raspberry Pi**

## **Overview**

This project performs real time pool ball detection using a trained computer vision model. It is capable of communicating with a rail system, and detecting obstructions on the table.

## **Features**

- **Ball detection** - Performs ball detection using pre-trained computer vision model.
- **Camera calibration** - Calibrates cameras to reduce distortion.
- **Obstruction detection** - Uses an autoencoder to detect obstructions on the table for increased safety.
- **Configuration** - Key features can be modified with an intuative configuration file, with support for multiple profiles.
- **Raspberry Pi support** - This branch is specifically optimised for the Raspberry Pi.

---

## **Installation on raspberry pi**

Ensure you are running the 64-bit of Raspberry Pi OS Bookworm.

Clone this branch of the respository.

```bash
git clone -b ras-pi https://github.com/SDP-2024/pool-ball-tracker.git
```

Create a virtual environment. Must include the `--system-site-packages` flag to allow `picamera2` to be accessed.

```bash
python3 -m venv --system-site-packages env
```

Then activate with `source env/bin/activate`.

Install dependencies:

```bash
pip install -r requirements.txt
```

**Important**

Numpy version `1.26.0` **must** be installed for compatibility. Some packages may overwrite this.

```bash
pip uninstall numpy
pip install numpy==1.26.0
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

- `calibrate_cameras`
- `camera_port_1`
- `use_networking`
- `conf_threshold`
- `anomaly_threshold`
- `position_threshold`

Optionally create a new configuration profile with:

```bash
python main.py --create-profile [name]
```

There are further optional flags.

- `--collect-ae-data` - Collect frames for use in autoencoder
- `--set-points` - Set points for pool table corners
- `--no-anomaly` - Disabled anomaly detection
- `--collect-model-images` - Collect images for training a new model
- `--no-draw` - Disable creating any windows

### **Running with Custom Config**

To test with a different config profile:

```bash
python main.py --profile [name]
```

## **License**

This project is licensed under the **MIT License**.
