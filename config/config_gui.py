from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QSlider
from PyQt6.QtCore import Qt
import sys
import logging

logger = logging.getLogger(__name__)

class ConfigInterface(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Configuration Tool")
        self.resize(300, 300)
        self.layout = QVBoxLayout()

        for key in self.config.schema.keys():
            setattr(self, f"{key}_label", None)
            setattr(self, f"{key}_option", None)

        self._setup()
    
    def _setup(self):
        for key, value in self.config.schema.items():
            if value["type"] == "integer":
                logger.info(f"{key} is of type integer")
                slider = QSlider(Qt.Orientation.Horizontal)
                max = value.get("max", value["min"] * 10)
                min = value.get("min", 0)
                max = 10 if max <= 0 else max
                slider.setRange(min, max)
                slider.setValue(getattr(self.config, key))
                setattr(self, f"{key}_option", slider)

                label = QLabel(f"{key}: {(getattr(self, f"{key}_option")).value()}")
                setattr(self, f"{key}_label", label)
                self.layout.addWidget(getattr(self, f"{key}_label"))
                self.layout.addWidget(getattr(self, f"{key}_option"))

        self.setLayout(self.layout)


def run_config_interface(config):
    """
    Entry point for config tool.
    This is run on its own thread.
    """
    app = QApplication(sys.argv)
    window = ConfigInterface(config)
    window.show()
    app.exec()