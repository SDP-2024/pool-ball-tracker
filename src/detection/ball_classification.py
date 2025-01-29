import cv2 as cv
import numpy as np

class Classifier:
    def __init__(self, config):
        self.config = config.get("classifier", {})
        