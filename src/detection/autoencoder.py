from tensorflow import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input # type: ignore
from keras.models import Model, load_model # type: ignore
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


class AutoEncoder:
    """
    This autoencoder is for detecting obstruction on the pool table.
    It takes images of the empty pool table and attempts to reconstruct these images.
    If the reconstruction error is sufficiently high, it will notify the user.
    """
    def __init__(self, config):
        self.config = config
        self.autoencoder = self.build_autoencoder()
        # Hold the last few detections to calculate a rolling average, to reduce the sensitivity of the system.
        self.detection_buffer : np.ndarray = np.array([])

    def build_autoencoder(self):
        """
        This function will load an autoencoder if it exists.
        If it does not then it will build a new one using the provided images.
        """
        model_path : str = self.config.autoencoder_model_path

        if os.path.exists(model_path):
            logger.info("Loading existing autoencoder model...")
            return load_model(model_path)

        # Define encoder
        input_img = Input(shape=(128, 128, 3))
        x = self._build_encoder(input_img)

        # Build decoder
        decoded = self._build_decoder(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        # Load images only if model is being trained
        clean_images_path : str = self.config.clean_images_path
        clean_images : np.ndarray = self.load_images(clean_images_path)

        if len(clean_images) == 0:
            logger.warning("No training images found. Check the dataset path!")
            return None

        # Train the model if it doesn't exist
        logger.info("Training new model...")
        autoencoder.fit(clean_images, clean_images, epochs=50, batch_size=32)
        autoencoder.save(model_path)  # Save after training
        logger.info("Model trained and saved!")

        return autoencoder
    
    
    def _build_encoder(self, input_img):
        """
        Helper function to build the encoder.
        """
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        return x
    
    
    def _build_decoder(self, x):
        """
        Helper function to build the decoder.
        """
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        return decoded

    
    def load_images(self, folder_path : str) -> np.ndarray:
        """
        Load and process the images from the provided path.
        """
        images : list = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png")):
                img_path : str = os.path.join(folder_path, filename)
                img :cv2.Mat = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                else:
                    logger.warning(f"Failed to load image: {img_path}")
                
        if not images:
            logger.warning("No images loaded. Please check the folder path and image files.")
            return np.array([])

        images : np.ndarray[np.float32] = np.array(images, dtype=np.float32) / 255.0  # Normalize

        # Ensure shape is correct
        logger.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")
        return images


    def detect_anomaly(self, table_only : cv2.Mat) -> bool:
        """
        Detect anomalies by comparing reconstruction errors and checking threshold against a rolling average.
        """
        if self.autoencoder is None:
            logger.error("Autoencoder model is not loaded or trained.")
            return False

        anomaly : cv2.Mat = cv2.resize(table_only, (128, 128)) / 255.0
        anomaly : np.ndarray = np.expand_dims(anomaly, axis=0)

        reconstructed = self.autoencoder.predict(anomaly, verbose=0)
        mse : float = np.mean(np.square(anomaly - reconstructed))

        return self._rolling_average(mse) > self.config.anomaly_threshold
    

    def _rolling_average(self, mse : float) -> float:
        """
        This function maintains a rolling average of the mse.
        """
        if len(self.detection_buffer) >= self.config.anomaly_buffer_size:
            self.detection_buffer : np.ndarray = np.delete(self.detection_buffer, 0) 
        self.detection_buffer : np.ndarray = np.append(self.detection_buffer, mse)
        mean : float = np.mean(self.detection_buffer)
        if abs(mean - self.config.anomaly_threshold) <= self.config.anomaly_threshold * 0.05:
            logger.warning(f"Mean is within 5% of the threshold: {mean}")
            
        return mean



    

