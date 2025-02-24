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
    def __init__(self, config):
        self.config = config
        self.autoencoder = self.build_autoencoder()

    def build_autoencoder(self):
        model_path = "model/autoencoder_model.keras"

        if os.path.exists(model_path):
            logger.info("Loading existing autoencoder model...")
            return load_model(model_path)

        # Define encoder
        input_img = Input(shape=(128, 128, 3))
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

        # Decoder
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        # Load images only if model is being trained
        clean_images = self.load_images(self.config["clean_images_path"])

        if len(clean_images) == 0:
            logger.warning("No training images found. Check the dataset path!")
            return None

        # Train the model if it doesn't exist
        logger.info("Training new model...")
        autoencoder.fit(clean_images, clean_images, epochs=50, batch_size=32)
        autoencoder.save(model_path)  # Save after training
        logger.info("Model trained and saved!")

        return autoencoder

    
    def load_images(self, folder_path):
        """Load and preprocess images from a given folder."""
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (128, 128))
                images.append(img)
                
        images = np.array(images, dtype="float32") / 255.0  # Normalize

        # Ensure shape is correct
        logger.info(f"Loaded {images.shape[0]} images with shape {images.shape[1:]}")
        return images


    def detect_anomaly(self, table_only):
        """Detect anomalies by comparing reconstruction errors."""
        anomaly = cv2.resize(table_only, (128, 128)) / 255.0
        anomaly = np.expand_dims(anomaly, axis=0)  # Ensure shape (1, 128, 128, 3)

        reconstructed = self.autoencoder.predict(anomaly, verbose=0)
        mse = np.mean(np.power(anomaly - reconstructed, 2))  # Compute Mean Squared Error
        logger.info(f"MSE: {mse}")
        return mse > self.config["anomaly_threshold"]



    

