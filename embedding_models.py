import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

import torch
from torchvision import transforms

class ConvAutoencoder:
    def __init__(self, input_shape=(28, 28, 1), latent_dim=32):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self._build_model()
    
    def _build_model(self):
        # Encoder
        encoder_inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation="relu")(x)
        latent = layers.Dense(self.latent_dim, activation="relu")(x)
        
        self.encoder = keras.Model(encoder_inputs, latent, name="encoder")
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation="relu")(latent_inputs)
        x = layers.Dense(7 * 7 * 64, activation="relu")(x)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(self.input_shape[2], 3, activation="sigmoid", padding="same")(x)
        
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
        
        # Autoencoder
        autoencoder_outputs = self.decoder(self.encoder(encoder_inputs))
        self.autoencoder = keras.Model(encoder_inputs, autoencoder_outputs, name="autoencoder")
    
    def compile(self, optimizer='adam', loss='mse'):
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
    
    
    
    def fit(self, x_train, x_val=None, epochs=50, batch_size=128):
        if x_val is not None:
            validation_data = (x_val, x_val)
        else:
            validation_data = None
            
        history = self.autoencoder.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=True
        )
        return history
    
    def get_embeddings(self, images):
        """Get latent space embeddings for input images"""
        return self.encoder.predict(images)
    
    def reconstruct(self, images):
        """Reconstruct images using the autoencoder"""
        return self.autoencoder.predict(images)
    

class EfficientNetLite0EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Base model
     
        base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        base.eval()

        # Remove classifier head
        self.encoder = torch.nn.Sequential(*list(base.children())[:-1])

    def forward(self, img):
        

        # If img is a list → batch
        if isinstance(img, list):
            tensors = [self.preprocess(i) for i in img]
            batch = torch.stack(tensors)
        else:
            # Single image
            batch = self.preprocess(Image.fromarray(img)).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(batch)

        return features.squeeze()  # (batch_size, 1280)


class MobileNetV3EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        

        
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.preprocess = weights.transforms()

        model = mobilenet_v3_small(weights=weights)
        model.classifier = torch.nn.Identity()  # remove classifier
        model.eval()

        self.encoder = model

    def forward(self, imgs):
    
        if isinstance(imgs, list):
            tensors = [self.preprocess(i) for i in imgs]
            batch = torch.stack(tensors)
        else:
            # Single image
            batch = self.preprocess(Image.fromarray(imgs)).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(batch)

        return features.squeeze()