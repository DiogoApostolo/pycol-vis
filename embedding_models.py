import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

import torch
from torchvision import transforms

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import cv2


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
    


class CNNEmbeddingModel():
    def __init__(self, image_shape, num_classes, depth=2):
        self.image_shape = image_shape
        self.num_classes = num_classes


        self.cnn_embedding_model(image_shape, num_classes, depth)

    
    def cnn_embedding_model(self, image_shape, num_classes, depth=1):
        '''
        define a simple CNN model for feature extraction. The model consists of a series of convolutional and max pooling layers, 
        followed by a flattening layer and two dense layers. 
        
        The number of convolutional layers can be adjusted using the depth parameter.


        Parameters:
        image_shape (tuple): The shape of the input images (width, height, channels).
        num_classes (int): The number of classes in the dataset.
        depth (int): The number of convolutional layers to include in the model (1 to 4).
        '''
        inputs = Input(shape=image_shape)
        layers = []
        
        x_1 = Conv2D(128, (3, 3), activation='relu')(inputs)
        x_1 = MaxPooling2D((2, 2), name='feature_map')(x_1)

        layers.append(x_1)
        x_conv = x_1
        if(depth>=2):
            x_2 = Conv2D(64, (3, 3), activation='relu')(x_1)
            x_2 = MaxPooling2D((2, 2))(x_2)
            layers.append(x_2)
            x_conv = x_2

        if(depth>=3):
            x_3 = Conv2D(32, (3, 3), activation='relu')(x_2)
            x_3 = MaxPooling2D((2, 2))(x_3)
            layers.append(x_3)
            x_conv = x_3
        if(depth>=4):
            x_4 = Conv2D(16, (3, 3), activation='relu')(x_3)
            x_4 = MaxPooling2D((2, 2))(x_4)
            layers.append(x_4)
            x_conv = x_4

        
        x_flatten = Flatten(name='flatten_layer')(x_conv)
        x_dense_1 = Dense(64, activation = 'relu')(x_flatten)
        x_dropout = Dropout(0.5)(x_dense_1)
        x_dense_2 = Dense(num_classes, activation = 'softmax')(x_dropout)

        layers.append(x_flatten)
        layers.append(x_dense_1)
        layers.append(x_dropout)


        self.model_to_train = Model(inputs=inputs, outputs=x_dense_2)
        self.model_all_layers = [Model(inputs=inputs, outputs=x) for x in layers]
        self.model = Model(inputs=inputs, outputs=x_conv)
        self.model_to_train.compile(optimizer='adam', loss='categorical_crossentropy')

    def get_feature_embeddings_all(self,images,layer_index=-1,batch_size=32):
        '''
        Auxiliary function to embed_images, it is used for the case of non-pre trained models.


        Extract the feature embeddings from the chosen model for all images in the dataset.
        The embeddings are extracted from the specified layer of the model.

        Parameters:
        layer_index (int): The index of the layer from which to extract the embeddings. Use -1 for the last feature layer.
        batch_size (int): The number of images to process in each batch when extracting embeddings.

        Returns:
        np.ndarray: A NumPy array containing the extracted feature embeddings for all images in the dataset.
        '''

        embeddings = []
        num_images = len(images)

        for batch_start_inx in range(0, num_images, batch_size):
            batch_end_inx = min(batch_start_inx + batch_size, num_images)
            batch_images = []

            for i in range(batch_start_inx, batch_end_inx):
                image = cv2.imread(images['image_path'].iloc[i])

                #resize to match the avg image size of the dataset
                image = cv2.resize(image, (self.model.input_shape[2], self.model.input_shape[1]))
                batch_images.append(image)

            batch_images = np.array(batch_images)

            #if layer_index is -1, use the last conv layer, otherwise use the specified layer
            if(layer_index==-1):
                
                features = self.model.predict(batch_images)
            else:
                features = self.model_all_layers[layer_index].predict(batch_images)

            embeddings.extend(features)

        
        return np.array(embeddings).reshape(len(images), -1)

    def get_feature_embeddings(self,image,layer_index=-1):
        '''
        Extract the feature embeddings from the chosen model for a single image.

        The embeddings are extracted from the specified layer of the model.

        Parameters:
        layer_index (int): The index of the layer from which to extract the embeddings. Use -1 for the last feature layer.
        
        Returns:
        np.ndarray: A NumPy array containing the extracted feature embeddings the image passed as input.
        '''
        
        
        image = cv2.resize(image, (self.model.input_shape[2], self.model.input_shape[1]))
        if(layer_index==-1):
            features = self.model.predict(np.expand_dims(image, axis=0))
        else:
            features = self.model_all_layers[layer_index].predict(np.expand_dims(image, axis=0))
        return features[0]
    

    def train_model(self,images,epochs=20):
        '''
        Train the defined model on the image dataset.

        Parameters:
        network_type (str): The type of model to train. Options are 'CNN' only at this time
        epochs (int): The number of epochs to train the model. 

        '''
        
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
       
        train_generator = train_datagen.flow_from_dataframe(
            dataframe=images,
            x_col="image_path",
            y_col="class",
            target_size=self.image_shape[:2],
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        self.model_to_train.fit(train_generator, epochs=epochs)
        self.is_trained = True
        return
        
        '''
        if(network_type=="CAE"):
            #load images for CAE
            for name in self.images['image_path']:
                image = self.load_image(name, convert_rgb=True)
                image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))
                image = image.astype('float32') / 255.0
                if 'x_data' not in locals():
                    x_data = np.expand_dims(image, axis=0)
                else:
                    x_data = np.vstack((x_data, np.expand_dims(image, axis=0)))


            self.model_to_train.fit(x_data, x_data, epochs=epochs,batch_size=32)
            self.is_trained = True
            return
        '''