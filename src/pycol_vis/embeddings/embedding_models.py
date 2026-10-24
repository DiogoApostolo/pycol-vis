

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, mobilenet_v3_small, MobileNet_V3_Small_Weights

from torchvision import transforms



import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.dataset_class import ImageDataset

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape=(28, 28, 1), latent_dim=32):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.encoder = None
        self.decoder = None
        self.autoencoder = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._build_model()
        self.to(self.device)

    def _build_model(self):
        height, width, channels = self.input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * ((height + 3) // 4) * ((width + 3) // 4), 128),
            nn.ReLU(),
            nn.Linear(128, self.latent_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder_dense = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.ReLU(),
            nn.Linear(
                128,
                64 * ((height + 3) // 4) * ((width + 3) // 4)
            ),
            nn.ReLU()
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 64, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 3, padding=1),
            nn.Sigmoid()
        )

        class Decoder(nn.Module):
            def __init__(self, dense, conv, shape):
                super().__init__()
                self.dense = dense
                self.conv = conv
                self.shape = shape

            def forward(self, z):
                h, w, c = self.shape
                x = self.dense(z)
                x = x.reshape(
                    -1, 64, (h + 3) // 4, (w + 3) // 4
                )
                x = self.conv(x)
                return F.interpolate(
                    x, size=(h, w), mode="bilinear",
                    align_corners=False
                )

        self.decoder = Decoder(
            self.decoder_dense, self.decoder_conv, self.input_shape
        )

        class Autoencoder(nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                return self.decoder(self.encoder(x))

        self.autoencoder = Autoencoder(self.encoder, self.decoder)

    def _prepare_images(self, images):
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(
                np.asarray(images), dtype=torch.float32
            )
        else:
            images = images.to(dtype=torch.float32)

        if images.ndim == 3:
            images = images.unsqueeze(0)

        # Convert NHWC to NCHW when necessary.
        if images.shape[-1] == self.input_shape[2]:
            images = images.permute(0, 3, 1, 2)

        return images.contiguous().to(self.device)

    def compile(self, optimizer="adam", loss="mse"):
        if isinstance(loss, str):
            if loss.lower() in ("mse", "mean_squared_error"):
                self.loss_fn = nn.MSELoss()
            else:
                raise ValueError(f"Unsupported loss: {loss}")
        else:
            self.loss_fn = loss

        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = torch.optim.Adam(
                    self.autoencoder.parameters()
                )
            elif optimizer.lower() == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.autoencoder.parameters(), lr=0.01
                )
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")
        else:
            self.optimizer = optimizer

    def fit(self, x_train, x_val=None, epochs=50, batch_size=128):
        if not hasattr(self, "optimizer"):
            self.compile()

        x_train = self._prepare_images(x_train)
        train_loader = DataLoader(
            TensorDataset(x_train),
            batch_size=batch_size,
            shuffle=True
        )

        x_val = (
            self._prepare_images(x_val)
            if x_val is not None else None
        )

        history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.autoencoder.train()
            total_loss = 0.0
            count = 0

            for (batch,) in train_loader:
                self.optimizer.zero_grad()
                output = self.autoencoder(batch)
                loss = self.loss_fn(output, batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss = total_loss / count
            history["loss"].append(train_loss)

            message = (
                f"Epoch {epoch + 1}/{epochs} "
                f"- loss: {train_loss:.6f}"
            )

            if x_val is not None:
                self.autoencoder.eval()
                with torch.no_grad():
                    val_loss = self.loss_fn(
                        self.autoencoder(x_val), x_val
                    ).item()

                history["val_loss"].append(val_loss)
                message += f" - val_loss: {val_loss:.6f}"

            print(message)

        return history

    def get_embeddings(self, images):
        self.encoder.eval()
        images = self._prepare_images(images)

        with torch.no_grad():
            return self.encoder(images).cpu().numpy()

    def reconstruct(self, images):
        self.autoencoder.eval()
        images = self._prepare_images(images)

        with torch.no_grad():
            output = self.autoencoder(images).cpu().numpy()

        return output.transpose(0, 2, 3, 1)

    


class EfficientNetLite0EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Let PyTorch handle devices natively via .to() externally
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.encoder.eval()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3, 224, 224)
        """
        # Dynamic check: ensure input matches whatever device the weights are currently on
        current_device = next(self.parameters()).device
        x = x.to(current_device)

        with torch.inference_mode():
            features = self.encoder(x)

        features = torch.flatten(features, 1)
        return features


class MobileNetV3EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base_model = mobilenet_v3_small(weights=weights)
        base_model.classifier = torch.nn.Identity()
        self.encoder = base_model.eval()
        self.preprocess_transform = weights.transforms()
        
    def forward(self, batch: torch.Tensor):
        """
        batch: Tensor of shape (batch_size, 3, 224, 224)
        """
        # Dynamic check: match the weight device dynamically
        current_device = next(self.parameters()).device
        batch = batch.to(current_device)
        
        with torch.inference_mode():
            features = self.encoder(batch)

        return features
    





class CNNEmbeddingModel:
    def __init__(self, image_shape, num_classes, depth=2):
        self.image_shape = image_shape
        self.num_classes = num_classes

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.class_indices = {}
        self.is_trained = False

        self.cnn_embedding_model(image_shape, num_classes, depth)

    def cnn_embedding_model(self, image_shape, num_classes, depth=1):
        """
        Build a CNN for classification and feature extraction.

        image_shape: (height, width, channels)
        num_classes: number of target classes
        depth: number of convolutional blocks (1 to 4)
        """
        if depth not in (1, 2, 3, 4):
            raise ValueError("depth must be between 1 and 4")

        self.depth = depth
        channels = image_shape[2]

        filters = [128, 64, 32, 16]

        self.conv_blocks = nn.ModuleList()
        in_channels = channels

        for i in range(depth):
            block = nn.Sequential(
                nn.Conv2d(in_channels, filters[i], kernel_size=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            )
            self.conv_blocks.append(block)
            in_channels = filters[i]

        with torch.no_grad():
            dummy = torch.zeros(
                1, channels, image_shape[0], image_shape[1]
            )
            x = dummy
            for block in self.conv_blocks:
                x = block(x)

            flattened_size = x.flatten(1).shape[1]
            self.feature_shape = tuple(x.shape[1:])

        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(flattened_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(64, num_classes)

        # Classification model and feature-extraction interface.
        self.model_to_train = _CNNNetwork(self)
        self.model = _CNNFeatureExtractor(self, -1)

        # Same layer ordering as the original Keras class:
        # pooled conv outputs, flatten, dense_1, dropout.
        self.model_all_layers = [
            _CNNFeatureExtractor(self, i)
            for i in range(depth + 3)
        ]

        self.model_to_train.to(self.device)
        self.model.to(self.device)
        for layer in self.model_all_layers:
            layer.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model_to_train.parameters()
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def _forward_features(self, x, layer_index=-1):
        """
        Layer indices:
          0 ... depth-1: pooled convolutional feature maps
          depth:         flattened features
          depth+1:       dense layer (ReLU)
          depth+2:       dropout output
         -1:              last convolutional feature map
        """
        if layer_index == -1:
            target = self.depth - 1
        else:
            target = layer_index

        if target < 0 or target >= self.depth + 3:
            raise IndexError("Invalid layer_index")

        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            if i == target:
                return x

        x = self.flatten(x)
        if target == self.depth:
            return x

        x = F.relu(self.dense_1(x))
        if target == self.depth + 1:
            return x

        return self.dropout(x)

    def _prepare_image(self, image):
        image = cv2.resize(
            image,
            (self.image_shape[1], self.image_shape[0])
        )

        # OpenCV returns BGR. Convert to RGB for conventional image input.
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(
            np.ascontiguousarray(image)
        ).float()

    def get_feature_embeddings_all(
        self, image_paths, layer_index=-1, batch_size=32
    ):
        """
        Extract flattened embeddings for all image paths.
        """
        self.model_to_train.eval()
        extractor = (
            self.model if layer_index == -1
            else self.model_all_layers[layer_index]
        )
        extractor.eval()

        embeddings = []

        with torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                batch_images = []

                for path in batch_paths:
                    image = cv2.imread(path)
                    if image is None:
                        raise ValueError(f"Could not read image: {path}")
                    batch_images.append(self._prepare_image(image))

                batch = torch.stack(batch_images).to(self.device)
                features = extractor(batch)
                embeddings.append(features.flatten(1).cpu().numpy())

        if not embeddings:
            return np.empty((0, 0), dtype=np.float32)

        return np.concatenate(embeddings, axis=0)

    def get_feature_embeddings(self, image, layer_index=-1):
        """
        Extract embeddings for one image supplied as a NumPy array.
        """
        extractor = (
            self.model if layer_index == -1
            else self.model_all_layers[layer_index]
        )
        extractor.eval()

        image_tensor = self._prepare_image(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            features = extractor(image_tensor)

        return features.flatten(1)[0].cpu().numpy()

    
    def train_model(self, images, epochs=10, batch_size=32):
        """
        Train the CNN using images loaded on demand in batches.
        """
        
        # Map class labels to integer indices
        classes = sorted(images["class"].unique())
        self.class_indices = {
            label: idx for idx, label in enumerate(classes)
        }

        if len(classes) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} classes, "
                f"but found {len(classes)}."
            )

        # Create the dataset using your existing ImageDataset
        image_paths = images["image_path"].tolist()
        labels = [
            self.class_indices[label]
            for label in images["class"].tolist()
        ]

        dataset = ImageDataset(
            image_paths=image_paths,
            preprocess=self.preprocess,
            labels=labels
        )

        # Load images in batches instead of loading everything into RAM
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        # Set up training
        self.model_to_train.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model_to_train.parameters(),
            lr=0.001
        )

        num_epochs = epochs
        self.model_to_train.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(
                    self.device, dtype=torch.long
                )

                optimizer.zero_grad()

                outputs = self.model_to_train(batch_images)
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                batch_size = batch_labels.size(0)
                running_loss += loss.item() * batch_size

                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_size

            epoch_loss = running_loss / total
            epoch_accuracy = correct / total

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Loss: {epoch_loss:.4f} "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

        self.model_to_train.eval()
        return self.model_to_train


    def preprocess(self, img):
        
        if img is None:
            raise ValueError("Could not load image.")

        # Resize to the expected image dimensions
        height, width, channels = self.image_shape
        img = cv2.resize(img, (width, height))

        # OpenCV uses BGR; convert to RGB for consistency
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[:, :, None]

        # Normalize pixel values from [0, 255] to [0, 1]
        img = img.astype("float32") / 255.0

        # Convert from (height, width, channels) to (channels, height, width)
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        return img



    def perform_classification(self, image_paths, target):
        """
        Evaluate classification accuracy and loss.
        """
        if not self.class_indices:
            raise RuntimeError("Train the model before evaluation.")

        images = []
        labels = []

        for path, label in zip(image_paths, target):
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not read image: {path}")
            if label not in self.class_indices:
                raise ValueError(f"Unknown class label: {label}")

            images.append(self._prepare_image(image))
            labels.append(self.class_indices[label])

        if not images:
            raise ValueError("No images provided for evaluation.")

        x = torch.stack(images).to(self.device)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)

        self.model_to_train.eval()
        with torch.no_grad():
            logits = self.model_to_train(x)
            loss = self.loss_fn(logits, y).item()
            accuracy = (logits.argmax(1) == y).float().mean().item()

        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

        return accuracy



class _CNNNetwork(nn.Module):
    def __init__(self, owner):
        super().__init__()

        # Register all trainable layers directly with PyTorch
        self.conv_blocks = owner.conv_blocks
        self.flatten = owner.flatten
        self.dense_1 = owner.dense_1
        self.dropout = owner.dropout
        self.dense_2 = owner.dense_2

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)

        x = self.flatten(x)
        x = F.relu(self.dense_1(x))
        x = self.dropout(x)

        return self.dense_2(x)



class _CNNFeatureExtractor(nn.Module):
    """Extract features from a selected CNN layer."""

    def __init__(self, owner, layer_index):
        super().__init__()
        self.owner = owner
        self.layer_index = layer_index

    def forward(self, x):
        return self.owner._forward_features(x, self.layer_index)
