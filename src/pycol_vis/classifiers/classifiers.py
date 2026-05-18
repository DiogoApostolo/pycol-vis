from sklearn.calibration import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


import tensorflow as tf
from tensorflow.keras import layers, models

def svm_classifier(X_train, y_train, X_test, y_test):
    
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def nn_classifier(X_train, y_train, X_test, y_test):
    
    

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def knn_classifier(X_train, y_train, X_test, y_test, n_neighbors=5):

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def xgb_classifier(X_train, y_train, X_test, y_test):
    
    

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    clf = XGBClassifier(eval_metric='mlogloss')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy





import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration (Global helper)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

class PathDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        dataframe: Your DataFrame containing 'image_path' and 'class' columns.
        transform: PyTorch transforms for resizing, normalizing, etc.
        """
        # Resetting index prevents KeyError if you passed a sliced/sampled dataframe
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Read the absolute or relative path directly from the row
        img_path = self.df.iloc[idx]['image_path']
        
        # Open the image file safely
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['class'])

        if self.transform:
            image = self.transform(image)

        return image, label


# =====================================================================
# A RESIDUAL BLOCK (The secret sauce behind modern, powerful CNNs)
# =====================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # If input size changes, match dimensions using a 1x1 convolution downsample
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection add
        out = self.relu(out)
        return out

# =====================================================================
# THE MODERN UPGRADED CNN CLASSIFIER
# =====================================================================
class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        
        # Initial Feature Extractor
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Deeper layers processing complex spatial features
        self.layer1 = ResidualBlock(32, 64, stride=2)   # Spatial size: 64x64 -> 32x32
        self.layer2 = ResidualBlock(64, 128, stride=2)  # Spatial size: 32x32 -> 16x16
        self.layer3 = ResidualBlock(128, 256, stride=2) # Spatial size: 16x16 -> 8x8
        
        # Global Average Pooling replaces massive flat linear layers to reduce parameter explosions
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output Classification Layer
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4), # Robust regularizer
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1) # Flattens seamlessly to (batch_size, 256)
        return self.fc(x)



def train_cnn(train_loader, num_classes, epochs=5, lr=0.001):
    print(f"Initializing training on device: {DEVICE}")
    model = CNNClassifier(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/total:.4f} - Accuracy: {(correct/total)*100:.2f}%")
        
    return model

def classify_images(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


def predict_on_test_df(model, test_df, transform, batch_size=16):
    """
    Takes a trained model and a testing DataFrame, runs batch inference, 
    and returns a list of predicted classes matching the rows of the test_df.
    """
    # FIX: Removed the "is_test=True" argument here so it matches your PathDataset class signature
    test_dataset = PathDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    
    print("Running predictions on test dataset...")
    with torch.no_grad():
        for batch_images, _ in test_loader:
            batch_images = batch_images.to(DEVICE)
            
            outputs = model(batch_images)
            _, predicted = torch.max(outputs, 1)
            
            # Move predictions back to CPU and collect them
            all_predictions.extend(predicted.cpu().numpy())
            
    return all_predictions


def cnn_classifier(df, test_df):
    # 1. Make local copies so we don't accidentally modify your original dataframes
    df = df.copy()
    test_df = test_df.copy()

    # 2. Convert text classes ('Triangle', etc.) to numerical codes (0, 1, 2...)
    df['class'] = df['class'].astype('category')
    class_mapping = df['class'].cat.categories
    
    df['class'] = df['class'].cat.codes
    test_df['class'] = pd.Categorical(test_df['class'], categories=class_mapping).codes

    # 3. Define transformations and dynamically assign number of classes
    data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    NUM_CLASSES = len(class_mapping)

    # 4. Set up the train dataset and loader
    train_dataset = PathDataset(df, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    try:
        # 5. Train and Predict
        trained_model = train_cnn(train_loader, num_classes=NUM_CLASSES, epochs=25)
        y_pred = predict_on_test_df(trained_model, test_df, transform=data_transforms, batch_size=32)

        # 6. Extract the true labels (which are now numerical codes matching y_pred)
        y_true = test_df['class'].tolist()

        # 7. Calculate and return accuracy
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    except FileNotFoundError as e:
        print(f"Execution stopped: Make sure the paths in your 'image_path' column exist. Details:\n{e}")
        return 0.0