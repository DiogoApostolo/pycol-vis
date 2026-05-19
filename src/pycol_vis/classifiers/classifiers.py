from sklearn.calibration import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


import tensorflow as tf
from tensorflow.keras import layers, models


import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")


import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np





from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd

def svm_classifier(X_train, y_train, X_test, y_test):
    
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


def nn_classifier(X_train, y_train, X_test, y_test):
    
    

    clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

\
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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class PathDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['class'])

        if self.transform:
            image = self.transform(image)

        return image, label



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return self.relu(x)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.layer1 = ResidualBlock(32, 64, 2)
        self.layer2 = ResidualBlock(64, 128, 2)
        self.layer3 = ResidualBlock(128, 256, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def train_cnn(train_loader, num_classes, epochs=25, lr=0.001):

    print(f"Training on device: {DEVICE}")

    model = CNNClassifier(num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()

        total, correct, loss_sum = 0, 0, 0.0

        for images, labels in train_loader:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {loss_sum/total:.4f} | "
            f"Acc: {(correct/total)*100:.2f}%"
        )

    return model

def classify_images(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.to(DEVICE)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    return predicted.cpu().numpy()


def predict_on_test_df(model, test_df, transform, batch_size=32):

    dataset = PathDataset(test_df, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    preds = []

    with torch.no_grad():
        for images, _ in loader:

            images = images.to(DEVICE, non_blocking=True)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().numpy())

    return preds


def cnn_classifier(df, test_df):

    df = df.copy()
    test_df = test_df.copy()

    df['class'] = df['class'].astype('category')
    class_mapping = df['class'].cat.categories

    df['class'] = df['class'].cat.codes
    test_df['class'] = pd.Categorical(test_df['class'], categories=class_mapping).codes

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    NUM_CLASSES = len(class_mapping)

    train_dataset = PathDataset(df, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(0)
    )

    model = train_cnn(train_loader, NUM_CLASSES, epochs=30)

    y_pred = predict_on_test_df(model, test_df, transform)

    y_true = test_df['class'].tolist()

    return accuracy_score(y_true, y_pred)