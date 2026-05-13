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



def cnn_classifier(X_train, y_train, X_test, y_test):
    
    #define CNN architecture and train it on the data. Return accuracy on test set

    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))

    model.add(layers.Dense(len(set(y_train)), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    return test_acc
