from pycol_vis import ImageComplexity
from pycol_vis.classifiers import cnn_classifier, svm_classifier, nn_classifier, knn_classifier, xgb_classifier
import random
import numpy as np
import tensorflow as tf
import os

"""
Model Selection use case example. In this example we embed the images using a mobile net 
and then train different classifiers on the embeddings and evaluate the accuracy.

Using the Overlap measures we can identify the difficulty of the classification task 
and then select a model that is more likely to perform well on the task.

Dataset URL: https://github.com/DiogoApostolo/pycol-vis/blob/main/Fruit_dataset.zip
"""

if __name__ == "__main__":

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Dataset directory configuration mapping
    dataset = "Fruit_dataset"
    folder = "./" + dataset + "/train/"

    if not os.path.exists(folder):
        raise ValueError(
            "Folder " + folder + " does not exist. "
            "Please download the dataset from https://github.com/DiogoApostolo/pycol-vis/blob/main/Fruit_dataset.zip "
            "or use the Fruit_dataset.zip in this repo and place it in the correct location."
        )

    emb_type = "mobile_net"
    classes = ["apple", "black_berry", "mango", "pineapple", "ackee"]
    depth = 3
    epochs = 10

    complexity_train = ImageComplexity(folder, keep_classes=classes)
    
    if emb_type == "CNN":
        complexity_train.embeddings.cnn_setup(epochs=epochs, depth=depth)

    metric_train = complexity_train.overlap.tabular_measure(
        emb_type=emb_type, 
        measure='n2', 
        reduction_type='pca', 
        n_components=3
    )

    X_train = complexity_train.feature_embeddings
    y_train = complexity_train.images['class'].values

    # Initialize evaluation pipelines over the test subset partitions
    folder = "./" + dataset + "/test/"
    complexity_test = ImageComplexity(folder, keep_classes=classes)

    reduction_method = complexity_train.reduction_method
    complexity_test.model = complexity_train.model
    
    metric_test = complexity_test.overlap.tabular_measure(
        emb_type=emb_type, 
        reduction_type='custom', 
        reduction_method=reduction_method
    )

    # Select and display the baseline task metrics
    print("Train Complexity Score:", np.max(metric_train))
    print("Test Complexity Score:", np.max(metric_test))

    X_test = complexity_test.feature_embeddings
    y_test = complexity_test.images['class'].values

    # Execute downstream analytical classifier comparisons
    accuracy_svm = svm_classifier(X_train, y_train, X_test, y_test)
    accuracy_nn = nn_classifier(X_train, y_train, X_test, y_test)
    accuracy_knn = knn_classifier(X_train, y_train, X_test, y_test)
    accuracy_xgb = xgb_classifier(X_train, y_train, X_test, y_test)
    accuracy_cnn = cnn_classifier(complexity_train.images, complexity_test.images)

    print("SVM Accuracy:", accuracy_svm)
    print("NN Accuracy:", accuracy_nn)
    print("KNN Accuracy:", accuracy_knn)
    print("XGB Accuracy:", accuracy_xgb)
    print("CNN Accuracy:", accuracy_cnn)