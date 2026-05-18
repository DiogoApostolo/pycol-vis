from pycol_vis.image_metrics import ImageComplexity
from pycol_vis.classifiers.classifiers import cnn_classifier, svm_classifier, nn_classifier, knn_classifier, xgb_classifier

import random
import numpy as np
import tensorflow as tf
import os


'''
Model Selection use case example. In this example we embed the images using an efficient net and then train different classifiers on the embeddings and evaluate the accuracy.

Using the Overalp measures we can identify the difficulty of the classification task and then select a model that is more likely to perform well on the task.

Download the Dataset in https://www.kaggle.com/datasets/marquis03/fruits-100


'''

if __name__ == "__main__":


    SEED=0
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    #Example of usage
    dataset = "Fruit_dataset"
    folder = "./" + dataset +  "/train/"

    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://www.kaggle.com/datasets/marquis03/fruits-100 and place it in the correct location.")

    emb_type = "mobile_net"

    #classes = ["Circle","Square","Triangle"]
    #classes = ["COVID19","PNEUMONIA","NORMAL"]
    classes = ["apple", "black_berry", "mango", "pineapple","ackee"]
    depth = 3
    epochs = 10

    complexity_train = ImageComplexity(folder,keep_classes=classes)
    
    if(emb_type=="CNN"):
        complexity_train.cnn_setup(epochs=epochs,depth=depth)

    metric_train = complexity_train.tabular_measure(emb_type=emb_type,measure='n2',reduction_type='pca',n_components=3)
    #metric_train = complexity_train.csg_measure(emb_type=emb_type,reduction_type='pca',n_components=3)

    X_train = complexity_train.feature_embeddings
    y_train = complexity_train.images['class'].values

    folder = "./" + dataset +  "/test/"
    complexity_test = ImageComplexity(folder,keep_classes=classes)

    #complexity_test.model_to_train = complexity_train.model_to_train
    #complexity_test.model_all_layers = complexity_train.model_all_layers
    #complexity_test.model = complexity_train.model

    reduction_method = complexity_train.reduction_method

    complexity_test.model = complexity_train.model
    #metric_test = complexity_test.tabular_measure(emb_type=emb_type,measure='n2',reduction_type='custom', reduction_method=reduction_method)
    metric_test = complexity_test.tabular_measure(emb_type=emb_type,reduction_type='custom', reduction_method=reduction_method)


    #select the class with the highest complexity
    print("Train Complexity Score:", np.max(metric_train))
    print("Test Complexity Score:", np.max(metric_test))

    X_test = complexity_test.feature_embeddings
    y_test = complexity_test.images['class'].values

    accuracy_svm = svm_classifier(X_train,y_train,X_test,y_test)
    accuracy_nn = nn_classifier(X_train,y_train,X_test,y_test)
    accuracy_knn = knn_classifier(X_train,y_train,X_test,y_test)
    accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)
    accuracy_cnn = cnn_classifier(complexity_train.images,complexity_test.images)

    
    
    #accuracy_cnn = complexity_test.model.perform_classification(complexity_test.images['image_path'],y_test)  

    

    print("SVM Accuracy:", accuracy_svm)
    print("NN Accuracy:", accuracy_nn)
    print("KNN Accuracy:", accuracy_knn)
    print("XGB Accuracy:", accuracy_xgb)
    print("CNN Accuracy:", accuracy_cnn)