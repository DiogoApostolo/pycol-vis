from pycol_vis import ImageComplexity
from pycol_vis.classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier

import random
import numpy as np
import tensorflow as tf

import pandas as pd

import os

'''
Use case of dimensionality reduction of feature embeddings. In this example we embed the images using an efficient net and then reduce the dimensionality of the embeddings using PCA.

We then train a classifier on the reduced embeddings and evaluate the accuracy.

dim_array variable can be changed to reduce to more or less dimensions. 

A user can indentify is a reduction method will be beneficial for the classification task by looking at the CSG measure before and after the dimensionality reduction. 

If the CSG measure decreases or remains low after the reduction, it is likely that the reduction has helped to improve class separability in the embedding space, which can lead to better classification performance.
Contrarily, if the CSG measure increases after the reduction, it may indicate that the reduction has removed important information from the embeddings, which can lead to worse classification performance.

Download the Dataset in https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

'''

if __name__ == "__main__":

    SEED=0
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)



    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    #check if folder exists and if not tell user to download the dataset and place it in the correct location
    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia and place it in the correct location.")

    classes = ["NORMAL","COVID19","PNEUMONIA"]
    emb_type = "efficient_net"
    dim_array = [2,50,100,1280]

    complexity_train = ImageComplexity(folder,keep_classes=classes,set_size=(200,200,3))
    
    #complexity_train.cnn_setup(epochs=10,depth=2)
    #complexity_train.cnn_setup(epochs=5,depth=3)
    complexity_train.embeddings.embed_images(emb_type=emb_type)
    
    folder = "./" + dataset +  "/test/"

    complexity_test = ImageComplexity(folder,keep_classes=classes,set_size=(200,200,3))
    complexity_test.model = complexity_train.model
    
    complexity_test.embeddings.embed_images(emb_type=emb_type)
    

    print("Train Shape")
    print(complexity_train.feature_embeddings.shape)

    print("Test Shape")
    print(complexity_test.feature_embeddings.shape)

    show_plt = False

    train_embeddings_original =   complexity_train.feature_embeddings.copy()
    test_embeddings_original  =   complexity_test.feature_embeddings.copy()

    perf_array = []
    comp_array = []
    comp_test_array = []

    for N_COMPONENTS in dim_array:

        print(N_COMPONENTS)
        complexity_train.feature_embeddings = complexity_train.embeddings.dim_reduction(train_embeddings_original,method='pca',n_components=N_COMPONENTS)
        reduction_method = complexity_train.reduction_method

        print("Reduction method used:")
        print(reduction_method)

       

        X_train = complexity_train.feature_embeddings
        y_train = complexity_train.images['class'].values

        print("Train set shape:")
        print(complexity_train.images.shape)

       
        complexity_test.feature_embeddings = complexity_test.embeddings.dim_reduction(test_embeddings_original,method='custom',custom_method=reduction_method)


        measure_test = complexity_test.overlap.csg_measure(emb_type="current",n_samples=1500, reduction_type='custom', reduction_method=reduction_method,auls=False)
        #measure_test = complexity_test.m_sep_measure(emb_type="current", reduction_type='custom', reduction_method=reduction_method)
        #measure_test = complexity_test.tabular_measure(emb_type="current", measure="kDN" , reduction_type='custom', reduction_method=reduction_method)


        if(show_plt):
            complexity_test.plot_tsne()

        X_test = complexity_test.feature_embeddings
        y_test = complexity_test.images['class'].values

        accuracy_nn = nn_classifier(X_train,y_train,X_test,y_test)
        print("NN Accuracy:", accuracy_nn)


        perf_array.append(accuracy_nn)

        comp_test_array.append(measure_test)

    
    df = pd.DataFrame({"Dim":dim_array, "Performace":perf_array, "Complexity Test":comp_test_array})
    print(df)