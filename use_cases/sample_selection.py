from pycol_vis.image_metrics import ImageComplexity
from pycol_vis.classifiers.classifiers import svm_classifier

import random
import numpy as np
import tensorflow as tf

import os

'''
Use Case example of how to sample a dataset based on a complexity measure. 
In this example we use the jpeg compression ratio as a complexity measure and sample the dataset based on that. 
We then embed the images, reduce the dimensionality of the embeddings and train a classifier on the sampled dataset.

Change the N_SAMPLES_PER_CLASS variable to sample more or less images per class.

Download the dataset at https://github.com/DiogoApostolo/pycol-vis/blob/main/shapes_dataset.zip





'''

if __name__ == "__main__":

    SEED=0
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


    dataset = "shapes_dataset"
    folder = "./" + dataset +  "/train/"

    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://github.com/DiogoApostolo/pycol-vis/blob/main/shapes_dataset.zip or use the shapes_dataset.zip in this repo and place it in the correct location.")

    classes = ["Square","Circle","Triangle"]

    complexity_train = ImageComplexity(folder,keep_classes=classes)

    sample_num_array = [8000,6000,4000,2000]

    #Get the jpeg compression ratio for each image and sample the dataset based on that
    complexity_train.entropy_measure()
   


   
    #Embed the images and reduce the dimensionality of the embeddings (In a realistic scenario this would be done after sampling as to not embbed the full dataset, since this is testing multiple values it is faster this way)
    complexity_train.embed_images(emb_type='efficient_net')
    complexity_train.feature_embeddings = complexity_train.dim_reduction(complexity_train.feature_embeddings,method='pca',n_components=50)
    reduction_method = complexity_train.reduction_method



    folder = "./" + dataset +  "/test/"

    #Create a complexity object for the test set and use the same reduction method as the train set
    complexity_test = ImageComplexity(folder,keep_classes=classes)
    complexity_test.embed_images(emb_type='efficient_net')
    complexity_test.feature_embeddings = complexity_test.dim_reduction(complexity_test.feature_embeddings,method='custom',custom_method=reduction_method)

    X_test = complexity_test.feature_embeddings
    y_test = complexity_test.images['class'].values

    print("------------TRY DIFFERENT SAMPLE SIZES-----------------")
    for N_SAMPLES_PER_CLASS in sample_num_array:

        print("\n")
        complexity_train.sample_dataset(n_samples_per_class=N_SAMPLES_PER_CLASS, sample_type='entropy')
        print("Dataset Sampled")
        X_train = complexity_train.feature_embeddings
        
        print("Image Dataset Shape:")
        print(complexity_train.feature_embeddings.shape)
        y_train = complexity_train.images['class'].values
        
        #Train a classifier and evaluate the accuracy
        accuracy_svm = svm_classifier(X_train,y_train,X_test,y_test)
       

        print("Performance of classifiers:")
        print("SVM Accuracy:", accuracy_svm)
        

        print("----------------------------------")