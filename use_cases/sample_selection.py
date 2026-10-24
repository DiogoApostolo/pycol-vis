from pycol_vis import ImageComplexity
from pycol_vis.classifiers import svm_classifier
import random
import numpy as np
import tensorflow as tf
import os

"""
Use Case example of how to sample a dataset based on a complexity measure. 
In this example we use the entropy as a complexity measure and sample the dataset based on that. 
We then embed the images, reduce the dimensionality of the embeddings and train a classifier 
on the sampled dataset.

Change the N_SAMPLES_PER_CLASS variable to sample more or less images per class.

Dataset URL: https://github.com/DiogoApostolo/pycol-vis/blob/main/shapes_dataset.zip
"""

if __name__ == "__main__":

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    dataset = "shapes_dataset"
    folder = "./" + dataset + "/train/"

    # Check data path boundaries before launching pipeline modules
    if not os.path.exists(folder):
        raise ValueError(
            "Folder " + folder + " does not exist. "
            "Please download the dataset from https://github.com/DiogoApostolo/pycol-vis/blob/main/shapes_dataset.zip "
            "or use the shapes_dataset.zip in this repo and place it in the correct location."
        )

    classes = ["Square", "Circle", "Triangle"]
    complexity_train = ImageComplexity(folder, keep_classes=classes)

    sample_num_array = [8000, 6000, 4000, 2000]

    # Calculate the structural information entropy for each baseline frame
    complexity_train.intrinsic.entropy_measure()
   
    # Extract feature representations and run global dimensionality reductions
    # (In a production deploy, extraction occurs post-pruning to preserve computation resources;
    # this layout is optimized here for benchmarking sequential sample limits fast)
    complexity_train.embeddings.embed_images(emb_type='efficient_net')
    complexity_train.feature_embeddings = complexity_train.embeddings.dim_reduction(
        complexity_train.feature_embeddings, method='pca', n_components=50
    )
    reduction_method = complexity_train.reduction_method

    folder = "./" + dataset + "/test/"

    # Isolate out-of-sample evaluation boundaries using matching projection layers
    complexity_test = ImageComplexity(folder, keep_classes=classes)
    complexity_test.embeddings.embed_images(emb_type='efficient_net')
    complexity_test.feature_embeddings = complexity_test.embeddings.dim_reduction(
        complexity_test.feature_embeddings, method='custom', custom_method=reduction_method
    )

    X_test = complexity_test.feature_embeddings
    y_test = complexity_test.images['class'].values

    print("------------TRY DIFFERENT SAMPLE SIZES-----------------")
    for N_SAMPLES_PER_CLASS in sample_num_array:

        print("\n")
        # Apply conditional structural subset extraction based on entropy indices
        complexity_train.sample_dataset(n_samples_per_class=N_SAMPLES_PER_CLASS, sample_type='entropy')
        print("Dataset Sampled")
        
        X_train = complexity_train.feature_embeddings
        print("Image Dataset Shape:")
        print(complexity_train.feature_embeddings.shape)
        
        y_train = complexity_train.images['class'].values
        
        # Fit downstream classifiers to observe classification boundary performance shifts
        accuracy_svm = svm_classifier(X_train, y_train, X_test, y_test)
       
        print("Performance of classifiers:")
        print("SVM Accuracy:", accuracy_svm)
        print("----------------------------------")