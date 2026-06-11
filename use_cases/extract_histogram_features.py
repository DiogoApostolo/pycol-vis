from pycol_vis import ImageComplexity

import random
import numpy as np
import tensorflow as tf
import os

from pycol_vis.classifiers import cnn_classifier

if __name__ == "__main__":


    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Dataset directory configuration mapping
    dataset = "Fruit_dataset"
    folder = "./" + dataset + "/train/"


    classes = ["apple", "banana"]

    complexity_train = ImageComplexity(folder, keep_classes=classes)

       

    value = complexity_train.overlap.tabular_measure(emb_type="histogram_texture", reduction_type=None)

    print("Tabular measure: ", value)

    value = complexity_train.overlap.tabular_measure(emb_type="efficient_net")

    print("Tabular measure (efficient net): ", value)

    folder = "./" + dataset + "/test/"
    complexity_test = ImageComplexity(folder, keep_classes=classes)

    accuracy_cnn = cnn_classifier(complexity_train.images, complexity_test.images)

    print("CNN Classifier Accuracy: ", accuracy_cnn)