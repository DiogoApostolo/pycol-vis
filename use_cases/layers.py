

import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_metrics import ImageComplexity
from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier
import matplotlib.pyplot as plt

'''
Use case example of how to use the CSG measure to evaluate the complexity of the feature embeddings at different layers of a CNN.

Download the dataset at https://www.kaggle.com/datasets/marquis03/fruits-100?select=train

OR

Use the Fruit_dataset.zip in this repo

'''


dataset = "Fruit_dataset"
emb = "CNN"
layer = -1
depth = 4
epochs = 10



    
folder = "./" + dataset +  "/train/"



complexity = ImageComplexity(folder,keep_classes = ['apple','banana'],number_per_class=200)
complexity.cnn_setup(depth=depth,epochs=epochs)

csg_measures = []
for layer in range(0,depth):
    csg = complexity.csg_measure(emb_type=emb, reduction_type="pca", layer_index=layer)

    if(layer == -1):
        layer_name = "fin"
    else:
        layer_name = str(layer)

    print("CSG Measure for layer", layer_name, ":", csg)

    #save csg measure to an array and plot it after the loop
    csg_measures.append(csg)


layer_names = ["Layer " + str(i+1) for i in range(0,depth)]



plt.plot(range(0,depth), csg_measures, marker='o')
plt.xlabel("Layer")
plt.ylabel("CSG Measure")
plt.title("CSG Measure per Layer")
plt.xticks(range(0,depth), layer_names)
plt.show()

