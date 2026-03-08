
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_metrics import ImageComplexity
from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier



'''
Use Case example of how to sample a dataset based on a complexity measure. 
In this example we use the jpeg compression ratio as a complexity measure and sample the dataset based on that. 
We then embed the images, reduce the dimensionality of the embeddings and train a classifier on the sampled dataset.

Change the N_SAMPLES_PER_CLASS variable to sample more or less images per class.

'''

dataset = "shapes_dataset"
folder = "./" + dataset +  "/train/"

classes = ["Circle","Square","Triangle"]

complexity_train = ImageComplexity(folder,keep_classes=classes)

N_SAMPLES_PER_CLASS = 2000

#Get the jpeg compression ratio for each image and sample the dataset based on that
complexity_train.jpeg_compression_ratio()
complexity_train.sample_dataset(n_samples_per_class=N_SAMPLES_PER_CLASS,sample_type='jpeg_compression')

#Embed the images and reduce the dimensionality of the embeddings
complexity_train.embed_images(emb_type='efficient_net')
complexity_train.feature_embeddings = complexity_train.dim_reduction(complexity_train.feature_embeddings,method='pca',n_compoments=10)
reduction_method = complexity_train.reduction_method



X_train = complexity_train.feature_embeddings
y_train = complexity_train.images['class'].values

print(complexity_train.images.shape)

folder = "./" + dataset +  "/test/"

#Create a complexity object for the test set and use the same reduction method as the train set
complexity_test = ImageComplexity(folder,keep_classes=classes)
complexity_test.embed_images(emb_type='efficient_net')
complexity_test.feature_embeddings = complexity_test.dim_reduction(complexity_test.feature_embeddings,method='custom',custom_method=reduction_method)


X_test = complexity_test.feature_embeddings
y_test = complexity_test.images['class'].values

#Train a classifier and evaluate the accuracy
accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)
print("XGB Accuracy:", accuracy_xgb)
