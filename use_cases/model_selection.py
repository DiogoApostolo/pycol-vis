
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_metrics import ImageComplexity
from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier


'''
Model Selection use case example. In this example we embed the images using an efficient net and then train different classifiers on the embeddings and evaluate the accuracy.

Using the Overalp measures we can identify the difficulty of the classification task and then select a model that is more likely to perform well on the task.

'''


#Example of usage
dataset = "CovidDataset"
folder = "./" + dataset +  "/train/"

#classes = ["Circle","Square","Triangle"]
classes = ["COVID19","PNEUMONIA"]

depth = 1
epochs = 1

complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=400)
#complexity_train.define_feature_embedding_model(network_type="CNN",depth=depth)
#complexity_train.train_model(epochs=epochs,network_type="CNN")
metric_train = complexity_train.tabular_measure(emb_type="efficient_net",measure='kdn',reduction_type='pca')

X_train = complexity_train.feature_embeddings
y_train = complexity_train.images['class'].values

folder = "./" + dataset +  "/test/"
complexity_test = ImageComplexity(folder,keep_classes=classes,number_per_class=400)

#complexity_test.model_to_train = complexity_train.model_to_train
#complexity_test.model_all_layers = complexity_train.model_all_layers
#complexity_test.model = complexity_train.model

reduction_method = complexity_train.reduction_method
metric_test = complexity_test.tabular_measure(emb_type="efficient_net",measure='kdn',reduction_type='custom', reduction_method=reduction_method)



X_test = complexity_test.feature_embeddings
y_test = complexity_test.images['class'].values

accuracy_svm = svm_classifier(X_train,y_train,X_test,y_test)
accuracy_nn = nn_classifier(X_train,y_train,X_test,y_test)
accuracy_knn = knn_classifier(X_train,y_train,X_test,y_test)
accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)


print("Train kDN Score:", metric_train)
print("Test kDN Score:", metric_test)

print("SVM Accuracy:", accuracy_svm)
print("NN Accuracy:", accuracy_nn)
print("KNN Accuracy:", accuracy_knn)
print("XGB Accuracy:", accuracy_xgb)