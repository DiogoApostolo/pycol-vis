from pycol_vis.image_metrics import ImageComplexity
from pycol_vis.classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier


'''
Use case of dimensionality reduction of feature embeddings. In this example we embed the images using an efficient net and then reduce the dimensionality of the embeddings using PCA.

We then train a classifier on the reduced embeddings and evaluate the accuracy.

N_COMPONENTS variable can be changed to reduce to more or less dimensions. 

A user can indentify is a reduction method will be beneficial for the classification task by looking at the CSG measure before and after the dimensionality reduction. 

If the CSG measure decreases after the reduction, it is likely that the reduction has helped to improve class separability in the embedding space, which can lead to better classification performance.
Contrarily, if the CSG measure increases after the reduction, it may indicate that the reduction has removed important information from the embeddings, which can lead to worse classification performance.

Download the Dataset in https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia

'''

if __name__ == "__main__":

    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    classes = ["PNEUMONIA","NORMAL"]

    N_COMPONENTS = 50

    complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=200)

    complexity_train.embed_images(emb_type='efficient_net')

    complexity_train.feature_embeddings = complexity_train.dim_reduction(complexity_train.feature_embeddings,method='pca',n_components=N_COMPONENTS)
    reduction_method = complexity_train.reduction_method

    print("Reduction method used:")
    print(reduction_method)

    measure = complexity_train.csg_measure(emb_type="current",n_samples=50, reduction_type='custom', reduction_method=reduction_method,auls=True)

    print("CSG Measure:", measure)

    X_train = complexity_train.feature_embeddings
    y_train = complexity_train.images['class'].values

    print("Train set shape:")
    print(complexity_train.images.shape)

    folder = "./" + dataset +  "/test/"

    complexity_test = ImageComplexity(folder,keep_classes=classes,number_per_class=400)
    complexity_test.embed_images(emb_type='efficient_net')
    complexity_test.feature_embeddings = complexity_test.dim_reduction(complexity_test.feature_embeddings,method='custom',custom_method=reduction_method)

    complexity_test.plot_tsne()

    X_test = complexity_test.feature_embeddings
    y_test = complexity_test.images['class'].values

    accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)
    print("XGB Accuracy:", accuracy_xgb)