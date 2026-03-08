



import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from image_metrics import ImageComplexity
from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier

'''
Visualization use case example. In this example we embed the images using an efficient net and then visualize the embeddings using t-SNE.

Furthermore we use multiple overlap measures and plot a bar plot to visualize the complexity of the dataset according to different measures.

Methods to visualize the metrics per class are also available, allowing the user to identify which classes are more complex and which are less complex according to a specific metric.
'''

dataset = "shapes_dataset"
folder = "./" + dataset +  "/train/"

classes = ["Circle","Square","Triangle"]

complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=200)




complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca')
complexity_train.tabular_measure(emb_type='efficient_net',measure='kdn',reduction_type='pca')
complexity_train.m_sep_measure(emb_type='efficient_net', reduction_type='pca')
complexity_train.plot_overlap_measures()

complexity_train.plot_tsne(embs=complexity_train.feature_embeddings)


#complexity_train.calculate_energy()


#complexity_train.jpeg_compression_ratio()
#complexity_train.calculate_entropy()
#complexity_train.edge_density_canny()

#complexity_train.visualize_metrics_per_class('entropy')

#complexity_train.sample_dataset(n_samples_per_class=5000,sample_type='jpeg_compression')
