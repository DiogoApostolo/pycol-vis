# overlap_measures.py


import random

import numpy as np
from pycol_complexity import complexity as pycol_complexity

from .overlap_utils import compute_class_evidence, compute_class_prototype_metrics, compute_decision_tree_metrics, compute_interaction_aware_metrics, compute_patch_auc_metrics, validate_inputs, compute_normalized_matrices,compute_m_sep_direct,compute_m_var,compute_similarity_matrix_S,compute_adjacency_matrix_W,compute_laplacian_matrix_L,compute_spectrum,compute_csg_complexity,compute_AULS_complexity,compute_cmsAULS_complexity, conditional_entropy, cross_class_edge_ratio, conditional_mst_entropy, extract_features, compute_between_within_metrics, metrics_to_scalars, metrics_to_maps, compute_knn_metrics, aggregate_descriptors
import cv2



def m_var_measure(embeddings, labels):
    '''
    Compute the M_var measure of class variability in the embedding space.

    M_var is calculated using the normalized within-class scatter matrix (S_w_hat) in the embedding space, which captures the variability of samples within each class. 
    A lower M_var value indicates that samples within the same class are more tightly clustered together, suggesting better class separability.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.

    Returns:
    - float: The calculated M_var measure, which quantifies the variability of samples within each class in the embedding space. A lower value indicates better class separability.
    '''

    validate_inputs(embeddings, labels)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    S_w_hat, S_b_hat = compute_normalized_matrices(embeddings, labels)

    m_var = compute_m_var(S_w_hat,len(np.unique(labels)),embeddings.shape[1])

    return m_var


def m_sep_measure(embeddings, labels):
    '''
    Compute the M_sep measure of class separability in the embedding space.
    M_sep is calculated using the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat) in the embedding space.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.

    Returns:
    - float: The calculated M_sep measure, which quantifies the separability of classes in the embedding space. A higher value indicates better class separability.
    
    
    '''
    validate_inputs(embeddings, labels)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    S_w_hat, S_b_hat = compute_normalized_matrices(embeddings, labels)

    m_sep = compute_m_sep_direct(S_w_hat, S_b_hat)

    return m_sep


def tabular_measure(embeddings, labels, measure='kdn', n_samples=None):
    '''
    Calculate overlap measures using the pycol complexity libray.

    A lower value of the meaure indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.
    - measure (str): The complexity measure to calculate. Options are 'n2', 'kdn', or 'lsc'.
    - n_samples (int): The number of samples to use for calculating the complexity measure.

    Returns:
    - float: The calculated complexity measure value for the given embeddings and labels.
    '''
     
    


    validate_inputs(embeddings, labels)

    inxs = np.arange(len(embeddings))

    if n_samples is not None:
        selected_inxs = np.random.choice(inxs, n_samples, replace=False) 
        embeddings = embeddings[selected_inxs]
        labels = labels[selected_inxs]

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)



    dataset_dic = {
        'X': embeddings,
        'y': labels
    }

    measure = measure.lower()

    print("Calculating " + measure + " measure using pycol_complexity library...")

    if(measure=='n2'):
        comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).N2(imb=True)

    elif(measure=='kdn'):
        comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).kDN(imb=True)

    elif(measure=='lsc'):
        comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).LSC(imb=True)
    elif(measure=='f4'):
        comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).F4(imb=True)


    else:
        raise ValueError("Measure must be one of 'f4', 'n2', 'kdn', or 'lsc'.")

    return comp_value


def auls_measure(embeddings, labels, n_samples=50):
    '''
    Calculate the AULS complexity measure based on the spectrum of the graph. AULS is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
    A lower AULS value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.
    - n_samples (int): The number of samples to use for calculating the similarity matrix. Default is 50.

    Returns:
    - float: The calculated AULS complexity measure value for the given embeddings and labels.
    
    '''
    validate_inputs(embeddings, labels)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity_matrix_S = compute_similarity_matrix_S(embeddings,labels,np.unique(labels),n_samples=n_samples)

    W = compute_adjacency_matrix_W(similarity_matrix_S)

    L, D = compute_laplacian_matrix_L(W)

    eigenvalues, eigenvectors = compute_spectrum(L)

    auls = compute_AULS_complexity(eigenvalues)

    return auls





def csg_measure(embeddings, labels, n_samples=50, auls=False, normalize_density=True):
    '''
    Calculate the CSG complexity measure based on the spectrum of the graph. CSG is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
    A lower CSG value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.
    - n_samples (int): The number of samples to use for calculating the similarity matrix. Default is 50.
    - auls (bool): Whether to calculate the CMS-AULS complexity measure instead of CSG. If True, the CMS-AULS measure will be calculated using the eigenvalues of the Laplacian matrix. Default is False.

    Returns:
    - float: The calculated CSG complexity measure value for the given embeddings and labels. If auls is True, the calculated CMS-AULS complexity measure value will be returned instead.
    '''
    

    #sanatize inputs
    validate_inputs(embeddings, labels)

    if(n_samples <= 0):
        raise ValueError("n_samples must be a positive integer.")
    
    if(n_samples > 2000):
        print("Warning: n_samples is set to a high value, which may lead to long computation times. Consider reducing n_samples for faster results.")

    if(n_samples < 20):
        print("Warning: n_samples is set to a low value, which may lead to less stable results. Consider increasing n_samples for more reliable results.")


    



    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity_matrix_S = compute_similarity_matrix_S(embeddings,labels,np.unique(labels),n_samples=n_samples)



    

    W = compute_adjacency_matrix_W(similarity_matrix_S)

    

    L, D = compute_laplacian_matrix_L(W)

    eigenvalues, eigenvectors = compute_spectrum(L)
    if(auls):
        measure = compute_cmsAULS_complexity(eigenvalues)

    else:
        measure = compute_csg_complexity(eigenvalues)

   
    return measure



#First Version
def entropy_based_measure(embeddings, labels, k=5):

    validate_inputs(embeddings, labels)

    H_cond = conditional_entropy(embeddings, labels)
    overlap = cross_class_edge_ratio(embeddings, labels, k)

    return H_cond * overlap


#Second version using entropic graphs
def entropic_graphs_measure(embeddings,labels, k=5):
    validate_inputs(embeddings, labels)

    H = conditional_mst_entropy(embeddings,labels,per_class=True)
    overlap = cross_class_edge_ratio(embeddings, labels, k, per_class=True)



    return H*overlap




def fisher_discriminative_map(image_paths, labels, image_size=(64, 64)):
    labels = np.asarray(labels)

    classes = np.unique(labels)

    images = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, image_size)

        images.append(img.astype(np.float32))

    images = np.stack(images)

    global_mean = np.mean(images, axis=0)

    maps = {}

    for c in classes:
        imgs_c = images[labels == c]

        if (len(imgs_c) < 2):
            continue

        mean_c = np.mean(imgs_c, axis=0)

        within = np.var(imgs_c, axis=0)

        between = (mean_c - global_mean) ** 2

        R = between / (between + within + 1e-10)

        maps[c] = R

    return maps


import matplotlib.pyplot as plt
def visualize_images(image_paths, labels, image_size=(64, 64), n_cols=5, n_images=None):

    
    if(n_images is not None):

        #sample the images and labels to visualize
        indices = random.sample(range(len(image_paths)), min(n_images, len(image_paths)))
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]
    else:
        n_images = len(image_paths)
    
    n_rows = (n_images + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 2, n_rows * 2))

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

MEASURES = {
    "fisher": compute_between_within_metrics,
    "dt": compute_decision_tree_metrics,
    "knn": compute_knn_metrics,
    "auc": compute_patch_auc_metrics,
    "proto": compute_class_prototype_metrics,
    "evidence": compute_class_evidence,
    "interaction": compute_interaction_aware_metrics
}

def patch_discriminative_maps(
    image_paths,
    labels,
    image_size=(64, 64),
    patch_size=8,
    overlap=True,
    descriptor_method="all",
    measure_method="fisher",
    aggregation=None,
    measure_params=None
):
    """
    Compute class discriminability maps.

    Parameters
    ----------
    image_paths : list
        Paths to images.

    labels : list
        Class labels.

    image_size : tuple
        Image size.

    patch_size : int
        Patch size.

    overlap : bool
        Whether to use overlapping patches.

    descriptor_method : str
        Descriptor extractor.

    measure_method : str
        Discriminability measure.

    aggregation : dict or None
        Optional descriptor aggregation.

        Examples
        --------
        None

        {"method": "mean",
         "kernel_size": 3}

        {"method": "gaussian",
         "sigma": 1.5}

        {"method": "max",
         "kernel_size": 5}
    """


    visualize_images(image_paths, labels, image_size=image_size, n_cols=5, n_images=10)


    # --------------------------------------------------------
    # Patch grid
    # --------------------------------------------------------

    stride = patch_size if not overlap else patch_size // 2

    H, W = image_size

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    # --------------------------------------------------------
    # Extract descriptors
    # --------------------------------------------------------

    features = extract_features(
        image_paths=image_paths,
        image_size=image_size,
        descriptor_method=descriptor_method,
        patch_size=patch_size,
        overlap=overlap,
    )

    # --------------------------------------------------------
    # Optional spatial aggregation
    # --------------------------------------------------------

    if aggregation is not None:

        aggregation_params = {
            k: v
            for k, v in aggregation.items()
            if k != "name"
        }

        features = aggregate_descriptors(
            features,
            n_rows=n_rows,
            n_cols=n_cols,
            **aggregation_params,
        )

    # --------------------------------------------------------
    # Compute discriminability
    # --------------------------------------------------------

    metrics = MEASURES[measure_method](
        features,
        labels,
        **measure_params
    )

    # --------------------------------------------------------
    # Convert vectors to maps
    # --------------------------------------------------------

    return metrics_to_maps(
        metrics,
        n_rows,
        n_cols,
    )




def global_discriminative_descriptors(
    image_paths,
    labels,
    image_size=(256, 256),
    descriptor_method="all",
    measure_method="fisher"
):

    features = extract_features(
        image_paths=image_paths,
        image_size=image_size,
        descriptor_method=descriptor_method,
        patch_size=None
    )

    metrics = MEASURES[measure_method](
        features,
        labels,
    )

    return metrics_to_scalars(metrics)