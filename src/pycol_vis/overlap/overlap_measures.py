# overlap_measures.py


import numpy as np
from pycol_complexity import complexity as pycol_complexity

from .overlap_utils import compute_normalized_matrices,compute_m_sep_direct,compute_m_var,compute_similarity_matrix_S,compute_adjacency_matrix_W,compute_laplacian_matrix_L,compute_spectrum,compute_csg_complexity,compute_AULS_complexity,compute_cmsAULS_complexity

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
    S_w_hat, S_b_hat = compute_normalized_matrices(embeddings, labels)

    m_sep = compute_m_sep_direct(S_w_hat, S_b_hat)

    return m_sep


def tabular_measure(embeddings, labels, measure='kdn'):
    '''
    Calculate overlap measures using the pycol complexity libray.

    A lower value of the meaure indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.

    Parameters:
    - embeddings (np.ndarray): The feature embeddings of the samples in the dataset.
    - labels (np.ndarray): The class labels corresponding to each sample in the dataset.
    - measure (str): The complexity measure to calculate. Options are 'n2', 'kdn', or 'lsc'.

    Returns:
    - float: The calculated complexity measure value for the given embeddings and labels.
    '''
     

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

    else:
        raise ValueError("Measure must be one of 'n2', 'kdn', or 'lsc'.")

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


    similarity_matrix_S = compute_similarity_matrix_S(embeddings,labels,np.unique(labels),n_samples=n_samples)

    W = compute_adjacency_matrix_W(similarity_matrix_S)

    L, D = compute_laplacian_matrix_L(W)

    eigenvalues, eigenvectors = compute_spectrum(L)

    auls = compute_AULS_complexity(eigenvalues)

    return auls


def csg_measure(embeddings, labels, n_samples=50, auls=False):

    similarity_matrix_S = compute_similarity_matrix_S(embeddings,labels,np.unique(labels),n_samples=n_samples)

    W = compute_adjacency_matrix_W(similarity_matrix_S)
    L, D = compute_laplacian_matrix_L(W)

    eigenvalues, eigenvectors = compute_spectrum(L)

    if(auls):
        measure = compute_cmsAULS_complexity(eigenvalues)

    else:
        measure = compute_csg_complexity(eigenvalues)

    return measure