# overlap_measures.py


import numpy as np
from pycol_complexity import complexity as pycol_complexity

from .overlap_utils import compute_normalized_matrices,compute_m_sep_direct,compute_m_var,compute_similarity_matrix_S,compute_adjacency_matrix_W,compute_laplacian_matrix_L,compute_spectrum,compute_csg_complexity,compute_AULS_complexity,compute_cmsAULS_complexity

def m_var_measure(embeddings, labels):

    S_w_hat, S_b_hat = compute_normalized_matrices(embeddings, labels)

    m_var = compute_m_var(S_w_hat,len(np.unique(labels)),embeddings.shape[1])

    return m_var


def m_sep_measure(embeddings, labels):

    S_w_hat, S_b_hat = compute_normalized_matrices(embeddings, labels)

    m_sep = compute_m_sep_direct(S_w_hat, S_b_hat)

    return m_sep


def tabular_measure(embeddings, labels, measure='kdn'):

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