
import numpy as np

from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors



def compute_normalized_matrices(X, y):
    '''
    Compute the normalized within-class scatter matrix (S_w_hat)
    and the normalized between-class scatter matrix (S_b_hat).
    '''

    n_samples, n_features = X.shape
    classes = np.unique(y)

    global_mean = np.mean(X, axis=0)

    S_w_hat = np.zeros((n_features, n_features))
    S_b_hat = np.zeros((n_features, n_features))

    total_samples = n_samples

    for cls in classes:

        class_mask = (y == cls)
        X_class = X[class_mask]

        m_i = len(X_class)

        class_mean = np.mean(X_class, axis=0)

        centered_class = X_class - class_mean
        S_w_hat += (1 / m_i) * centered_class.T @ centered_class
        mean_diff = class_mean - global_mean

        weight = m_i / total_samples
        S_b_hat += weight * np.outer(mean_diff, mean_diff)

    return S_w_hat, S_b_hat


def compute_m_sep_direct(S_w_hat, S_b_hat):
    '''
    Auxiliary function to compute M_sep directly from
    S_w_hat and S_b_hat.
    '''

    try:

        eigenvalues, eigenvectors = eigh(S_b_hat, S_w_hat)
        max_idx = np.argmax(eigenvalues)
        m_sep = eigenvalues[max_idx]

        return m_sep

    except np.linalg.LinAlgError:

        S_w_pinv = np.linalg.pinv(S_w_hat)
        matrix = S_w_pinv @ S_b_hat
        eigenvalues = np.linalg.eigvals(matrix)
        m_sep = np.max(np.real(eigenvalues))

        return m_sep


def compute_m_var(S_w_hat, num_classes, dim):
    '''
    Auxiliary function to compute M_var based on
    the within-class scatter matrix.
    '''

    eigenvalues = np.linalg.eigvalsh(S_w_hat)
    lambda_min = np.min(eigenvalues)
    m_var = lambda_min / (num_classes * dim)

    return m_var




def knn_density_estimation(query_points, reference_points, k_neighbors=5):

    if(len(reference_points) < k_neighbors):
        k = len(reference_points)
    else:
        k = k_neighbors

    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')

    knn.fit(reference_points)
    distances, _ = knn.kneighbors(query_points)
    d = reference_points.shape[1]

    volumes = (distances[:, -1] * 2) ** d
    volumes = np.maximum(volumes, 1e-10)

    n_ref = len(reference_points)
    densities = k / (n_ref * volumes)

    return densities


def compute_pairwise_similarity(embeddings_i, embeddings_j, n_samples=50):

    inxs = np.random.choice(len(embeddings_i), min(n_samples, len(embeddings_i)), replace=False)
    monte_carlo_samples = embeddings_i[inxs]
    probabilities = knn_density_estimation( monte_carlo_samples, embeddings_j )
    similarity = np.mean(probabilities)

    return similarity


def compute_similarity_matrix_S(data, labels, class_labels, n_samples=50):
    '''
    Compute similarity matrix S.
    '''

    num_classes = len(class_labels)
    similarity_matrix_S = np.zeros((num_classes, num_classes))

    for i in range(num_classes):

        for j in range(num_classes):

            embeddings_i = data[labels == class_labels[i]]
            embeddings_j = data[labels == class_labels[j]]

            similarity_matrix_S[i, j] = compute_pairwise_similarity( embeddings_i,embeddings_j,   n_samples=n_samples )

    return similarity_matrix_S


def compute_adjacency_matrix_W(similarity_matrix_S):

    size = similarity_matrix_S.shape[0]
    adjacency_matrix_W = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if(i == j):
                adjacency_matrix_W[i, j] = 1.0

            else:

                numerator = np.sum(np.abs(similarity_matrix_S[i, :] - similarity_matrix_S[j, :]))
                denominator = np.sum(np.abs(similarity_matrix_S[i, :] + similarity_matrix_S[j, :]))

                if(denominator == 0):
                    adjacency_matrix_W[i, j] = 0.0
                else:
                    adjacency_matrix_W[i, j] = 1.0 - (numerator / denominator)

    adjacency_matrix_W = (adjacency_matrix_W + adjacency_matrix_W.T) / 2

    return adjacency_matrix_W


def compute_laplacian_matrix_L(adjacency_matrix_W):

    degree_matrix_D = np.diag(np.sum(adjacency_matrix_W, axis=1))
    laplacian_matrix_L = degree_matrix_D - adjacency_matrix_W

    return laplacian_matrix_L, degree_matrix_D


def compute_spectrum(laplacian_matrix_L):

    eigenvalues, eigenvectors = eigh(laplacian_matrix_L)

    sort_idx = np.argsort(eigenvalues)

    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    return eigenvalues, eigenvectors



def compute_csg_complexity(eigenvalues):

    n = len(eigenvalues)

    if(n < 2):
        return 0

    normalized_eigengaps = np.zeros(n - 1)

    for i in range(n - 1):
        normalized_eigengaps[i] = ( (eigenvalues[i + 1] - eigenvalues[i])/ (n - i)  )

    cumulative_max = np.zeros_like(normalized_eigengaps)
    current_max = 0

    for i in range(len(normalized_eigengaps)):
        current_max = max(current_max, normalized_eigengaps[i])
        cumulative_max[i] = current_max

    csg = np.sum(cumulative_max)

    return csg


def compute_AULS_complexity(eigenvalues):

    n = len(eigenvalues)

    if(n < 2):
        return 0

    normalized_gaps = np.zeros(n - 1)

    for i in range(n - 1):
        normalized_gaps[i] = ( (eigenvalues[i + 1] - eigenvalues[i]) / (n - i))

    auls = np.sum(normalized_gaps)

    return auls


def compute_cmsAULS_complexity(eigenvalues):

    n = len(eigenvalues)

    if(n < 2):
        return 0

    normalized_eigengaps = np.zeros(n - 1)

    for i in range(n - 1):
        normalized_eigengaps[i] = ( (eigenvalues[i + 1] ** 2 - eigenvalues[i] ** 2) / (2 * (n - i)))

    cumulative_max = np.zeros_like(normalized_eigengaps)
    current_max = 0

    for i in range(len(normalized_eigengaps)):
        current_max = max(current_max, normalized_eigengaps[i])
        cumulative_max[i] = current_max

    cmsAULS = np.sum(cumulative_max)

    return cmsAULS