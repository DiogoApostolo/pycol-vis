
import numpy as np

from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.preprocessing import StandardScaler
import cv2

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from sklearn.decomposition import PCA

import torch

from PIL import Image

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

import numpy as np


from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


from scipy.ndimage import (
    uniform_filter,
    gaussian_filter,
    maximum_filter,
)


def validate_inputs(embeddings, labels):
    if(embeddings.shape[0] != len(labels)):
        raise ValueError("Number of embeddings must match number of labels.")
    
    
    if(len(np.unique(labels)) < 2):
        raise ValueError("There must be at least 2 classes in the dataset to calculate the CSG measure.")
    
    if(embeddings.shape[1] < 2):
        raise ValueError("Embeddings must have at least 2 dimensions to calculate the CSG measure.")
    
    if(embeddings.shape[0] < 10):
        raise ValueError("There must be at least 10 samples in the dataset to calculate the CSG measure.")

    if(embeddings.shape[0] < 100):
        print("Warning: The dataset contains a small number of samples, which may lead to less stable results. Consider using a larger dataset for more reliable results.")


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



def average_knn_distance(query_points, reference_points, k_neighbors=5):
    '''
    Compute the average k-nearest neighbor distance between query points and reference points.

    Parameters:
    - query_points (np.ndarray): Points for which nearest neighbors will be queried.
    - reference_points (np.ndarray): Reference points used to fit the nearest neighbor structure.
    - k_neighbors (int): Number of nearest neighbors to consider.

    Returns:
    - np.ndarray: Average k-nearest neighbor distance for each query point.
    '''

    if(len(reference_points) < k_neighbors):
        k = len(reference_points)
    else:
        k = k_neighbors

    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')

    knn.fit(reference_points)

    distances, _ = knn.kneighbors(query_points)
    avg_distances = np.mean(distances, axis=1)

    return avg_distances


def knn_density_estimation(query_points, reference_points, k_neighbors=5,normalize_density=True, log_scale=False):

    if(len(reference_points) < k_neighbors):
        k = len(reference_points)
    else:
        k = k_neighbors

    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')

    knn.fit(reference_points)

    distances, _ = knn.kneighbors(query_points)

    d = reference_points.shape[1]

    if(normalize_density):
        log_volumes = np.log((distances[:, -1] * 2) + 1e-12)
    else:
        log_volumes = d * np.log((distances[:, -1] * 2) + 1e-12)

    log_densities = np.log(k) - np.log(len(reference_points)) - log_volumes
    log_densities = log_densities - np.max(log_densities)
    if(log_scale):
        return log_densities

    densities = np.exp(log_densities)

    return densities


def compute_pairwise_similarity(embeddings_i, embeddings_j, n_samples=50,normalize_density=False,use_distance=False,sigma=None):

    inxs = np.random.choice(len(embeddings_i), min(n_samples, len(embeddings_i)), replace=False)
    monte_carlo_samples = embeddings_i[inxs]
    
    if(not use_distance):
        probabilities = knn_density_estimation( monte_carlo_samples, embeddings_j, normalize_density=False, log_scale=True)
        similarity = np.mean(probabilities)
        similarity = np.exp(similarity)  
    else:
        

        distance = average_knn_distance(monte_carlo_samples, embeddings_j)

        if(sigma == None):
            sigma = np.mean(distance)

        similarity = np.mean(np.exp(-(distance)))


    return similarity


def compute_similarity_matrix_S(data, labels, class_labels, n_samples=50, normalize_density=True):
    '''
    Compute similarity matrix S.
    '''

    num_classes = len(class_labels)
    similarity_matrix_S = np.zeros((num_classes, num_classes))

    for i in range(num_classes):

        for j in range(num_classes):

            embeddings_i = data[labels == class_labels[i]]
            embeddings_j = data[labels == class_labels[j]]

            similarity_matrix_S[i, j] = compute_pairwise_similarity( embeddings_i,embeddings_j,   n_samples=n_samples, normalize_density=normalize_density )

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


def compute_laplacian_matrix_L(adjacency_matrix_W, normalized=False):

    degree_matrix_D = np.diag(np.sum(adjacency_matrix_W, axis=1))

    

    if(normalized):
        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix_D) + 1e-10))
        laplacian_matrix_L = (np.eye(len(adjacency_matrix_W))- d_inv_sqrt @ adjacency_matrix_W @ d_inv_sqrt)
    else:
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




def build_knn_graph(X, k=5):
    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean"
    )

    nbrs.fit(X)

    _, indices = nbrs.kneighbors(X)

    return indices[:, 1:]

def cross_class_edge_ratio(X, y, k=5, per_class=False):
    neighbors = build_knn_graph(X, k)


    if(per_class):
        total_edges = np.zeros(len(np.unique(y)))
        cross_edges = np.zeros(len(np.unique(y)))

        for i in range(len(X)):
            for j in neighbors[i]:
                cls = np.where(np.unique(y) == y[i])[0][0]
                total_edges[cls] += 1

                if (y[i] != y[j]):
                    cross_edges[cls] += 1
        
        return cross_edges / (total_edges + 1e-10)

    else:
        total_edges = 0
        cross_edges = 0

        for i in range(len(X)):
            for j in neighbors[i]:
                total_edges += 1

                if (y[i] != y[j]):
                    cross_edges += 1

        return cross_edges / (total_edges + 1e-10)




def mst_entropy(X):

    X = StandardScaler().fit_transform(X)

    n = len(X)

    if (n < 2):
        return 0.0

    D = squareform(pdist(X))

    mst = minimum_spanning_tree(D)

    L = mst.sum()

    return np.log(L + 1e-10) / np.log(n)

def conditional_entropy(X, y):
    classes = np.unique(y)

    H = 0.0

    for c in classes:
        Xc = X[y == c]

        if (len(Xc) < 2):
            continue

        Hc = mst_entropy(Xc)

        H += (len(Xc) / len(X)) * Hc

    return H



def mst_renyi_entropy(X, gamma=1.0):
    n, d = X.shape

    if (n < 2):
        return 0.0

    alpha = (d - gamma) / d

    D = squareform(pdist(X))

    mst = minimum_spanning_tree(D)

    L = mst.sum()

    if (L <= 0):
        return 0.0

    H = np.log(L / (n ** alpha)) / (1.0 - alpha)

    return float(H)


def conditional_mst_entropy(X, y, gamma=1.0, per_class=False):
    classes = np.unique(y)

    H_all = mst_renyi_entropy(X, gamma=gamma)

    if(per_class):
        H = np.zeros(len(classes))

        cls_count = 0
        for c in classes:
            Xc = X[y == c]

            if (len(Xc) < 2):
                H[c] = np.nan
                continue

            Hc = mst_renyi_entropy(Xc, gamma=gamma)

            H[cls_count] = Hc / H_all
            cls_count += 1

        return H
    else:
        S = 0.0

        for c in classes:
            Xc = X[y == c]

            if (len(Xc) < 2):
                continue

            pc = len(Xc) / len(X)
            Hc = mst_renyi_entropy(Xc,gamma=gamma)

            S += (pc * Hc)/H_all

        return float(S)
    

def patch_descriptor(patch):

    patch = patch.astype(np.float32)

    # ---------- intensity ----------
    gray = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180

    hist, _ = np.histogram(
        ang,
        bins=8,
        range=(0, 180),
        weights=mag
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)

    # ---------- colour ----------
    lab = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)

    colour_stats = np.array([
        np.mean(lab[:, :, 0]),
        np.std(lab[:, :, 0]),
        np.mean(lab[:, :, 1]),
        np.std(lab[:, :, 1]),
        np.mean(lab[:, :, 2]),
        np.std(lab[:, :, 2]),
    ], dtype=np.float32)

    descriptor = np.concatenate([
        np.array([
            np.mean(gray),
            np.std(gray),
            np.mean(mag),
            np.std(mag)
        ], dtype=np.float32),
        hist,
        colour_stats
    ])

    return descriptor
    
def patch_descriptor_old(patch):
    patch = patch.astype(np.float32)

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180

    hist, _ = np.histogram(
        ang,
        bins=8,
        range=(0, 180),
        weights=mag
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)

    descriptor = np.concatenate([
        np.array([
            np.mean(patch),
            np.std(patch),
            np.mean(mag),
            np.std(mag)
        ], dtype=np.float32),
        hist
    ])

    return descriptor

    


def intensity_descriptor(patch):
    patch = patch.astype(np.float32)

    gray = cv2.cvtColor(
        patch.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    return np.array([
        np.mean(gray),
        np.std(gray)
    ], dtype=np.float32)

def gradient_descriptor(patch):
    patch = patch.astype(np.float32)

    gray = cv2.cvtColor(
        patch.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)

    return np.array([
        np.mean(mag),
        np.std(mag)
    ], dtype=np.float32)


def hog_descriptor(patch):
    patch = patch.astype(np.float32)

    gray = cv2.cvtColor(
        patch.astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180

    hist, _ = np.histogram(
        ang,
        bins=8,
        range=(0, 180),
        weights=mag
    )

    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-10)

    return hist

def color_descriptor(patch):
    patch = patch.astype(np.float32)

    lab = cv2.cvtColor(
        patch.astype(np.uint8),
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    return np.array([
        np.mean(lab[:, :, 0]),
        np.std(lab[:, :, 0]),
        np.mean(lab[:, :, 1]),
        np.std(lab[:, :, 1]),
        np.mean(lab[:, :, 2]),
        np.std(lab[:, :, 2]),
    ], dtype=np.float32)


_resnet = None
_transform = None
_device = None



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

@torch.no_grad()
def learned_descriptor(patch):

    global _resnet, _transform, _device

    if _resnet is None:

        weights = ResNet18_Weights.DEFAULT

        _transform = weights.transforms()

        _device = get_device()

        _resnet = resnet18(weights=weights)
        _resnet.fc = torch.nn.Identity()
        _resnet.eval()
        _resnet.to(_device)

    patch = patch.astype(np.uint8)
    patch = cv2.resize(patch, (224, 224))
    patch = Image.fromarray(patch)

    tensor = _transform(patch).unsqueeze(0).to(_device)

    descriptor = _resnet(tensor)

    return descriptor.squeeze(0).cpu().numpy().astype(np.float32)


def extract_features(
    image_paths,
    image_size=(64, 64),
    descriptor_method="all",
    patch_size=None,
    overlap=True
):

    images = []

    for path in image_paths:

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, image_size)

        images.append(img.astype(np.float32))

    images = np.asarray(images)

    H, W = image_size

    features = []

    # ----------------------------------------------------------
    # GLOBAL DESCRIPTORS
    # ----------------------------------------------------------

    if patch_size is None:

        for img in images:

            if descriptor_method == "all":
                feat = patch_descriptor(img)

            elif descriptor_method == "mean":
                feat = np.array([np.mean(img)], dtype=np.float32)

            elif descriptor_method == "intensity":
                feat = intensity_descriptor(img)

            elif descriptor_method == "gradient":
                feat = gradient_descriptor(img)

            elif descriptor_method == "hog":
                feat = hog_descriptor(img)

            elif descriptor_method == "color":
                feat = color_descriptor(img)

            elif descriptor_method == "learned":
                feat = learned_descriptor(img)

            else:
                raise ValueError("Unknown descriptor.")

            # one location
            features.append([feat])

    # ----------------------------------------------------------
    # PATCH DESCRIPTORS
    # ----------------------------------------------------------

    else:

        stride = patch_size if not overlap else patch_size // 2

        for img in images:

            feats = []

            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):

                    patch = img[
                        y:y + patch_size,
                        x:x + patch_size
                    ]

                    if descriptor_method == "all":
                        feats.append(patch_descriptor(patch))

                    elif descriptor_method == "mean":
                        feats.append(
                            np.array([np.mean(patch)], dtype=np.float32)
                        )

                    elif descriptor_method == "intensity":
                        feats.append(intensity_descriptor(patch))

                    elif descriptor_method == "gradient":
                        feats.append(gradient_descriptor(patch))

                    elif descriptor_method == "hog":
                        feats.append(hog_descriptor(patch))

                    elif descriptor_method == "color":
                        feats.append(color_descriptor(patch))

                    elif descriptor_method == "learned":
                        feats.append(learned_descriptor(patch))

                    else:
                        raise ValueError("Unknown descriptor.")

            features.append(feats)

    return np.asarray(features)


def metrics_to_maps(metrics, n_rows, n_cols):

    maps = {}

    for metric in metrics:

        maps[metric] = {}

        for c in metrics[metric]:

            maps[metric][c] = metrics[metric][c].reshape(
                n_rows,
                n_cols
            )

    return maps

def metrics_to_scalars(metrics):

    results = {}

    for metric in metrics:

        results[metric] = {}

        for c in metrics[metric]:

            results[metric][c] = float(metrics[metric][c][0])

    return results


def compute_class_evidence(
    features,
    labels,
    epsilon=1e-5
):
    """
    Compute class-specific evidence maps.

    For each class c and spatial location p:

        L(x,c) =
            log P(x_p | c)
            -
            log P(x_p | not c)

    Positive values indicate evidence for class c.
    Negative values indicate evidence against class c.

    The feature dimensions are globally standardized before
    estimating the class distributions.

    Returns class-level maps by averaging evidence over the
    images belonging to each class.

    Parameters
    ----------
    features : np.ndarray
        Shape:
            (n_samples, n_locations, feature_dim)

    labels : array-like
        Class labels.

    epsilon : float
        Numerical regularization.

    Returns
    -------
    metrics : dict

        LOG_LIKELIHOOD_RATIO[class]
            Raw signed log-likelihood ratio.

        EVIDENCE_FOR[class]
            Positive evidence supporting the class.

        EVIDENCE_AGAINST[class]
            Positive magnitude of evidence against the class.

        EVIDENCE_STRENGTH[class]
            Bounded evidence score in [0, 1].

        STANDARDIZED_FEATURES
            Standardized feature tensor.
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(labels)

    classes = np.unique(labels)

    if len(classes) < 2:

        raise ValueError(
            "compute_class_evidence requires "
            "at least two classes."
        )

    # ---------------------------------------------------------
    # Standardize feature dimensions globally
    # ---------------------------------------------------------
    #
    # Standardization is performed independently for each
    # feature dimension, across images and spatial locations.
    #
    # This prevents very large feature scales from dominating
    # the likelihood calculation.
    #
    # ---------------------------------------------------------

    feature_mean = np.mean(
        features,
        axis=(0, 1),
        keepdims=True
    )

    feature_std = np.std(
        features,
        axis=(0, 1),
        keepdims=True
    )

    feature_std = np.maximum(
        feature_std,
        epsilon
    )

    features = (
        features - feature_mean
    ) / feature_std

    # ---------------------------------------------------------
    # Output structure
    # ---------------------------------------------------------

    metrics = {
        "LOG_LIKELIHOOD_RATIO": {},
        "EVIDENCE_FOR": {},
        "EVIDENCE_AGAINST": {},
        "EVIDENCE_STRENGTH": {}
    }

    n_samples = features.shape[0]
    n_locations = features.shape[1]

    # ---------------------------------------------------------
    # Process each class
    # ---------------------------------------------------------

    for class_name in classes:

        mask_class = (
            labels == class_name
        )

        mask_other = (
            labels != class_name
        )

        features_class = (
            features[mask_class]
        )

        features_other = (
            features[mask_other]
        )

        if len(features_class) < 2:

            print(
                f"Skipping {class_name}: "
                f"not enough class samples."
            )

            continue

        if len(features_other) < 2:

            print(
                f"Skipping {class_name}: "
                f"not enough non-class samples."
            )

            continue

        # -----------------------------------------------------
        # Image-level signed evidence
        # -----------------------------------------------------

        image_log_likelihood_ratio = np.zeros(
            (n_samples, n_locations),
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Process each patch
        # -----------------------------------------------------

        for p in range(n_locations):

            X_class = (
                features_class[:, p, :]
            )

            X_other = (
                features_other[:, p, :]
            )

            X = features[:, p, :]

            # -------------------------------------------------
            # Class distribution
            # -------------------------------------------------

            mean_class = np.mean(
                X_class,
                axis=0
            )

            var_class = (
                np.var(
                    X_class,
                    axis=0
                )
                + epsilon
            )

            # -------------------------------------------------
            # Not-class distribution
            # -------------------------------------------------

            mean_other = np.mean(
                X_other,
                axis=0
            )

            var_other = (
                np.var(
                    X_other,
                    axis=0
                )
                + epsilon
            )

            # -------------------------------------------------
            # Log determinant
            #
            # For diagonal covariance:
            #
            # log |Sigma| = sum(log(var))
            #
            # -------------------------------------------------

            logdet_class = np.sum(
                np.log(var_class)
            )

            logdet_other = np.sum(
                np.log(var_other)
            )

            # -------------------------------------------------
            # Mahalanobis distance
            # -------------------------------------------------

            diff_class = (
                X - mean_class
            )

            diff_other = (
                X - mean_other
            )

            mahal_class = np.sum(
                (
                    diff_class ** 2
                ) / var_class,
                axis=1
            )

            mahal_other = np.sum(
                (
                    diff_other ** 2
                ) / var_other,
                axis=1
            )

            # -------------------------------------------------
            # Gaussian log likelihood
            #
            # Constant:
            #
            # -D/2 log(2*pi)
            #
            # cancels when taking the difference.
            # -------------------------------------------------

            log_prob_class = -0.5 * (
                mahal_class
                + logdet_class
            )

            log_prob_other = -0.5 * (
                mahal_other
                + logdet_other
            )

            # -------------------------------------------------
            # Signed evidence
            # -------------------------------------------------

            image_log_likelihood_ratio[:, p] = (
                log_prob_class
                - log_prob_other
            )

        # -----------------------------------------------------
        # Only images belonging to this class
        # -----------------------------------------------------
        #
        # This creates the class-level profile used by
        # metrics_to_maps().
        #
        # Shape:
        #
        #     (n_locations,)
        #
        # -----------------------------------------------------

        class_evidence = (
            image_log_likelihood_ratio[
                mask_class
            ]
        )

        # -----------------------------------------------------
        # Raw signed class evidence
        # -----------------------------------------------------

        mean_log_likelihood_ratio = np.mean(
            class_evidence,
            axis=0
        )

        # -----------------------------------------------------
        # Evidence FOR the class
        # -----------------------------------------------------

        evidence_for = np.maximum(
            class_evidence,
            0
        )

        evidence_for = np.mean(
            evidence_for,
            axis=0
        )

        # -----------------------------------------------------
        # Evidence AGAINST the class
        # -----------------------------------------------------

        evidence_against = np.maximum(
            -class_evidence,
            0
        )

        evidence_against = np.mean(
            evidence_against,
            axis=0
        )

        # -----------------------------------------------------
        # Bounded evidence strength
        # -----------------------------------------------------
        #
        # Instead of using sigmoid(raw LLR), which can saturate
        # immediately for very large likelihood ratios, first
        # use a symmetric normalization.
        #
        # This gives:
        #
        #       0   = strong evidence against class
        #      .5   = neutral
        #       1   = strong evidence for class
        #
        # -----------------------------------------------------

        scale = np.percentile(
            np.abs(
                class_evidence
            ),
            95
        )

        scale = max(
            scale,
            epsilon
        )

        normalized = (
            mean_log_likelihood_ratio
            / scale
        )

        normalized = np.clip(
            normalized,
            -1,
            1
        )

        evidence_strength = (
            normalized + 1
        ) / 2

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        metrics[
            "LOG_LIKELIHOOD_RATIO"
        ][class_name] = (
            mean_log_likelihood_ratio
        )

        metrics[
            "EVIDENCE_FOR"
        ][class_name] = (
            evidence_for
        )

        metrics[
            "EVIDENCE_AGAINST"
        ][class_name] = (
            evidence_against
        )

        metrics[
            "EVIDENCE_STRENGTH"
        ][class_name] = (
            evidence_strength
        )

    return metrics


def compute_class_prototype_metrics(
    features,
    labels
):
    labels = np.asarray(labels)

    classes = np.unique(labels)

    metrics = {
        "PROTOTYPE": {},
        "BACKGROUND": {},
        "CLASS_EVIDENCE": {}
    }

    # ---------------------------------------------------------
    # General dataset prototype
    # ---------------------------------------------------------

    background_prototype = np.mean(
        features,
        axis=0
    )

    # ---------------------------------------------------------
    # Compute class-specific prototypes
    # ---------------------------------------------------------

    class_prototypes = {}

    for c in classes:

        feats_c = features[
            labels == c
        ]

        if len(feats_c) == 0:
            continue

        class_prototypes[c] = np.mean(
            feats_c,
            axis=0
        )

    # ---------------------------------------------------------
    # Compute class-specific evidence
    # ---------------------------------------------------------

    for c in classes:

        feats_c = features[
            labels == c
        ]

        if len(feats_c) == 0:
            continue

        prototype = class_prototypes[c]

        n_locations = feats_c.shape[1]

        prototype_similarity = np.zeros(
            n_locations,
            dtype=np.float32
        )

        background_similarity = np.zeros(
            n_locations,
            dtype=np.float32
        )

        class_evidence = np.zeros(
            n_locations,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Each spatial location
        # -----------------------------------------------------

        for p in range(n_locations):

            X = feats_c[:, p, :]

            class_vector = prototype[p]

            background_vector = (
                background_prototype[p]
            )

            # -------------------------------------------------
            # Normalize
            # -------------------------------------------------

            class_norm = np.linalg.norm(
                class_vector
            )

            background_norm = np.linalg.norm(
                background_vector
            )

            if class_norm < 1e-10:
                continue

            if background_norm < 1e-10:
                continue

            class_vector = (
                class_vector /
                class_norm
            )

            background_vector = (
                background_vector /
                background_norm
            )

            # -------------------------------------------------
            # Similarity of class examples to prototype
            # -------------------------------------------------

            X_norm = np.linalg.norm(
                X,
                axis=1,
                keepdims=True
            )

            X_norm[X_norm < 1e-10] = 1e-10

            X_normalized = (
                X /
                X_norm
            )

            class_sim = np.mean(
                X_normalized @ class_vector
            )

            background_sim = np.mean(
                X_normalized @ background_vector
            )

            # -------------------------------------------------
            # Store
            # -------------------------------------------------

            prototype_similarity[p] = (
                class_sim
            )

            background_similarity[p] = (
                background_sim
            )

            class_evidence[p] = (
                class_sim -
                background_sim
            )

        metrics["PROTOTYPE"][c] = (
            prototype_similarity
        )

        metrics["BACKGROUND"][c] = (
            background_similarity
        )

        metrics["CLASS_EVIDENCE"][c] = (
            class_evidence
        )

    return metrics



def compute_patch_auc_metrics(
    features,
    labels,
    cv=5
):
    """
    Compute class-separability maps using patch-level ROC-AUC.

    For each spatial location, a classifier is trained using
    only the feature vector from that location.

    Parameters
    ----------
    features : np.ndarray
        Shape:
            (n_samples, n_locations, feature_dim)

    labels : array-like
        Class label for each sample.

    cv : int
        Number of cross-validation folds.

    Returns
    -------
    metrics : dict
        Dictionary containing the AUC map.

        metrics["AUC"] has shape:
            (n_locations,)
    """

    labels = np.asarray(labels)

    classes = np.unique(labels)

    if len(classes) != 2:
        raise ValueError(
            "compute_patch_auc_metrics currently "
            "requires exactly two classes."
        )

    n_samples, n_locations, feature_dim = (
        features.shape
    )

    metrics = {
        "AUC": np.full(
            n_locations,
            0.5,
            dtype=np.float32
        )
    }

    # ---------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------

    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    # ---------------------------------------------------------
    # Compute AUC independently for every location
    # ---------------------------------------------------------

    for p in range(n_locations):

        X_patch = features[:, p, :]

        # -----------------------------------------------------
        # Classifier using ONLY this patch
        # -----------------------------------------------------

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000
            )
        )

        # -----------------------------------------------------
        # Cross-validated ROC-AUC
        # -----------------------------------------------------

        scores = cross_val_score(
            classifier,
            X_patch,
            labels,
            cv=cv_splitter,
            scoring="roc_auc"
        )

        metrics["AUC"][p] = np.mean(
            scores
        )

    return metrics

def compute_between_within_metrics(
    features,
    labels
):

    labels = np.asarray(labels)
    classes = np.unique(labels)

    metrics = {
        "R": {},
        "R_MAH": {},
        "B": {},
        "B_MAH": {},
        "W": {},
        "B_W": {}
    }

    for c in classes:

        feats_c = features[labels == c]

        if len(feats_c) < 2:
            continue

        mean_c = np.mean(feats_c, axis=0)
        var_c = np.var(feats_c, axis=0)

        n_locations = mean_c.shape[0]

        between = np.full(n_locations, np.inf, dtype=np.float32)
        between_mah = np.full(n_locations, np.inf, dtype=np.float32)

        for other in classes:

            if other == c:
                continue

            feats_other = features[labels == other]

            mean_other = np.mean(feats_other, axis=0)

            # Euclidean between-class distance
            euclidean = np.sum(
                (mean_c - mean_other) ** 2,
                axis=1
            )

            between = np.minimum(
                between,
                euclidean
            )

            # Mahalanobis distance
            for p in range(n_locations):

                X1 = feats_c[:, p, :]
                X2 = feats_other[:, p, :]

                cov1 = np.cov(X1, rowvar=False)
                cov2 = np.cov(X2, rowvar=False)

                pooled = (
                    ((len(X1) - 1) * cov1)
                    + ((len(X2) - 1) * cov2)
                ) / (len(X1) + len(X2) - 2)

                pooled += 1e-5 * np.eye(
                    pooled.shape[0]
                )

                diff = mean_c[p] - mean_other[p]

                try:

                    d2 = diff @ np.linalg.solve(
                        pooled,
                        diff
                    )

                    between_mah[p] = min(
                        between_mah[p],
                        d2
                    )

                except np.linalg.LinAlgError:
                    pass

        within = np.sum(
            var_c,
            axis=1
        )

        metrics["B"][c] = between
        metrics["B_MAH"][c] = between_mah
        metrics["W"][c] = within

        metrics["B_W"][c] = (
            between /
            (within + 1e-10)
        )

        metrics["R"][c] = (
            between /
            (between + within + 1e-10)
        )

        metrics["R_MAH"][c] = (
            between_mah /
            (between_mah + within + 1e-10)
        )

    return metrics




def compute_knn_metrics(
    features,
    labels,
    k=5,
    samples_per_class=500,
    random_state=42
):

    labels = np.asarray(labels)

    # --------------------------------------------------------
    # Stratified sampling
    # --------------------------------------------------------

    if samples_per_class is not None:

        rng = np.random.default_rng(random_state)

        selected_indices = []

        for c in np.unique(labels):

            class_indices = np.where(labels == c)[0]

            n = min(samples_per_class, len(class_indices))

            sampled = rng.choice(
                class_indices,
                size=n,
                replace=False
            )

            selected_indices.extend(sampled)

        selected_indices = np.array(selected_indices)

        features = features[selected_indices]
        labels = labels[selected_indices]

    classes = np.unique(labels)

    n_locations = features.shape[1]

    metrics = {
        "KNN": {}
    }

    for c in classes:

        metrics["KNN"][c] = np.zeros(
            n_locations,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Compute kNN discriminability
    # --------------------------------------------------------

    for p in range(n_locations):

        X = features[:, p, :]

        nbrs = NearestNeighbors(
            n_neighbors=k + 1
        ).fit(X)

        indices = nbrs.kneighbors(
            return_distance=False
        )

        # Remove self
        indices = indices[:, 1:]

        neighbor_labels = labels[indices]

        # ----------------------------------------------------
        # Majority-vote prediction
        # ----------------------------------------------------

        predictions = np.empty_like(labels)

        for i in range(len(labels)):

            votes = neighbor_labels[i]

            values, counts = np.unique(
                votes,
                return_counts=True
            )

            predictions[i] = values[np.argmax(counts)]

        correct = (
            predictions == labels
        ).astype(np.float32)

        # ----------------------------------------------------
        # Per-class score
        # ----------------------------------------------------

        for c in classes:

            metrics["KNN"][c][p] = (
                correct[labels == c]
                .mean()
            )

    return metrics




def aggregate_descriptors(
    features,
    n_rows,
    n_cols,
    method="mean",
    kernel_size=3,
    sigma=1.0,
):
    """
    Spatially aggregate patch descriptors.

    Parameters
    ----------
    features : ndarray
        Shape (N_images, N_patches, D)

    n_rows, n_cols : int
        Spatial patch layout.

    method : str
        "mean"
        "gaussian"
        "max"

    kernel_size : int
        Window size.

    sigma : float
        Used for gaussian aggregation.

    Returns
    -------
    ndarray
        Shape (N_images, N_patches, D)
    """

    n_images, n_patches, D = features.shape

    aggregated = np.empty_like(features)

    for i in range(n_images):

        fmap = features[i].reshape(
            n_rows,
            n_cols,
            D,
        )

        out = np.empty_like(fmap)

        for d in range(D):

            channel = fmap[:, :, d]

            if method == "mean":

                out[:, :, d] = uniform_filter(
                    channel,
                    size=kernel_size,
                    mode="nearest",
                )

            elif method == "gaussian":

                out[:, :, d] = gaussian_filter(
                    channel,
                    sigma=sigma,
                    mode="nearest",
                )

            elif method == "max":

                out[:, :, d] = maximum_filter(
                    channel,
                    size=kernel_size,
                    mode="nearest",
                )

            else:
                raise ValueError(
                    f"Unknown aggregation method: {method}"
                )

        aggregated[i] = out.reshape(
            n_patches,
            D,
        )

    return aggregated


def fit_patch_pca(
    features,
    n_components=None
):
    """
    Apply PCA independently to the feature dimension
    of patch descriptors.

    Expected shape:

        (n_samples, n_patches, feature_dim)
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    if n_components is None:
        return features, None

    if features.ndim != 3:

        raise ValueError(
            "Interaction-aware PCA expects features "
            "with shape "
            "(n_samples, n_patches, feature_dim). "
            f"Got {features.shape}."
        )

    n_samples, n_patches, feature_dim = (
        features.shape
    )

    if n_components < 1:
        raise ValueError(
            "n_components must be >= 1."
        )

    if n_components > feature_dim:
        raise ValueError(
            f"n_components={n_components} "
            f"cannot exceed feature dimension "
            f"{feature_dim}."
        )

    X = features.reshape(
        -1,
        feature_dim
    )

    pca = PCA(
        n_components=n_components
    )

    X = pca.fit_transform(
        X
    )

    transformed = X.reshape(
        n_samples,
        n_patches,
        n_components
    )

    return transformed, pca


# ============================================================
# Class statistics
# ============================================================

def compute_class_patch_statistics(
    features,
    labels,
    class_name
):
    """
    Compute mean and variance for every patch
    belonging to one class.

    Parameters
    ----------
    features : np.ndarray
        Shape:
            (n_samples, height, width, feature_dim)

    labels : array-like

    class_name :
        Class whose statistics are computed.

    Returns
    -------
    mean : np.ndarray
        Shape:
            (height, width, feature_dim)

    var : np.ndarray
        Shape:
            (height, width, feature_dim)

    samples : np.ndarray
        Original samples belonging to the class.
        Shape:
            (n_class_samples, height, width, feature_dim)
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    )

    samples = features[
        labels == class_name
    ]

    if len(samples) < 2:
        raise ValueError(
            f"Class '{class_name}' "
            "must contain at least two samples."
        )

    mean = np.mean(
        samples,
        axis=0
    )

    var = np.var(
        samples,
        axis=0
    )

    return (
        mean,
        var,
        samples
    )


# ============================================================
# Fisher discriminability from statistics
# ============================================================

def fisher_from_statistics(
    mean_a,
    var_a,
    mean_b,
    var_b,
    eps=1e-10
):
    """
    Compute Fisher-style discriminability for each patch.

    This is:

        B / W

    where

        B = ||mean_a - mean_b||^2

        W = sum(var_a + var_b)

    Parameters
    ----------
    mean_a, var_a : np.ndarray
        Class A statistics.

    mean_b, var_b : np.ndarray
        Class B statistics.

    Returns
    -------
    score : np.ndarray
        Spatial discriminability map.
    """

    between = np.sum(
        (
            mean_a -
            mean_b
        ) ** 2,
        axis=-1
    )

    within = np.sum(
        var_a +
        var_b,
        axis=-1
    )

    return (
        between /
        (within + eps)
    )


# ============================================================
# Joint Fisher discriminability
# ============================================================

def compute_joint_fisher(
    samples_a_p,
    samples_a_q,
    samples_b_p,
    samples_b_q,
    eps=1e-10
):
    """
    Compute Fisher discriminability when two patches
    are considered jointly.

    The feature vectors of the two patches are concatenated.

    Importantly, the covariance structure of the joint
    representation is retained.

    Parameters
    ----------
    samples_a_p : np.ndarray
        Class A features for patch p.

        Shape:
            (n_a, feature_dim)

    samples_a_q : np.ndarray
        Class A features for patch q.

    samples_b_p : np.ndarray
        Class B features for patch p.

    samples_b_q : np.ndarray
        Class B features for patch q.

    Returns
    -------
    score : float
    """

    joint_a = np.concatenate(
        [
            samples_a_p,
            samples_a_q
        ],
        axis=1
    )

    joint_b = np.concatenate(
        [
            samples_b_p,
            samples_b_q
        ],
        axis=1
    )

    mean_a = np.mean(
        joint_a,
        axis=0
    )

    mean_b = np.mean(
        joint_b,
        axis=0
    )

    var_a = np.var(
        joint_a,
        axis=0
    )

    var_b = np.var(
        joint_b,
        axis=0
    )

    between = np.sum(
        (
            mean_a -
            mean_b
        ) ** 2
    )

    within = np.sum(
        var_a +
        var_b
    )

    return float(
        between /
        (within + eps)
    )


# ============================================================
# Patch interaction
# ============================================================

def compute_patch_interaction(
    individual_a,
    individual_b,
    joint
):
    """
    Compute the interaction between two patches.

    Positive:
        complementary / synergistic information

    Negative:
        redundancy

    Near zero:
        little interaction
    """

    return (
        joint
        -
        0.5 * (
            individual_a +
            individual_b
        )
    )

# ============================================================
# Interaction reduction
# ============================================================

def reduce_interactions(
    interactions,
    method="mean",
    top_k=10
):
    """
    Reduce a pairwise interaction matrix to
    one score per patch.

    Parameters
    ----------
    interactions : np.ndarray
        Shape:
            (n_patches, n_patches)

    method : str
        "mean"
            Mean interaction with all other patches.

        "topk_mean"
            Mean of the strongest positive interactions.

        "max"
            Strongest positive interaction.

        "topk_abs"
            Mean of the strongest interactions by absolute
            magnitude.

    top_k : int
        Number of interactions used by topk_mean/topk_abs.

    Returns
    -------
    scores : np.ndarray
        Shape:
            (n_patches,)
    """

    interactions = np.asarray(
        interactions,
        dtype=np.float32
    )

    n_patches = (
        interactions.shape[0]
    )

    scores = np.zeros(
        n_patches,
        dtype=np.float32
    )

    for p in range(
        n_patches
    ):

        values = np.delete(
            interactions[p],
            p
        )

        if len(values) == 0:
            continue

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        if method == "mean":

            scores[p] = np.mean(
                values
            )

        # ----------------------------------------------------
        # Strongest positive interaction
        # ----------------------------------------------------

        elif method == "max":

            scores[p] = np.max(
                values
            )

        # ----------------------------------------------------
        # Mean of strongest positive interactions
        # ----------------------------------------------------

        elif method == "topk_mean":

            positive = values[
                values > 0
            ]

            if len(positive) == 0:

                scores[p] = 0.0

                continue

            k = min(
                top_k,
                len(positive)
            )

            indices = np.argpartition(
                positive,
                -k
            )[-k:]

            scores[p] = np.mean(
                positive[indices]
            )

        # ----------------------------------------------------
        # Strongest interactions by magnitude
        # ----------------------------------------------------

        elif method == "topk_abs":

            k = min(
                top_k,
                len(values)
            )

            indices = np.argpartition(
                np.abs(values),
                -k
            )[-k:]

            scores[p] = np.mean(
                values[indices]
            )

        else:

            raise ValueError(
                f"Unknown interaction reduction: "
                f"{method}"
            )

    return scores


# ============================================================
# Pairwise interaction map
# ============================================================


def compute_interaction_map(
    features,
    labels,
    class_a,
    class_b,
    interaction_reduction="topk_mean",
    usefulness_reduction="topk_mean",
    top_k=10,
    eps=1e-10
):
    """
    Compute pairwise patch interactions for two classes.

    Three quantities are computed for every patch:

        B
            Individual discriminability.

        I
            Interaction effect.

            Measures how much the joint representation
            differs from the individual usefulness of
            the two patches.

        C
            Combined usefulness.

            Measures how discriminative a patch is when
            combined with another patch.

    Parameters
    ----------
    features : np.ndarray
        Shape:

            (n_samples, n_patches, feature_dim)

    labels : array-like

    class_a, class_b :
        Classes being compared.

    interaction_reduction : str
        How to reduce the interaction of each patch
        across its partners.

        Supported by reduce_interactions():

            "mean"
            "max"
            "topk_mean"
            "topk_abs"

    usefulness_reduction : str
        How to reduce the joint usefulness of each patch
        across its partners.

        Supported by reduce_interactions().

    top_k : int
        Number of strongest partners used by top-k reductions.

    eps : float
        Numerical stability constant.

    Returns
    -------
    interaction_map : np.ndarray
        Shape:

            (height, width)

        Interaction / synergy / redundancy.

    individual_map : np.ndarray
        Shape:

            (height, width)

        Individual Fisher discriminability.

    usefulness_map : np.ndarray
        Shape:

            (height, width)

        Joint usefulness of each patch with its strongest
        partners.
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    )

    # --------------------------------------------------------
    # Validate feature representation
    # --------------------------------------------------------

    if features.ndim != 3:

        raise ValueError(
            "Expected features with shape "
            "(n_samples, n_patches, feature_dim). "
            f"Got {features.shape}."
        )

    n_samples, n_patches, feature_dim = (
        features.shape
    )

    # --------------------------------------------------------
    # Infer spatial grid
    # --------------------------------------------------------

    height, width = infer_patch_grid(
        n_patches
    )

    # --------------------------------------------------------
    # Extract class samples
    # --------------------------------------------------------

    Xa = features[
        labels == class_a
    ]

    Xb = features[
        labels == class_b
    ]

    if len(Xa) < 2 or len(Xb) < 2:

        raise ValueError(
            "Both classes must contain "
            "at least two samples."
        )

    # --------------------------------------------------------
    # Individual patch statistics
    # --------------------------------------------------------

    mean_a = np.mean(
        Xa,
        axis=0
    )

    mean_b = np.mean(
        Xb,
        axis=0
    )

    var_a = np.var(
        Xa,
        axis=0
    )

    var_b = np.var(
        Xb,
        axis=0
    )

    # --------------------------------------------------------
    # Individual discriminability B
    # --------------------------------------------------------

    individual = fisher_from_statistics(
        mean_a,
        var_a,
        mean_b,
        var_b,
        eps=eps
    )

    individual = individual.ravel()

    # --------------------------------------------------------
    # Pairwise matrices
    # --------------------------------------------------------

    interactions = np.zeros(
        (
            n_patches,
            n_patches
        ),
        dtype=np.float32
    )

    usefulness = np.zeros(
        (
            n_patches,
            n_patches
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Pairwise computation
    #
    # Only calculate p < q because the relationship is
    # symmetric.
    # --------------------------------------------------------

    for p in range(
        n_patches
    ):

        Xa_p = Xa[:, p, :]
        Xb_p = Xb[:, p, :]

        for q in range(
            p + 1,
            n_patches
        ):

            Xa_q = Xa[:, q, :]
            Xb_q = Xb[:, q, :]

            # ------------------------------------------------
            # Joint discriminability
            # ------------------------------------------------

            joint = compute_joint_fisher(
                samples_a_p=Xa_p,
                samples_a_q=Xa_q,
                samples_b_p=Xb_p,
                samples_b_q=Xb_q,
                eps=eps
            )

            # ------------------------------------------------
            # Interaction
            #
            # Positive:
            #   synergy
            #
            # Negative:
            #   redundancy
            #
            # Zero:
            #   no additional interaction
            # ------------------------------------------------

            interaction = (
                compute_patch_interaction(
                    individual[p],
                    individual[q],
                    joint
                )
            )

            interactions[p, q] = (
                interaction
            )

            interactions[q, p] = (
                interaction
            )

            # ------------------------------------------------
            # Combined usefulness
            #
            # This is the important new quantity.
            #
            # Unlike interaction, this does NOT subtract
            # the individual discriminability.
            #
            # Therefore a patch can have:
            #
            #   high B
            #   low I
            #   high C
            #
            # which means:
            #
            # "already useful, but not made more useful
            # by its partner."
            # ------------------------------------------------

            usefulness[p, q] = joint
            usefulness[q, p] = joint

    # --------------------------------------------------------
    # Reduce interaction across partners
    # --------------------------------------------------------

    interaction_scores = reduce_interactions(
        interactions,
        method=interaction_reduction,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Reduce joint usefulness across partners
    # --------------------------------------------------------

    usefulness_scores = reduce_interactions(
        usefulness,
        method=usefulness_reduction,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Convert to spatial maps
    # --------------------------------------------------------

    individual_map = (
        individual.reshape(
            height,
            width
        )
    )

    interaction_map = (
        interaction_scores.reshape(
            height,
            width
        )
    )

    usefulness_map = (
        usefulness_scores.reshape(
            height,
            width
        )
    )

    return (
        interaction_map,
        individual_map,
        usefulness_map
    )


# ============================================================
# Joint discriminability reduction
# ============================================================

def reduce_joint_scores(
    joint_scores,
    method="mean",
    top_k=10
):
    """
    Reduce a pairwise joint-discriminability matrix
    to one score per patch.

    Parameters
    ----------
    joint_scores : np.ndarray
        Shape:
            (n_patches, n_patches)

        joint_scores[p, q] is the Fisher discriminability
        of patches p and q considered jointly.

    method : str
        "mean"
            Mean joint discriminability with all other patches.

        "topk_mean"
            Mean joint discriminability of the strongest
            top-k partners.

        "max"
            Best joint discriminability obtained with
            any other patch.

    top_k : int
        Number of partners used by topk_mean.

    Returns
    -------
    scores : np.ndarray
        Shape:
            (n_patches,)
    """

    joint_scores = np.asarray(
        joint_scores,
        dtype=np.float32
    )

    n_patches = (
        joint_scores.shape[0]
    )

    scores = np.zeros(
        n_patches,
        dtype=np.float32
    )

    for p in range(
        n_patches
    ):

        values = np.delete(
            joint_scores[p],
            p
        )

        if len(values) == 0:
            continue

        # ----------------------------------------------------
        # Mean
        # ----------------------------------------------------

        if method == "mean":

            scores[p] = np.mean(
                values
            )

        # ----------------------------------------------------
        # Strongest joint discriminability
        # ----------------------------------------------------

        elif method == "max":

            scores[p] = np.max(
                values
            )

        # ----------------------------------------------------
        # Mean of strongest joint discriminability values
        # ----------------------------------------------------

        elif method == "topk_mean":

            k = min(
                top_k,
                len(values)
            )

            indices = np.argpartition(
                values,
                -k
            )[-k:]

            scores[p] = np.mean(
                values[indices]
            )

        else:

            raise ValueError(
                f"Unknown joint reduction: "
                f"{method}"
            )

    return scores


# ============================================================
# Strongest interaction partners
# ============================================================

def get_top_interaction_partners(
    interactions,
    top_k=10,
    positive_only=True
):
    """
    Return the strongest interaction partners
    for every patch.

    Returns
    -------
    partners : dict

        {
            patch_index: [
                (partner_index, score),
                ...
            ]
        }
    """

    interactions = np.asarray(
        interactions,
        dtype=np.float32
    )

    n_patches = (
        interactions.shape[0]
    )

    partners = {}

    for p in range(
        n_patches
    ):

        values = interactions[p].copy()

        # Exclude self-interaction
        values[p] = -np.inf

        if positive_only:

            values[
                values <= 0
            ] = -np.inf

        valid = np.isfinite(
            values
        )

        if not np.any(valid):

            partners[p] = []

            continue

        valid_indices = np.where(
            valid
        )[0]

        k = min(
            top_k,
            len(valid_indices)
        )

        # Only consider valid interactions
        valid_values = values[
            valid_indices
        ]

        top_indices = np.argpartition(
            valid_values,
            -k
        )[-k:]

        top_indices = top_indices[
            np.argsort(
                valid_values[top_indices]
            )[::-1]
        ]

        partners[p] = [
            (
                int(valid_indices[i]),
                float(valid_values[i])
            )
            for i in top_indices
        ]

    return partners

def infer_patch_grid(n_patches):
    """
    Infer a square spatial patch grid from the number of patches.
    """

    side = int(np.sqrt(n_patches))

    if side * side != n_patches:
        raise ValueError(
            f"Cannot infer a square patch grid from "
            f"{n_patches} patches."
        )

    return side, side


def compute_interaction_aware_metrics(
    features,
    labels,
    pca_components=None,
    interaction_reduction="topk_mean",
    usefulness_reduction="topk_mean",
    top_k=10,
    eps=1e-10
):
    """
    Compute class-specific:

        B = individual patch discriminability

        I = patch interaction
            (synergy / redundancy)

        C = combined usefulness
            (usefulness of a patch when paired
             with another patch)

    Parameters
    ----------
    features : np.ndarray
        Shape:
            (n_samples, n_patches, feature_dim)

    labels : array-like

    pca_components : int or None
        Optional PCA dimensionality reduction.

    interaction_reduction : str
        Reduction used for interaction scores.

        Examples:
            "mean"
            "max"
            "topk_mean"
            "topk_abs"

    usefulness_reduction : str
        Reduction used for joint usefulness.

        Examples:
            "mean"
            "max"
            "topk_mean"

    top_k : int
        Number of strongest partners used by top-k reductions.

    eps : float
        Numerical stability constant.

    Returns
    -------
    metrics : dict

        {
            "B": {
                class_name: map
            },

            "I": {
                class_name: map
            },

            "C": {
                class_name: map
            }
        }
    """

    features = np.asarray(
        features,
        dtype=np.float32
    )

    labels = np.asarray(
        labels
    )

    # --------------------------------------------------------
    # Validate feature representation
    # --------------------------------------------------------

    if features.ndim != 3:

        raise ValueError(
            "Interaction-aware metric expects "
            "features with shape "
            "(n_samples, n_patches, feature_dim). "
            f"Got {features.shape}."
        )

    # --------------------------------------------------------
    # Classes
    # --------------------------------------------------------

    classes = np.unique(
        labels
    )

    # --------------------------------------------------------
    # Optional PCA
    #
    # PCA is deliberately kept inside this metric.
    # Nothing outside this metric needs to change.
    # --------------------------------------------------------

    features, _ = fit_patch_pca(
        features,
        n_components=pca_components
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    metrics = {
        "B": {},
        "I": {},
        "C": {}
    }

    # --------------------------------------------------------
    # Infer spatial grid
    # --------------------------------------------------------

    n_patches = features.shape[1]

    height, width = infer_patch_grid(
        n_patches
    )

    # --------------------------------------------------------
    # Class-specific maps
    # --------------------------------------------------------

    for class_name in classes:

        other_classes = [
            c
            for c in classes
            if c != class_name
        ]

        if not other_classes:
            continue

        individual_maps = []
        interaction_maps = []
        usefulness_maps = []

        # ----------------------------------------------------
        # Compare against every competing class
        # ----------------------------------------------------

        for other in other_classes:

            (
                interaction_map,
                individual_map,
                usefulness_map
            ) = compute_interaction_map(
                features,
                labels,
                class_a=class_name,
                class_b=other,
                interaction_reduction=interaction_reduction,
                usefulness_reduction=usefulness_reduction,
                top_k=top_k,
                eps=eps
            )

            individual_maps.append(
                individual_map
            )

            interaction_maps.append(
                interaction_map
            )

            usefulness_maps.append(
                usefulness_map
            )

        # ----------------------------------------------------
        # Average competing-class comparisons
        #
        # This is across classes, not across partners.
        # The partner reduction already happened inside
        # compute_interaction_map().
        # ----------------------------------------------------

        metrics["B"][class_name] = np.mean(
            individual_maps,
            axis=0
        )

        metrics["I"][class_name] = np.mean(
            interaction_maps,
            axis=0
        )

        metrics["C"][class_name] = np.mean(
            usefulness_maps,
            axis=0
        )

    return metrics





def compute_decision_tree_metrics(
    features,
    labels,
    max_depth=4,
    min_samples_leaf=2,
    cv=5,
    random_state=42
):

    labels = np.asarray(labels)
    classes = np.unique(labels)

    metrics = {
        "DT_ERROR": {},
        "DT_ACC": {}
    }

    n_locations = features.shape[1]

    for c in classes:

        # -----------------------------------------------------
        # Store one score per patch
        # -----------------------------------------------------

        errors = np.full(
            n_locations,
            np.nan,
            dtype=np.float32
        )

        accuracies = np.full(
            n_locations,
            np.nan,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # One-vs-rest labels
        # -----------------------------------------------------

        binary_labels = (
            labels == c
        ).astype(np.int32)

        class_counts = np.bincount(
            binary_labels
        )

        if len(class_counts) < 2:
            continue

        min_class_count = np.min(
            class_counts
        )

        n_splits = min(
            cv,
            min_class_count
        )

        if n_splits < 2:
            continue

        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

        # -----------------------------------------------------
        # Evaluate each patch independently
        # -----------------------------------------------------

        for p in range(n_locations):

            X_patch = features[
                :,
                p,
                :
            ]

            tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state
            )

            scores = cross_val_score(
                tree,
                X_patch,
                binary_labels,
                cv=splitter,
                scoring="accuracy"
            )

            accuracy = np.mean(
                scores
            )

            error = 1.0 - accuracy

            accuracies[p] = accuracy
            errors[p] = error

        # -----------------------------------------------------
        # Store results
        # -----------------------------------------------------

        metrics["DT_ERROR"][c] = errors
        metrics["DT_ACC"][c] = accuracies

    return metrics


