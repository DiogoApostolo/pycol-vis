import numpy as np

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE


def dim_reduction(embs,method='pca',n_components=50,custom_method=None,return_model=False):
    '''
        Perform dimensionality reduction on the feature embeddings.

        Parameters:
        - emb (numpy.ndarray): The feature embeddings to reduce.
        - method (str): The dimensionality reduction method to use. Options include 'pca', 'tsne', or 'custom'. Default is 'pca'.
        - n_components (int): The number of components to keep after dimensionality reduction. Default is 50.
        - custom_method (callable): A custom dimensionality reduction method. If provided, this will be used instead of the default methods.
        - return_model (bool): Whether to return the fitted dimensionality reduction model along with the reduced embeddings. Default is False.

        Returns:
        - np.ndarray: A NumPy array containing the reduced feature embeddings.
        - object (optional): The fitted dimensionality reduction model, returned if return_model is True
    '''



    
    if(method == "pca"):
        reduction_method = PCA(n_components=n_components)
        reduced_embs = reduction_method.fit_transform(embs)

    elif(method == "tsne"):
        reduction_method = TSNE(n_components=n_components,random_state=42)
        reduced_embs = reduction_method.fit_transform(embs)

    elif(method == "custom"):
        if(custom_method is None):
            raise ValueError("custom_method must be provided when method='custom'")
        
        reduced_embs = custom_method.fit_transform(embs)
        reduction_method = custom_method

    else:
        raise ValueError(f"Unknown reduction method: {method}")

    if(return_model):
        return reduced_embs, reduction_method

    return reduced_embs


def normalize_embs(embs):
    ''' 
    Normalize the feature embeddings to the range [0, 1].
        
    Parameters: - embs (np.ndarray): The feature embeddings to normalize.
        
    Returns: - np.ndarray: A NumPy array containing the normalized feature embeddings. 
    '''

    embs_min = np.array(embs.min(axis=0))
    embs_max = np.array(embs.max(axis=0))

    zro_mask = (embs_max - embs_min) == 0
    embs_max[zro_mask] = 1
    embs_min[zro_mask] = 0

    embs = ((embs - embs_min) / (embs_max - embs_min))

    return embs