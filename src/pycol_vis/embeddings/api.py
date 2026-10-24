from .embedding_utils import embed_images, setup_cnn
from .reduction_utils import dim_reduction

class EmbeddingAPI:

    def __init__(self, parent):
        self.parent = parent

    def embed_images(self, emb_type, layer_index=-1, num_workers=0, device=None):
        '''
        Embed the images using the specified embedding type and layer index. The resulting embeddings are stored in the self.feature_embeddings attribute for later use in overlap measure calculations.
        
        Parameters:
        - emb_type (str): The type of embeddings to generate for the images. Options include:
          'raw' for raw pixel values
          'CNN' for embeddings extracted from a convolutional neural network (requires cnn_setup to be called first)
          'efficient_net' for embeddings generated using the EfficientNet architecture
          'mobile_net' for embeddings generated using the MobileNet architecture
          'current' to use previously calculated embeddings stored in self.feature_embeddings
          'histogram_texture' to extract histogram and texture features from the images
        - layer_index (int): The index of the layer from which to extract embeddings if emb_type is 'CNN'. If -1 is specified, the final layer embeddings will be used.
        - num_workers (int): The number of worker processes to use for parallel embedding generation. Default is 0, which means that the embedding generation will be performed in the main process.
        - device (str): The device to use for embedding generation (e.g., 'cpu' or 'cuda'). If None, the device will be automatically selected based on availability.
        '''
        
        

        if(emb_type != "CNN"):
            self.parent.model = None
        
        
        if(emb_type == 'current'):
            if(self.parent.feature_embeddings is None):
                print("No current embeddings found.")
                return None
            return self.parent.feature_embeddings
        else:
            self.parent.feature_embeddings = embed_images(image_paths=self.parent.images['image_path'], emb_type=emb_type, model=self.parent.model, layer_index=layer_index, num_workers=num_workers, device=device)
        return self.parent.feature_embeddings  
    
    def cnn_setup(self,depth=2,epochs=10,is_train=True):
        '''
        Set up the CNN model for embedding generation.
        Parameters:
        - depth (int): The number of layers in the CNN model from which to extract embeddings. Default is 2.
        - epochs (int): The number of epochs to train the CNN model if is_train is True. Default is 10.
        - is_train (bool): Whether to train the CNN model or use a pre-trained model for embedding extraction. If True, the model will be trained on the dataset. If False, a pre-trained model will be used without additional training. Default is True.
        '''
        self.parent.model = setup_cnn(image_shape=self.parent.image_shape,num_classes=self.parent.num_classes,images=self.parent.images,depth=depth,epochs=epochs,train_model=is_train)


    def dim_reduction(self,emb,method='pca',n_components=50,custom_method=None):
        '''
        Perform dimensionality reduction on the feature embeddings.

        Parameters:
        - emb (numpy.ndarray): The feature embeddings to reduce.
        - method (str): The dimensionality reduction method to use. Options include 'pca', 'tsne', or 'custom'. Default is 'pca'.
        - n_components (int): The number of components to keep after dimensionality reduction. Default is 50.
        - custom_method (callable): A custom dimensionality reduction method. If provided, this will be used instead of the default methods.

        Returns:
        - numpy.ndarray: The reduced feature embeddings.
        '''
        reduced_embs, reduction_method = dim_reduction(embs=emb,method=method,n_components=n_components,custom_method=custom_method,return_model=True)
        self.parent.reduction_method = reduction_method
        self.parent.feature_embeddings = reduced_embs

        return reduced_embs
    