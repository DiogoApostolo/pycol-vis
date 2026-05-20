

import os






os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

from .overlap.api import OverlapAPI
from .intrinsic.api import IntrinsicAPI
from .embeddings.api import EmbeddingAPI


from .utils.utils import get_average_image_shape, load_images,  sample_dataset

#TODO: remove the following imports since they are now imported in the respective API files
from .embeddings.embedding_utils import setup_cnn, embed_images
from .embeddings.reduction_utils import dim_reduction
from .intrinsic import intrinsic_measures
from .overlap import overlap_measures

from .visualization import plot_overlap_measures,plot_intrinsic_measures,plot_tsne,visualize_metrics_per_class,visualize_measure_distribution,visualize_specific_images



class ImageComplexity:
    def __init__(self, folder, keep_classes = 'all', number_per_class= -1, set_size=None):
        '''
        Initialize the ImageComplexity class by loading images from the specified folder and setting up the necessary attributes for further analysis. 
        The method validates the input parameters, loads the images, and prepares the dataset for embedding generation and complexity analysis.

        Parameters:
        - folder (str): The path to the folder containing the image dataset. The folder should be organized with subdirectories for each class, where each subdirectory contains the images belonging to that class.
        - keep_classes (str or list): Specifies which classes to keep in the dataset. If set to 'all', all classes will be included. If set to a list of class labels, only those classes will be retained in the dataset.
        - number_per_class (int): The number of samples to include for each class. If set to -1, all samples from each class will be included. If set to a positive integer, only that many samples will be randomly selected from each class.
        - set_size (tuple or None): The desired size for the images in the dataset, specified as a tuple (height, width, channels). If set to None, the average image shape will be calculated from the loaded images and used as the image shape for further processing.
        
        '''

        if(number_per_class != -1 and number_per_class <= 0):
            raise ValueError("number_per_class must be a positive integer or -1 for all samples.")
        
        if(keep_classes != 'all' and not isinstance(keep_classes, list)):
            raise ValueError("keep_classes must be 'all' or a list of class labels to keep.")
        
        if(keep_classes != 'all' and isinstance(keep_classes, list) and len(keep_classes) == 0):
            raise ValueError("keep_classes list cannot be empty if keep_classes is not 'all'.")
        
        if(set_size != None and (not isinstance(set_size, tuple) or len(set_size) != 3)):
            raise ValueError("set_size must be a tuple of the form (height, width, channels) or None.")
        
        if(set_size != None and (not all(isinstance(x, int) and x > 0 for x in set_size))):
            raise ValueError("All values in set_size must be positive integers.")
        
        if(not os.path.exists(folder) or not os.path.isdir(folder)):

            raise ValueError("The specified folder does not exist or is not a directory.")
        
        if(keep_classes != 'all'):
            if(not os.path.exists(folder) or not os.path.isdir(folder)):
                raise ValueError("The specified folder does not exist or is not a directory.")
            available_classes = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            for cls in keep_classes:
                if cls not in available_classes:
                    raise ValueError(f"Class '{cls}' specified in keep_classes does not exist in the dataset folder.")
        

        self.images = load_images(folder,keep_classes,number_per_class)

        if(set_size==None):
            self.image_shape = get_average_image_shape(self.images)
        else:
            self.image_shape = set_size
        
        self.num_classes = len(self.images['class'].unique())
        self.class_labels = self.images['class'].unique()
        self.model = None
        self.is_trained = False
        self.overlap_measures_dic= {}
        print("Dataset loaded")


        self.embeddings = EmbeddingAPI(self)
        self.intrinsic = IntrinsicAPI(self)
        self.overlap =  OverlapAPI(self)    
        

    def sample_dataset(self, n_samples_per_class, sample_type='random'):
        '''
        Sample the dataset based on the specified sampling type and number of samples per class. The sampling is performed separately for each class to ensure a balanced representation of classes in the sampled dataset.
        Parameters:
        - n_samples_per_class (int): The number of samples to select for each class in the dataset. If set to -1, all samples from each class will be included in the sampled dataset.
        - sample_type (str): The type of sampling to perform. Options are 'random', 'complexity', or 'jpeg_compression'. 
            - 'random': Randomly select samples from each class without considering any specific criteria.
            - 'jpeg_compression': Select samples based on their JPEG compression ratio, which can be an indicator of image quality and complexity, to include images with varying levels of compression in the sampled dataset.
        
        '''
       
        self.images, indexes = sample_dataset(self.images, n_samples_per_class=n_samples_per_class, sample_type=sample_type)
       
        self.feature_embeddings = self.feature_embeddings[indexes]

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
        - layer_index (int): The index of the layer from which to extract embeddings if emb_type is 'CNN'. If -1 is specified, the final layer embeddings will be used.
        - num_workers (int): The number of worker processes to use for parallel embedding generation. Default is 0, which means that the embedding generation will be performed in the main process.
        '''
        
        #check if emb_type is valid
        if(emb_type not in ["raw", "CNN", "efficient_net", "mobile_net", "current"]):
            raise ValueError("Invalid embedding type. Supported types are: 'raw', 'CNN', 'efficient_net', 'mobile_net', 'current'.")

        if(emb_type == 'current'):
            if(self.feature_embeddings is None):
                print("No current embeddings found.")
                return None
            return self.feature_embeddings
        else:
            self.feature_embeddings = embed_images(image_paths=self.images['image_path'], emb_type=emb_type, model=self.model, layer_index=layer_index, num_workers=num_workers, device=device)
        return self.feature_embeddings  


    def cnn_setup(self,depth=2,epochs=10,is_train=True):
        '''
        Set up the CNN model for embedding generation.
        Parameters:
        - depth (int): The number of layers in the CNN model from which to extract embeddings. Default is 2.
        - epochs (int): The number of epochs to train the CNN model if is_train is True. Default is 10.
        - is_train (bool): Whether to train the CNN model or use a pre-trained model for embedding extraction. If True, the model will be trained on the dataset. If False, a pre-trained model will be used without additional training. Default is True.
        '''
        self.model = setup_cnn(image_shape=self.image_shape,num_classes=self.num_classes,images=self.images,depth=depth,epochs=epochs,train_model=is_train)


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
        self.reduction_method = reduction_method
        self.feature_embeddings = reduced_embs

        return reduced_embs
    


    # ==========================================
    # Intrinsic Measure Wrappers
    # ==========================================

    def edge_density_canny(self, low_threshold=0.11, high_threshold=0.27):
        '''
        Calculate the edge density of an image using the Canny edge detection algorithm and add the values to the images dataframe in a new column 'edge_density_canny'.

        Parameters:
        - low_threshold (float): The lower threshold for the Canny edge detection algorithm.
        - high_threshold (float): The upper threshold for the Canny edge detection algorithm.
        '''

        edge_density = intrinsic_measures.edge_density_canny(image_paths=self.images['image_path'],low_threshold=low_threshold,high_threshold=high_threshold)
        self.images['edge_density_canny'] = edge_density

    def edge_density_sobel(self, threshold=0.2):
        '''
        Calculate the edge density of an image using the Sobel edge detection algorithm and add the values to the images dataframe in a new column 'edge_density_sobel'.

        Parameters:
        - threshold (float): A value between 0 and 1 to determine the edge density threshold. The Sobel edge image is normalized to the range [0, 1], and edges are considered present where the normalized value exceeds the threshold.
        '''

        edge_density = intrinsic_measures.edge_density_sobel(image_paths=self.images['image_path'],threshold=threshold)
        self.images['edge_density_sobel'] = edge_density


    def hsv_std(self):
        '''
        Calculate the standard deviation of the color channels of each image in the HSV color space and store the values in the self.images DataFrame under the columns 'H_std', 'S_std', and 'V_std'.
        '''

        H_std, S_std, V_std = intrinsic_measures.hsv_std(image_paths=self.images['image_path'])
        self.images['H_std'] = H_std
        self.images['S_std'] = S_std
        self.images['V_std'] = V_std

    def hsv_mean(self):
        '''
        Calculate the average color of each image in the HSV color space and store the values in the self.images DataFrame under the columns 'H_mean', 'S_mean', and 'V_mean'.
        '''

        H_mean, S_mean, V_mean = intrinsic_measures.hsv_mean(image_paths=self.images['image_path'])
        self.images['H_mean'] = H_mean
        self.images['S_mean'] = S_mean
        self.images['V_mean'] = V_mean

    def rgb_mean(self):
        '''
        Calculate the average color of each image in the RGB color space and store the values in the self.images DataFrame under the columns 'R_mean', 'G_mean', and 'B_mean'.
        '''

        R_mean, G_mean, B_mean = intrinsic_measures.rgb_mean(image_paths=self.images['image_path'])
        self.images['R_mean'] = R_mean
        self.images['G_mean'] = G_mean
        self.images['B_mean'] = B_mean

    def rgb_std(self):
        '''
        Calculate the standard deviation of the RGB channels for each image and store the values in the self.images DataFrame under the columns 'R_std', 'G_std', and 'B_std'.
        '''

        R_std, G_std, B_std = intrinsic_measures.rgb_std(image_paths=self.images['image_path'])
        self.images['R_std'] = R_std
        self.images['G_std'] = G_std
        self.images['B_std'] = B_std

    def entropy_measure(self):
        '''
        Calculate the entropy of each image and store the values in the self.images DataFrame under the column 'entropy'.
        '''

        entropy = intrinsic_measures.entropy_measure(image_paths=self.images['image_path'])
        self.images['entropy'] = entropy


    def energy_measure(self):
        '''
        Calculate the energy of each image and store the values in the self.images DataFrame under the column 'energy'.
        '''

        energy = intrinsic_measures.energy_measure(image_paths=self.images['image_path'])
        self.images['energy'] = energy


    def n_regions(self, scale_factor=0.02, color_factor=0.1, area_factor=0.001):
        '''
        Calculate the number of regions in each image using a combination of color quantization and edge detection, and store the values in the self.images DataFrame under the column 'n_regions'.
        The mean shift algorithm is applied to each image to segment it into regions based on color and spatial proximity.
        Images are specified in the image_paths list, and the resulting number of regions for each image is returned as a list. 

        Parameters:
        - scale_factor (float): A value to determine the spatial radius for mean shift segmentation based on the image dimensions.
        - color_factor (float): A value to determine the color radius for mean shift segmentation based on the image dimensions.
        - area_factor (float): A value to determine the minimum region size for mean shift segmentation based on the image dimensions.

        '''

        n_regions = intrinsic_measures.n_regions(image_paths=self.images['image_path'], scale_factor=scale_factor, color_factor=color_factor, area_factor=area_factor)
        self.images['n_regions'] = n_regions


    def jpeg_compression_ratio(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        '''
        Calculate the JPEG compression ratio of each image at the specified quality level and store the values in the self.images DataFrame under the column 'jpeg_compression_ratio' and the RMSE values in the column 'jpeg_rmse'.

        The method compresses each image using JPEG compression at the specified quality level and calculates the compression ratio as the size of the compressed image divided by the size of the original image.

        User may choose to first apply edge processing to the image before compression, which may affect the compression ratio. 
        If edge processing is applied, the user can specify the method and direction for edge detection.

        Parameters:
        - quality (int): The quality level for JPEG compression (0 to 100).
        - channel (str): The image channel to use for compression. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
        - is_edge_processing (bool): Whether to apply edge processing to the image before compression.
        - edge_method (str): The method to use for edge processing if is_edge_processing is True.
        - direction (str): The direction of edges to calculate for edge processing. Options are 'x' for horizontal edges, 'y' for vertical edges, and 'all' for both.
        '''

        jpeg_metrics = intrinsic_measures.jpeg_compression_ratio( image_paths=self.images['image_path'],quality=quality,channel=channel,is_edge_processing=is_edge_processing,edge_method=edge_method,direction=direction )
        
        #unzip the list of tuples into two separate lists
        jpeg_compression_ratio, jpeg_rmse = zip(*jpeg_metrics)

        self.images['jpeg_compression_ratio'] = jpeg_compression_ratio
        self.images['jpeg_rmse'] = jpeg_rmse


    def zipf_rank(self, channel='all'):
        '''
        Calculate the Zipf rank slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_slope' and 'zipf_r_value'.

        The method computes the frequency of pixel values, ranks them, and performs a linear regression on the log-log scale to determine the slope and R-value of the distribution, which can provide insights into the complexity and structure of the image.


        Parameters:
            - channel (str): The image channel to use for calculating the Zipf rank. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
        '''

        zipf = intrinsic_measures.zipf_rank(image_paths=self.images['image_path'], channel=channel )
        
        #unzip the list of tuples into two separate lists
        zipf_slope, zipf_r_value = zip(*zipf)

        self.images['zipf_slope'] = zipf_slope
        self.images['zipf_r_value'] = zipf_r_value


    def zipf_difference(self, channel='all'):
        '''
        Calculate the Zipf difference slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_diff_slope' and 'zipf_diff_r_value'.

        Parameters:
            - channel (str): The image channel to use for calculating the Zipf rank. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
        '''

        zipf =  intrinsic_measures.zipf_difference(image_paths=self.images['image_path'],channel=channel)

        #unzip the list of tuples into two separate lists
        zipf_diff_slope, zipf_diff_r_value = zip(*zipf)

        self.images['zipf_diff_slope'] = zipf_diff_slope
        self.images['zipf_diff_r_value'] = zipf_diff_r_value


    def count_unique_colors(self, bits_per_channel=8, use_mask=False):
        '''
        Count the number of unique colors in each image and store the values in the self.images DataFrame under the column 'unique_colors', with optional quantization and edge masking. The method quantizes the colors of the image to reduce the number of unique colors, 
        making the computation more efficient and counting only the most relevant colors. If use_mask is True, an edge mask is applied to the image before counting unique colors, 
        which can help to focus on the most important regions of the image and reduce noise from irrelevant areas.

        Parameters:
        - bits_per_channel (int): The number of bits to use for quantization per color channel.
        - use_mask (bool): Whether to apply an edge mask to the image before counting unique colors. If True, an edge mask is applied to the image to focus on important regions and reduce noise from irrelevant areas. Default is False.
        '''

        colors_count, unique_colors = intrinsic_measures.count_unique_colors(image_paths=self.images['image_path'],bits_per_channel=bits_per_channel,use_mask=use_mask)
        self.images['unique_colors'] = colors_count
        self.unique_colors_array = unique_colors


    def fft_measures(self):
        '''
        Calculate the FFT-based measures for each image and store the values in the self.images DataFrame under the columns 'fft_entropy', 'fft_energy', and 'fft_bandwidth'.
        '''

        df_fft = intrinsic_measures.fft_measures(image_paths=self.images['image_path'])

        df_fft['class'] = self.images['class'].values
        df_fft['image_path'] = self.images['image_path'].values

        self.images = self.images.merge(df_fft, on=['class', 'image_path'])

    def haralick_measures(self):
        '''
        Calculate the Haralick texture features for each image and store the values in the self.images DataFrame under the columns 'contrast_haralick', 'correlation_haralick', 'energy_haralick', and 'homogeneity_haralick'.
        '''

        df_haralick = intrinsic_measures.haralick_measures(image_paths=self.images['image_path'])
    
        df_haralick['class'] = self.images['class'].values
        df_haralick['image_path'] = self.images['image_path'].values

        self.images = self.images.merge(df_haralick, on=['class', 'image_path'])


    def all_intrinsic_measures(self):
        '''
        Calculate all intrinsic measures for each image and store the values in the self.images DataFrame under the corresponding columns.
        '''
        print("Calculating all intrinsic measures...")
        self.intrinsic.edge_density_canny()
        self.intrinsic.edge_density_sobel()
        self.intrinsic.hsv_std()
        self.intrinsic.hsv_mean()
        self.intrinsic.rgb_mean()
        self.intrinsic.rgb_std()
        self.intrinsic.entropy_measure()
        self.intrinsic.energy_measure()
        self.intrinsic.jpeg_compression_ratio()
        self.intrinsic.zipf_rank()
        self.intrinsic.zipf_difference()
        self.intrinsic.count_unique_colors()
        self.intrinsic.fft_measures()
        self.intrinsic.haralick_measures()

    #--------------------------------Per class averages -------------------------

    def jpeg_compression_ratio_per_class(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        '''
        Get the average JPEG compression ratio values per class for the specified quality and channel.
        If the compression ratio values are not yet calculated, it will calculate them first.

        Returns:
        - pd.DataFrame: A DataFrame containing the average JPEG compression ratio values for each class.
        '''


        if('jpeg_compression_ratio' not in self.images.columns or 'jpeg_rmse' not in self.images.columns):
            self.jpeg_compression_ratio(quality=quality, channel=channel, is_edge_processing=is_edge_processing, edge_method=edge_method, direction=direction)
        
        return self.images.groupby('class')[['jpeg_compression_ratio', 'jpeg_rmse']].mean().reset_index()

    def edge_density_per_class(self, method='canny'):
        '''
        Get the average edge density values per class for the specified edge detection method.
        If the edge density values are not yet calculated for the specified method, it will calculate them using default paramenters.
        
        Returns:
        - pd.DataFrame: A DataFrame containing the average edge density values for each class and the specified method.
        
        '''
        if(method == 'canny'):
            if('edge_density_canny' not in self.images.columns):
                self.edge_density_canny()
            return self.images.groupby('class')[['edge_density_canny']].mean().reset_index()
        
        elif(method == 'sobel'):
            if('edge_density_sobel' not in self.images.columns):
                self.edge_density_sobel()
            return self.images.groupby('class')[['edge_density_sobel']].mean().reset_index()
        
        else:
            raise ValueError("Method must be either 'canny' or 'sobel'.")

    def get_rgb_mean_per_class(self):
        '''
        Get the average RGB mean values per class.
        If the RGB mean values are not yet calculated, it will calculate them first. 

        Returns:
        - pd.DataFrame: A DataFrame containing the average RGB mean values for each class.
        '''

        if('R_mean' not in self.images.columns or 'G_mean' not in self.images.columns or 'B_mean' not in self.images.columns):
            self.rgb_mean()
        
        return self.images.groupby('class')[['R_mean', 'G_mean', 'B_mean']].mean().reset_index()

    def get_hsv_mean_per_class(self):

        '''
        Get the average HSV mean values per class.
        If the HSV mean values are not yet calculated, it will calculate them first. 

        Returns:
        - pd.DataFrame: A DataFrame containing the average HSV mean values for each class.
        '''
    
        if('H_mean' not in self.images.columns or 'S_mean' not in self.images.columns or 'V_mean' not in self.images.columns):
            self.hsv_mean()
        
        return self.images.groupby('class')[['H_mean', 'S_mean', 'V_mean']].mean().reset_index()

    def calculate_entropy_per_class(self):
        '''
        Get the average entropy values per class.
        If the entropy values are not yet calculated, it will calculate them first.

        Returns:
        - pd.DataFrame: A DataFrame containing the average entropy values for each class.

        '''


        if('entropy' not in self.images.columns):
            self.entropy_measure()
        
        return self.images.groupby('class')[['entropy']].mean().reset_index()

    def zipf_difference_per_class(self, channel='all'):
        '''
        Get the average Zipf difference slope and r-value per class for the specified channel.
        If the Zipf difference values are not yet calculated for the specified channel, it will calculate them first.

        Returns:
        - pd.DataFrame: A DataFrame containing the average Zipf difference slope and r-value
        '''
        if('zipf_diff_slope' not in self.images.columns or 'zipf_diff_r_value' not in self.images.columns):
            self.zipf_difference(channel=channel)
        
        return self.images.groupby('class')[['zipf_diff_slope', 'zipf_diff_r_value']].mean().reset_index()
    
    
    def zipf_rank_per_class(self, channel='all'):
        '''
        Get the average Zipf rank slope and r-value per class for the specified channel.
        If the Zipf rank values are not yet calculated for the specified channel, it will calculate them first.
        Returns:
        - pd.DataFrame: A DataFrame containing the average Zipf rank slope and r-value for each class.
        '''

        if('zipf_slope' not in self.images.columns or 'zipf_r_value' not in self.images.columns):
            self.zipf_rank(channel=channel)
        
        return self.images.groupby('class')[['zipf_slope', 'zipf_r_value']].mean().reset_index()
    


    
    def get_all_values_per_class(self):
        '''
        Get the average values of all intrinsic measures calculated so far for each class.
        '''
        #check the columns self.images already has
        existing_columns = self.images.columns.tolist()
        
        #Remove 'image_path' and 'class' columns from the list of existing columns to get only the intrinsic measure columns
        existing_columns.pop(0)
        existing_columns.pop(0)
        return self.images.groupby('class')[existing_columns].mean().reset_index()

    
    

    # -------------------- OVERLAP METRICS -------------------------


    # ==========================================
    # Overlap measure wrappers
    # ==========================================

    def handle_embs_reduction(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        embs = self.embed_images(emb_type=emb_type, layer_index=layer_index, num_workers=num_workers, device=device)

        if(embs is None):
            return None

        if(reduction_type is not None):
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_components=n_components)

        return embs

    def m_var_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Compute the M_var measure of class variability in the embedding space.

        M_var is calculated using the normalized within-class scatter matrix (S_w_hat) in the embedding space, which captures the variability of samples within each class. 
        A lower M_var value indicates that samples within the same class are more tightly clustered together, suggesting better class separability.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_var. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 10.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        Returns:
        - float: The calculated M_var value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")


        measure = overlap_measures.m_var_measure(embeddings=embs,labels=self.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"

        self.overlap_measures_dic['m_var_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def m_sep_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Compute the M_sep measure of class separability in the embedding space.

        M_sep is calculated using the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat) in the embedding space.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_sep. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 10.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated M_sep value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.m_sep_measure(embeddings=embs,labels=self.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['m_sep_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def tabular_measure(self, layer_index=-1, reduction_type='pca', reduction_method=None, emb_type='efficient_net', measure='kdn', n_components=10, num_workers=0, device=None):
        '''
        Calculate overlap measures using the pycol complexity libray.

        Measure is stored in the self.overlap_measures_dic dictionary with a key composed of the measure name, embedding type, and layer index.

        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the overlap measures. Options are 'pca', 'tsne', or 'custom'. 
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - emb_type (str): The type of embeddings to use for the calculation. 
        - measure (str): The specific overlap measure to calculate. Options are 'n2', 'kdn', or 'lsc'. Each measure captures different aspects of class overlap and complexity in the feature space.
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 2.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.tabular_measure(embeddings=embs, labels=self.images['class'].values, measure=measure)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['tabular_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def auls_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Calculate the AULS complexity measure based on the spectrum of the graph. AULS is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
        A lower AULS value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated AULS complexity score for the dataset based on the specified embedding
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.auls_measure(embeddings=embs, labels=self.images['class'].values, n_samples=n_samples)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['auls_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def csg_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, auls=False, num_workers=0, device=None):
        '''
         Calculate the CSG complexity measure based on the spectrum of the graph. CSG is calculated using the eigenvalues of the Laplacian matrix derived from the similarity graph of the embeddings.
        A lower CSG value indicates better class separability in the embedding space, while a higher value indicates more overlap between classes.
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.
        - auls (bool): Whether to calculate the cmsAULS complexity measure instead of CSG. 
        - num_workers (int): The number of worker processes to use for parallel computation.
        - device (str): The device to use for computation (e.g., 'cpu' or 'cuda').

        Returns:
        - float: The calculated CSG or csmAULS complexity score for the dataset based on the specified embedding
        '''


        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.csg_measure(embeddings=embs, labels=self.images['class'].values, n_samples=n_samples, auls=auls)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['csg_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    
    def all_overlap_measures(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, num_workers=0, device=None):
        '''
        Calculate all overlap measures for the specified embedding type and layer index, and store the results in the self.overlap_measures_dic dictionary.
        '''

        
        self.overlap.m_var_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)
        self.overlap.m_sep_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)
        self.overlap.tabular_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, measure='kdn', num_workers=num_workers, device=device)
        self.overlap.auls_measure(emb_type=emb_type, layer_index=layer_index, n_samples=n_samples, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)
        self.overlap.csg_measure(emb_type=emb_type, layer_index=layer_index, n_samples=n_samples, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, num_workers=num_workers, device=device)

        print("All overlap measures calculated for embedding type:", emb_type, "and layer index:", layer_index) 

    # ==========================================
    # Visualization wrappers
    # ==========================================

    def plot_overlap_measures(self, cls='average'):
        '''
        Plot the overlap measures stored in a dictionary as a bar chart. cls can be set to a specific class to plot the measures for that class, or set to 'average' to plot the average values of the measures across all classes.

        Parameters:
        - cls (str): The class for which to plot the overlap measures. If 'average', the average values of the measures across all classes will be plotted. Default is 'average'.
        '''

        plot_overlap_measures(overlap_measures=self.overlap_measures_dic, labels=self.images['class'], cls=cls)


    def plot_intrinsic_measures(self):
        '''
        Plot the intrinsic measures stored in the dataframe as a bar chart.
        '''

        plot_intrinsic_measures(images_df=self.images)


    def plot_tsne(self, embs=None, save_image=False, name="tsne_plot.png", folder="./"):
        '''
        Plot a t-SNE visualization of feature embeddings. If embs is not provided, it will use the feature embeddings stored in self.feature_embeddings. If no embeddings are found, it will raise a ValueError. 
        The plot will be saved to the specified folder with the given name if save_image is True.

        Parameters:
        - embs (np.ndarray): The feature embeddings to visualize. If None, the method will use self.feature_embeddings.
        - save_image (bool): Whether to save the t-SNE plot as an image file. Default is False.
        - name (str): The name of the image file to save the plot. Default is "tsne_plot.png".
        - folder (str): The folder path where the image file will be saved if save_image is True. Default is "./".

        '''

        embeddings = embs if embs is not None else self.feature_embeddings

        if embeddings is None:
            raise ValueError("No embeddings found. Run embed_images() first or provide embeddings manually.")

        plot_tsne(embeddings=embeddings, labels=self.images['class'], save_image=save_image, name=name, folder=folder)


    def visualize_metrics_per_class(self, metric_name):
        '''
        Visualize the average values of a specific intrinsic measure for each class as a bar plot.  
        Parameters:
        - metric_name (str): The name of the intrinsic measure to visualize. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure. The method will calculate the average value of the specified measure for each class and create a bar plot to visualize the differences between classes.

        '''

        visualize_metrics_per_class(images_df=self.images, metric_name=metric_name)


    def visualize_measure_distribution(self, measure="entropy", n=10, figsize=(15, 6), seed=None, by_class=False):
        '''
        Method to visualize the images in a dataset
        based on their measured complexity.

        Images are presented in 3 Rows,
        corresponding to High, Medium and Low Complexity.

        Parameters:
        - measure (str): The name of the intrinsic measure to use for visualizing the distribution. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure.
        - n (int): The number of images to display for each complexity level (High, Medium, Low). Default is 10.
        - figsize (tuple): The size of the figure for the plot. Default is (15, 6).
        - seed (int): The random seed for reproducibility when selecting images to display. Default is None, which means that the selection will be random each time the method is called.
        - by_class (bool): Whether to visualize the distribution of the measure separately for each class. If True, the method will create separate plots for each class, showing the distribution of the specified measure within each class. If False, a single plot will be created showing the overall distribution of the measure across all classes. Default is False.
        '''

        visualize_measure_distribution(images_df=self.images, measure=measure, n=n, figsize=figsize, seed=seed, by_class=by_class)


    def visualize_specific_images(self, image_list, measure="entropy", figsize=(15, 3)):
        '''
        Visualize a specific list of images along with their corresponding values for a specified intrinsic measure. The method will create a plot displaying the selected images and annotate each image with its value for the specified measure.

        Parameters:
        - image_list (list): A list of image paths to visualize. Each path should correspond to an image in the dataset that has a calculated value for the specified measure in the self.images DataFrame.
        - measure (str): The name of the intrinsic measure to display for each image. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure. The method will annotate each image with its value for this measure.
        - figsize (tuple): The size of the figure for the plot. Default is (15, 3).
        '''

        visualize_specific_images(images_df=self.images, image_list=image_list, measure=measure, figsize=figsize)
        
    
#add main function to test the class
if __name__ == "__main__":
    dataset = "Fruit_dataset"
    folder = "./" + dataset +  "/train/"

    classes = ["apple","banana"]

    list_of_images = [folder + "apple\\0.jpg",folder + "apple\\2.jpg"]

    complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=200)
    
    complexity_train.entropy_measure()
    complexity_train.visualize_specific_images(image_list=list_of_images)

    '''
    complexity_train.haralick_measures()
    print(complexity_train.images.head())
    
    
    cmsAULS = complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca',n_components=10,auls=True)
    csg = complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca',n_components=10,auls=False)
    auls = complexity_train.auls_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca',n_components=10)   
    
    m_sep = complexity_train.m_sep_measure(emb_type="efficient_net", reduction_type='pca',n_components=10)
    m_var = complexity_train.m_var_measure(emb_type="efficient_net", reduction_type='pca',n_components=10)
    
    print("CSG measure:", csg)
    print("cmsAULS measure:", cmsAULS)
    print("AULS measure:", auls)
    print("M_sep measure:", m_sep)
    print("M_var measure:", m_var)
    '''