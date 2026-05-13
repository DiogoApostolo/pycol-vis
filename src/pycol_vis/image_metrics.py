
import cv2
import numpy as np
import os
import pandas as pd
from scipy import stats



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


from skimage.feature import graycomatrix, graycoprops
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from pycol_complexity import complexity as pycol_complexity

from scipy.linalg import eigh


from .utils.utils import load_image, load_image_gs, select_channel, get_average_image_shape, load_images, convert_to_hsv, sample_dataset, quantized_color_set, edge_mask

from sklearn.decomposition import PCA




from scipy.linalg import eigh


from .embeddings.embedding_utils import generate_embeddings, setup_cnn, embed_images
from .embeddings.reduction_utils import dim_reduction_aux, normalize_embs


from .visualization import plot_overlap_measures,plot_intrinsic_measures,plot_tsne,visualize_metrics_per_class,visualize_measure_distribution,visualize_specific_images

from .intrinsic import intrinsic_measures
from .overlap import overlap_measures
class ImageComplexity:
    def __init__(self, folder, keep_classes = 'all', number_per_class= -1, use_keras_dataset=False):
        self.use_keras_dataset = use_keras_dataset
        self.images = load_images(folder,keep_classes,number_per_class)
        self.image_shape = get_average_image_shape(self.images)
        
        self.num_classes = len(self.images['class'].unique())
        self.class_labels = self.images['class'].unique()

        self.is_trained = False
        self.overlap_measures_dic= {}
        print("Dataset loaded")




    def sample_dataset(self, n_samples_per_class, sample_type='jpeg_compression'):
        self.images = sample_dataset(self.images, n_samples_per_class=n_samples_per_class, sample_type=sample_type)
    

    def embed_images(self, emb_type, layer_index=-1, num_workers=0):

        if(emb_type != "CNN"):
            self.model = None

        self.feature_embeddings = embed_images(image_paths=self.images['image_path'], emb_type=emb_type, model=self.model, layer_index=layer_index, num_workers=num_workers)
        return self.feature_embeddings  


    def cnn_setup(self,depth=2,epochs=10,is_train=True):
        self.model = setup_cnn(image_shape=self.image_shape,num_classes=self.num_classes,images=self.images,depth=depth,epochs=epochs,train_model=is_train)


    def dim_reduction(self,emb,method='pca',n_components=50,custom_method=None):

        reduced_embs, reduction_method = dim_reduction_aux(embs=emb,method=method,n_components=n_components,custom_method=custom_method,return_model=True)
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
        '''

        n_regions = intrinsic_measures.n_regions(image_paths=self.images['image_path'], scale_factor=scale_factor, color_factor=color_factor, area_factor=area_factor)
        self.images['n_regions'] = n_regions


    def jpeg_compression_ratio(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        '''
        Calculate the JPEG compression ratio of each image at the specified quality level and store the values in the self.images DataFrame under the column 'jpeg_compression_ratio' and the RMSE values in the column 'jpeg_rmse'.
        '''

        jpeg_metrics = intrinsic_measures.jpeg_compression_ratio( image_paths=self.images['image_path'],quality=quality,channel=channel,is_edge_processing=is_edge_processing,edge_method=edge_method,direction=direction )
        
        #unzip the list of tuples into two separate lists
        jpeg_compression_ratio, jpeg_rmse = zip(*jpeg_metrics)

        self.images['jpeg_compression_ratio'] = jpeg_compression_ratio
        self.images['jpeg_rmse'] = jpeg_rmse


    def zipf_rank(self, channel='all'):
        '''
        Calculate the Zipf rank slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_slope' and 'zipf_r_value'.
        '''

        zipf = intrinsic_measures.zipf_rank(image_paths=self.images['image_path'], channel=channel )
        
        #unzip the list of tuples into two separate lists
        zipf_slope, zipf_r_value = zip(*zipf)

        self.images['zipf_slope'] = zipf_slope
        self.images['zipf_r_value'] = zipf_r_value


    def zipf_difference(self, channel='all'):
        '''
        Calculate the Zipf difference slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_diff_slope' and 'zipf_diff_r_value'.
        '''

        zipf =  intrinsic_measures.zipf_difference(image_paths=self.images['image_path'],channel=channel)

        #unzip the list of tuples into two separate lists
        zipf_diff_slope, zipf_diff_r_value = zip(*zipf)

        self.images['zipf_diff_slope'] = zipf_diff_slope
        self.images['zipf_diff_r_value'] = zipf_diff_r_value


    def count_unique_colors(self, bits_per_channel=8, use_mask=False):
        '''
        Calculate the number of unique colors in each image and store the values in the self.images DataFrame under the column 'unique_colors'.
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
        self.edge_density_canny()
        self.edge_density_sobel()
        self.hsv_std()
        self.hsv_mean()
        self.rgb_mean()
        self.rgb_std()
        self.entropy_measure()
        self.energy_measure()
        self.jpeg_compression_ratio()
        self.zipf_rank()
        self.zipf_difference()
        self.count_unique_colors()
        self.fft_measures()
        self.haralick_measures()
            
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

    def handle_embs_reduction(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10):
        embs = self.embed_images(emb_type=emb_type,layer_index=layer_index)

        if(embs is None):
            return None

        if(reduction_type is not None):
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_components=n_components)

        return embs

    def m_var_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10):
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

        Returns:
        - float: The calculated M_var value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")


        measure = overlap_measures.m_var_measure(embeddings=embs,labels=self.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"

        self.overlap_measures_dic['m_var_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def m_sep_measure(self, emb_type='efficient_net', layer_index=-1, reduction_type='pca', reduction_method=None, n_components=10):
        '''
        Compute the M_sep measure of class separability in the embedding space.

        M_sep is calculated using the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat) in the embedding space.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_sep. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to use for dimensionality reduction if reduction_type is specified. Default is 10.

        Returns:
        - float: The calculated M_sep value representing class separability in the embedding space.
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.m_sep_measure(embeddings=embs,labels=self.images['class'].values)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['m_sep_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def tabular_measure(self, layer_index=-1, reduction_type='pca', reduction_method=None, emb_type='efficient_net', measure='kdn', n_components=10):
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
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.tabular_measure(embeddings=embs, labels=self.images['class'].values, measure=measure)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['tabular_' + emb_type + '_layer' + layer_index_str] = measure

        return measure

    def auls_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10):
        '''
        Calculate the AULS complexity measure based on the spectrum of the graph. 
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.

        Returns:
        - float: The calculated AULS complexity score for the dataset based on the specified embedding
        '''

        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.auls_measure(embeddings=embs, labels=self.images['class'].values, n_samples=n_samples)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['auls_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    def csg_measure(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10, auls=False):
        '''
        Calculate the CSG complexity measure based on the spectrum of the graph. 
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - n_components (int): The number of components to keep if dimensionality reduction is applied. Only used if reduction_type is not None.
        - auls (bool): Whether to calculate the cmsAULS complexity measure instead of CSG. 

        Returns:
        - float: The calculated CSG or csmAULS complexity score for the dataset based on the specified embedding
        '''


        embs = self.handle_embs_reduction(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        if(embs is None):
            return ValueError("No embeddings found for the specified embedding type and layer index.")

        measure = overlap_measures.csg_measure(embeddings=embs, labels=self.images['class'].values, n_samples=n_samples, auls=auls)

        layer_index_str = str(layer_index) if layer_index >= 0 else "final"
        self.overlap_measures_dic['csg_' + emb_type + '_layer' + layer_index_str] = measure

        return measure


    
    def all_overlap_measures(self, layer_index=-1, emb_type='efficient_net', n_samples=50, reduction_type='pca', reduction_method=None, n_components=10):
        '''
        Calculate all overlap measures for the specified embedding type and layer index, and store the results in the self.overlap_measures_dic dictionary.
        '''

        self.m_var_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)
        self.m_sep_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)
        self.tabular_measure(emb_type=emb_type, layer_index=layer_index, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components, measure='kdn')
        self.auls_measure(emb_type=emb_type, layer_index=layer_index, n_samples=n_samples, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)
        self.csg_measure(emb_type=emb_type, layer_index=layer_index, n_samples=n_samples, reduction_type=reduction_type, reduction_method=reduction_method, n_components=n_components)

        print("All overlap measures calculated for embedding type:", emb_type, "and layer index:", layer_index) 

    # ==========================================
    # Visualization wrappers
    # ==========================================

    def plot_overlap_measures(self, cls='average'):
        '''
        Wrapper for overlap measure visualization.
        '''

        plot_overlap_measures(overlap_measures=self.overlap_measures_dic, labels=self.images['class'], cls=cls)


    def plot_intrinsic_measures(self):
        '''
        Wrapper for intrinsic measure visualization.
        '''

        plot_intrinsic_measures(images_df=self.images)


    def plot_tsne(self, embs=None, save_image=False, name="tsne_plot.png", folder="./"):
        '''
        Wrapper for t-SNE visualization.
        '''

        embeddings = embs if embs is not None else self.feature_embeddings

        if embeddings is None:
            raise ValueError("No embeddings found. Run embed_images() first or provide embeddings manually.")

        plot_tsne(embeddings=embeddings, labels=self.images['class'], save_image=save_image, name=name, folder=folder)


    def visualize_metrics_per_class(self, metric_name):
        '''
        Wrapper for metric-per-class visualization.
        '''

        visualize_metrics_per_class(images_df=self.images, metric_name=metric_name)


    def visualize_measure_distribution(self, measure="entropy", n=10, figsize=(15, 6), seed=None, by_class=False):
        '''
        Wrapper for complexity distribution visualization.
        '''

        visualize_measure_distribution(images_df=self.images, measure=measure, n=n, figsize=figsize, seed=seed, by_class=by_class)


    def visualize_specific_images(self, image_list, measure="entropy", figsize=(15, 3)):
        '''
        Wrapper for visualizing specific images.
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