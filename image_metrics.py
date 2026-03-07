import cv2
from matplotlib import image
from matplotlib.widgets import EllipseSelector
import numpy as np
import os
import pandas as pd
from scipy import stats

from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping


from pycol_complexity import complexity as pycol_complexity

from scipy.linalg import eigh
from embedding_models import ConvAutoencoder, EfficientNetLite0EmbeddingModel, MobileNetV3EmbeddingModel, CNNEmbeddingModel


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier

class ImageComplexity:
    def __init__(self, folder, keep_classes = 'all', number_per_class= -1, use_keras_dataset=False):
        self.use_keras_dataset = use_keras_dataset
        self.images = self.load_images(folder,keep_classes,number_per_class)
        self.image_shape = self.get_average_image_shape()
        
        self.num_classes = len(self.images['class'].unique())
        self.class_labels = self.images['class'].unique()

        self.is_trained = False
        self.overlap_measures_dic= {}
        print("Dataset loaded")

    

    def get_average_image_shape(self):
        '''
        Calculate the average image shape (height, width) across all images in the dataset.

        Returns:
        -tuple: A tuple containing the average width, average height, and number of channels (3 for RGB) for the dataset.
        '''
        total_height = 0
        total_width = 0
        count = 0

        for name in self.images['image_path']:
            image = self.load_image(name, convert_rgb=False)
            h, w = image.shape[:2]
            total_height += h
            total_width += w
            count += 1

        avg_height = total_height // (count)
        avg_width = total_width // (count)

        return (avg_width, avg_height, 3)
        


    def load_images(self,folder,keep_classes,number_per_class): 
        '''
        Load images from a folder and create a DataFrame with the image paths and corresponding class labels
        
        folder is expected to have the following structure:
        
        folder/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                image2.jpg
                ...
            ...

        Parameters:
        - folder (str): The path to the folder containing the images, organized in subfolders by class.
        - keep_classes (list or str): A list of class names to keep or 'all' to keep all classes.
        - number_per_class (int): The maximum number of images to load per class. Use -1 to load all images.

        Returns:
        - pd.DataFrame: A DataFrame with two columns: 'image_path' and 'class', containing the paths to the images and their corresponding class labels.
        '''       
        data = []
        
        if(keep_classes == 'all'):
            keep_classes = os.listdir(folder)

        for class_name in keep_classes:
            class_path = os.path.join(folder, class_name)
            
            if os.path.isdir(class_path):
                count = 0
                for image_name in os.listdir(class_path):
                    #if the number of images for this class is reached, stop loading more images for this class
                    if(number_per_class != -1 and count >= number_per_class):
                        break
                    image_path = os.path.join(class_path, image_name)
                    data.append([image_path, class_name])
                    count += 1


        df = pd.DataFrame(data, columns=["image_path", "class"])
        return df
        


    
    def load_image(self, image_path, convert_rgb=True):
        '''
        Load an image from the specified path.

        Parameters:
        - image_path (str): The path to the image file.
        - convert_rgb (bool): Whether to convert the image to RGB format. 

        Returns:
        - np.ndarray: The loaded image as a NumPy array. 
        '''
        image = cv2.imread(image_path)
        
        if convert_rgb:
            return self.convert_to_rgb(image)
        
        return image
    

    
    def load_image_gs(self,image_path):
        '''
        Load an image from the specified path in grayscale.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - np.ndarray: The loaded image as a NumPy array.
        '''
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

   
    def convert_to_rgb(self, image):
        '''
        Convert an image to RGB format.

        Parameters:
        - image (np.ndarray): The image to convert to RGB format.

        Returns:
        - np.ndarray: The converted image as a NumPy array.
        '''
         
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    



    

    def sample_dataset(self, n_samples_per_class,sample_type='complexity'):
        '''
        Sample the dataset based on the specified sampling type. Use either random sampling or the complexity meausures
        to select the most diverse/complex images from a dataset.

        The method modifies the self.images DataFrame to keep only the sampled images.

        Parameters:
        - n_samples_per_class (int): The number of samples to select per class.
        - sample_type (str): The type of sampling to perform. Options are 'random', 'complexity', or 'jpeg_compression'.
            - 'random': Randomly sample images from each class.
            - 'jpeg_compression': Sample images based on JPEG compression ratios.

        '''


        if(sample_type=='random'):
            sampled_images = self.images.groupby('class').apply(lambda x: x.sample(n=n_samples_per_class, random_state=42)).reset_index(drop=True)
        
        elif(sample_type=='jpeg_compression'):
            sampled_images = self.images.groupby('class').apply(lambda x: x.nsmallest(n_samples_per_class, 'jpeg_compression_ratio')).reset_index(drop=True)
            self.images = sampled_images


    def select_channel(self, name, channel='all'):
        


        if(channel=='all'):
            original_image = self.load_image(name,convert_rgb=False)
        elif(channel=='R'):
            original_image = self.load_image(name)[:,:,0]
        elif(channel=='G'):
            original_image = self.load_image(name)[:,:,1]
        elif(channel=='B'):
            original_image = self.load_image(name)[:,:,2]
        elif(channel=='H'):
            original_image = self.convert_to_hsv(self.load_image(name))[:,:,0]
        elif(channel=='S'):
            original_image = self.convert_to_hsv(self.load_image(name))[:,:,1]
        elif(channel=='V'):
            original_image = self.convert_to_hsv(self.load_image(name))[:,:,2]
        else:
            raise ValueError("Channel must be one of 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.")
        return original_image



    def cnn_setup(self, depth=2, epochs=10, is_train=True):
        '''
        Setup the CNN model for feature embedding and train it (optionally).

        Parameters:
        - depth: Number of convolutional layers in the CNN.
        - epochs: Number of training epochs.
        - is_train: Whether to train the model.
        '''

        cnn_model = CNNEmbeddingModel(image_shape=self.image_shape, num_classes=self.num_classes, depth=depth)
        
        if is_train:
            cnn_model.train_model(self.images,epochs=epochs)
        
        self.model = cnn_model


    def quantized_color_set(self, image, bits_per_channel):

        '''
        Auxiliary function to quantize the colors of an image

        Parameters:
        - image (np.ndarray): The input image as a NumPy array.
        - bits_per_channel (int): The number of bits to use for quantization per color channel.
        '''

        shift = 8 - bits_per_channel
        img_quantized = np.right_shift(image, shift).astype(np.uint16)

    
        color_indices = (
            (img_quantized[:, :, 0] << (2 * bits_per_channel)) +
            (img_quantized[:, :, 1] << bits_per_channel) +
            img_quantized[:, :, 2]
        )

        
        return color_indices
    
    
    
    def edge_mask(self,img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)

        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        return mask

    #-------------------------------



    def embed_images(self, emb_type, layer_index=-1):
        '''
        Extract feature embeddings for all images in the dataset.

        Parameters:
        - emb_type (str): The type of embeddings to extract. Options are 'raw', 'CNN', 'efficient_net', 'mobile_net', or 'current'.
        - layer_index (int): The index of the layer from which to extract the embeddings. Use -1 for the last feature layer.
        
        Returns:
        - np.ndarray: A NumPy array containing the extracted feature embeddings for all images in the dataset.
        '''


        if(emb_type in ['CNN'] and not hasattr(self, 'model')):
            
            print("Model not loaded. Please load a model before extracting CNN embeddings.")
            return None 

        if(emb_type == 'current'):
            if(self.feature_embeddings is None):
                print("No current embeddings found.")
                return None
            return self.feature_embeddings
        
        #flatten the images (not recomended for large image sizes)
        if(emb_type == 'raw'):
            feature_embeddings = []

            for image_path in self.images['image_path']:
                img = self.load_image(image_path)
                flattened_img = img.flatten()
                feature_embeddings.append(flattened_img)

            feature_embeddings = np.array(feature_embeddings)
            
        elif(emb_type == 'CNN'):
            print("Extracting CNN embeddings...")
            self.feature_embeddings = self.model.get_feature_embeddings_all(self.images,layer_index=layer_index)
            print("CNN embeddings extracted.")
        
        # -------- PRE TRAINED MODELS --------

        elif(emb_type == 'efficient_net'):

            print("Extracting EfficientNet-Lite0 embeddings...")
            model = EfficientNetLite0EmbeddingModel()
            feature_embeddings = []
            for image_path in self.images['image_path']:

                img = self.load_image(image_path)
                embedding = model(img)
                feature_embeddings.append(np.array(embedding))

            self.feature_embeddings = feature_embeddings
            print("EfficientNet-Lite0 embeddings extracted.")
        
        elif(emb_type == 'mobile_net'):

            print("Extracting MobileNetV3 embeddings...")
            model = MobileNetV3EmbeddingModel()
            feature_embeddings = []

            for image_path in self.images['image_path']:

                img = self.load_image(image_path)
                embedding = model(img)
                feature_embeddings.append(np.array(embedding))

            self.feature_embeddings = feature_embeddings
            print("MobileNetV3 embeddings extracted.")
            
        return self.feature_embeddings
    

    def dim_reduction(self,emb,method='pca',n_compoments=50,custom_method=None): 
        '''
        Reduce the dimensionality of the feature embeddings using the specified method.

        Parameters:
        - emb (np.ndarray): The feature embeddings to reduce.
        - method (str): The dimensionality reduction method to use. Options are 'pca', 'tsne', or 'custom'.
        - n_components (int): The number of components to keep.
        - custom_method (callable): A custom dimensionality reduction method. method parameter must be set to 'custom' to use this. 
        Returns:
        - np.ndarray: A NumPy array containing the reduced feature embeddings.
        '''
        
        
        if(method=='pca'):
            reduction_method = PCA(n_components=n_compoments)
            reduced_embs = reduction_method.fit_transform(emb)
            self.reduction_method = reduction_method
        
        elif(method=='tsne'):
            reduction_method = TSNE(n_components=n_compoments, random_state=42)
            reduced_embs = reduction_method.fit_transform(emb)
            self.reduction_method = reduction_method

        elif(method=='custom'):
            reduced_embs = custom_method(emb)
        
        
        return reduced_embs


    def normalize_embs(self,embs):
        '''
        Normalize the feature embeddings to the range [0, 1].

        Parameters:
        - embs (np.ndarray): The feature embeddings to normalize.

        Returns:
        - np.ndarray: A NumPy array containing the normalized feature embeddings.

        '''

        #normalize
        embs_min = np.array(embs.min(axis=0))
        embs_max = np.array(embs.max(axis=0))

        #check if max equals min to avoid division by zero
        zro_mask = (embs_max - embs_min) == 0
        embs_max[zro_mask] = 1
        embs_min[zro_mask] = 0

        embs = (embs - embs_min) / (embs_max - embs_min)

        return embs




    #-------------------------------



    def sobel_edges(self,channel, direction = 'all'):
        '''
        Calculate the Sobel edges for a given image channel and direction.

        Parameters:
        - channel (np.ndarray): The image channel for which to calculate the Sobel edges.
        - direction (str): The direction of edges to calculate. Options are 'x' for horizontal edges, 'y' for vertical edges, and 'all' for both.
        
        Returns:
        - np.ndarray: A NumPy array containing the calculated Sobel edges for the specified channel and direction.

        '''
        if(direction == 'x'):
            # Horizontal edges (dx=1, dy=0)
            sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobel_scale = cv2.convertScaleAbs(sobel_x)
        if(direction == 'y'):
            # Vertical edges (dx=0, dy=1)
            sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            sobel_scale = cv2.convertScaleAbs(sobel_y)
        if(direction == 'all'):
            # Magnitude of gradients (all directions)
            sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
            sobel_all = cv2.magnitude(sobel_x, sobel_y)
            sobel_scale = cv2.convertScaleAbs(sobel_all)
        
        return sobel_scale
    
    
    def edge_processing(self,channel, method='sobel', direction = 'all'):
        #for each channel
        if(method=='sobel'):
            edge_image = self.sobel_edges(channel, direction = direction)
    
        return edge_image
    
    def edge_density_canny(self, low_threshold=0.11, high_threshold=0.27):
        '''
        Calculate the edge density of an image using the Canny edge detection algorithm.

        Adds the calculated edge density values to the self.images DataFrame.

        Parameters:
        - low_threshold (float): The lower threshold required for the Canny Edge filter. Should be a value between 0 and 1.
        -high_threshold (float): The upper threshold required for the Canny Edge filter. Should be a value between 0 and 1.
        
        '''
        density_array = []
        for name in self.images['image_path']:

            image = self.load_image(name,convert_rgb=False)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert normalized thresholds to 8-bit scale
            low = int(low_threshold * 255)
            high = int(high_threshold * 255)
            
            edges = cv2.Canny(gray, low, high)
            
            density = np.sum(edges > 0) / edges.size
            density_array.append(density)

        self.images['edge_density_canny'] = density_array


    def edge_density_sobel(self, threshold=0.2):
        '''
        Calculate the edge density of an image using the Sobel edge detection algorithm.

        Adds the calculated edge density values to the self.images DataFrame.

        Parameters:
        - threshold (float): The threshold for edge detection. Should be a value between 0 and 1.
        
        '''

        density_array = []

        for name in self.images['image_path']:

            image = self.load_image(name, convert_rgb=False)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            sobel = self.sobel_edges(gray, direction='all')

            # Normalize to 0–1
            sobel_normalized = sobel / 255.0

            edges = sobel_normalized > threshold
            density = np.sum(edges) / edges.size
            density_array.append(density)

        self.images['edge_density_sobel'] = density_array

    def convert_to_hsv(self,image):
        '''
        Convert an image to the HSV color space.

        Parameters:
        - image (np.ndarray): The image to convert to the HSV color space.
        Returns:
        - np.ndarray: The converted image in HSV color space as a NumPy array.
        '''
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    
    
    def calculate_color_average(self,image):
        '''
        Calculate the average color of an image.

        Parameters:
        - image (np.ndarray): The image for which to calculate the average color.

        Returns:
        -list: A list containing the average values for each channel.
        '''
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return [avg_color[0], avg_color[1], avg_color[2]]
    

    def calculate_color_std(self,image):
        '''
        Calculate the standard deviation of the color channels of an image.

        Parameters:
        - image (np.ndarray): The image for which to calculate the standard deviation.

        Returns:
        - list: A list containing the standard deviation values for each channel.
        '''
        std_color_per_row = np.std(image, axis=0)
        std_color = np.std(std_color_per_row, axis=0)
        return [std_color[0], std_color[1], std_color[2]]
    
    
    def hsv_std(self):

        '''
        Calculate the standard deviation of the HSV color channels for each image and store them in the self.images DataFrame.
        '''

        H_std, S_std, V_std = [], [], []

        for name in self.images['image_path']:
            h, s, v = self.calculate_color_std(self.convert_to_hsv((self.load_image(name))))

            H_std.append(h)
            S_std.append(s)
            V_std.append(v)


        self.images['H_std'] = H_std
        self.images['S_std'] = S_std
        self.images['V_std'] = V_std

    def hsv_mean(self):
        '''
        Calculate the average of the HSV color channels for each image and store them in the self.images DataFrame.
        '''


        H_mean, S_mean, V_mean = [], [], []
        for name in self.images['image_path']:
            h, s, v = self.calculate_color_average(self.convert_to_hsv((self.load_image(name))))

            H_mean.append(h)
            S_mean.append(s)
            V_mean.append(v)

            

        self.images['H_mean'] = H_mean
        self.images['S_mean'] = S_mean 
        self.images['V_mean'] = V_mean

      
    
    
    def rgb_mean(self):
        '''
        Calculate the average RGB values for each image and store them in the self.images DataFrame.
        '''
        R_means, G_means, B_means = [], [], []
        for name in self.images['image_path']:
            r, g, b = self.calculate_color_average((self.load_image(name)))
            
            R_means.append(r)
            G_means.append(g)
            B_means.append(b)
            
        self.images['R_mean'] = R_means
        self.images['G_mean'] = G_means
        self.images['B_mean'] = B_means


    def rgb_std(self):
        '''
        Calculate the standard deviation of the RGB color channels for each image and store them in the self.images DataFrame.
        '''
        R_std, G_std, B_std = [], [], []
        for name in self.images['image_path']:
            r, g, b = self.calculate_color_std((self.load_image(name)))
            
            R_std.append(r)
            G_std.append(g)
            B_std.append(b)
            
        self.images['R_std'] = R_std
        self.images['G_std'] = G_std
        self.images['B_std'] = B_std




    def entropy_measure(self):
        '''
        Calculate the entropy of each image and store the values in the self.images DataFrame under the column 'entropy'.
        '''

        entropy_array = []
        for image in self.images['image_path']:
            image = self.load_image_gs(image)
            histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
            
            # Get proabilities 
            histogram = histogram / histogram.sum()  

            # Remove zero entries to avoid log(0)
            histogram = histogram[histogram > 0]  
            
            entropy_value = -np.sum(histogram * np.log2(histogram))
            entropy_array.append(entropy_value)
        
        self.images['entropy'] = entropy_array

    

    def energy_measure(self):
        '''
        Calculate the energy of each image and store the values in the self.images DataFrame under the column 'energy'.
        '''
        
        energy_array_spacial = []

        for image in self.images['image_path']:
            image = self.load_image_gs(image)
            energy_spacial = np.sum(image.astype(np.float64) ** 2) / (image.shape[0] * image.shape[1])
            energy_array_spacial.append(energy_spacial)

        self.images['energy'] = energy_array_spacial
    

    def n_regions(self, scale_factor=0.02, color_factor=0.1, area_factor=0.001):
        '''
        Calculate the number of regions in each image and store the values in the self.images DataFrame under the column 'n_regions'.
        This method uses mean shift segmentation to identify regions in the image. 

        Parameters:
        - scale_factor (float): A value to determine the spatial radius for mean shift segmentation based on the image dimensions.
        - color_factor (float): A value to determine the color radius for mean shift segmentation based on the image dimensions.
        - area_factor (float):  A value to determine the minimum region size for mean shift segmentation based on the image dimensions.
        '''


        n_regions_array = []
        for img_path in self.images['image_path']:
            image = self.load_image(img_path, convert_rgb=False)    
            
            
            h, w = image.shape[:2]
            total_pixels = h * w

            
            spatial_radius = int(min(h, w) * scale_factor)      
            color_radius = int(255 * color_factor)               
            min_region_size = int(total_pixels * area_factor)    
            
            shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)

            gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels = cv2.connectedComponents(binary)

            unique_labels, counts = np.unique(labels, return_counts=True)

            # Exclude the background label (0) and count valid regions based on size
            region_sizes = counts[1:]  
            valid_regions = region_sizes[region_sizes >= min_region_size]

            num_valid = len(valid_regions)
            n_regions_array.append(num_valid)

        self.images['n_regions'] = n_regions_array 


   

    def jpeg_compression_ratio(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        
        '''
        Calculate the JPEG compression ratio for each image and store the values in the self.images DataFrame under the column 'jpeg_compression_ratio'.
        Furthermore, calculate the root mean square error (RMSE) between the original and compressed images and store the values in the self.images DataFrame under the column 'jpeg_rmse'.

        User may choose to first apply edge processing to the image before compression, which may affect the compression ratio. 
        If edge processing is applied, the user can specify the method and direction for edge detection.

        Parameters:
        - quality (int): The quality level for JPEG compression (0 to 100).
        - channel (str): The image channel to use for compression. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'. If 'all' is selected, the original RGB image will be used for compression. If a specific channel is selected, only that channel will be used for compression. If H, S or V are chosen, image is first converted to HSV format.
        - is_edge_processing (bool): Whether to apply edge processing to the image before compression.
        - edge_method (str): The method to use for edge processing if is_edge_processing is True. Options are 'sobel'. Only 'sobel' is currently implemented.
        - direction (str): The direction of edges to calculate for edge processing. Options are 'x' for horizontal edges, 'y' for vertical edges, and 'all' for both. Only used if is_edge_processing is True.

        '''
        
        ratios = []
        rmses = []


        for name in self.images['image_path']:
            original_image = self.select_channel(name, channel=channel)
            

            if(is_edge_processing == True):
                
                original_image = self.edge_processing(original_image, method=edge_method, direction= direction)
                
            #convert
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_img = cv2.imencode('.jpg', original_image, encode_param)
            
            if(channel=='all'):
                jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            else:   
                jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)

            #save original and compressed images temporarily to calculate sizes for compression ratio
            cv2.imwrite("./temp.png", original_image)
            cv2.imwrite("./temp.jpg", original_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            original_size = os.path.getsize("./temp.png")
            jpeg_size = os.path.getsize("./temp.jpg")

            #compression ratio
            compression_ratio =  jpeg_size / original_size

            ratios.append(compression_ratio)

            #root mean square error between original and compressed image
            diff = (original_image.astype(np.float32) - jpeg_image.astype(np.float32)) ** 2
            mse = np.mean(diff)
            rmse = np.sqrt(mse)

            rmses.append(rmse)

            
           
        self.images['jpeg_compression_ratio'] = ratios
        self.images['jpeg_rmse'] = rmses
    

    def zipf_rank(self, channel='all'):
        '''
        Calculate the Zipf's law slope and R-value for the pixel value distribution of each image and store the values 
        in the self.images DataFrame under the columns 'zipf_slope' and 'zipf_r_value', respectively.
        
        Function is based on Zipf-like statistics and Zipf's Law, which claims that in many natural 
        processes the frequency of something is inversely proportional to its rank. 

        Parameters:
        - channel (str): The image channel to use for the calculation. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'. If 'all' is selected, the original RGB image will be used for the calculation. If a specific channel is selected, only that channel will be used for the calculation. If H, S or V are chosen, image is first converted to HSV format.
        '''

        slopes = []
        r_values = []
        for name in self.images['image_path']:
            image = self.select_channel(name, channel=channel)
            values, counts = np.unique(image, return_counts=True)

           
            counts_sorted = np.sort(counts)[::-1]
            ranks = np.arange(1, len(counts_sorted) + 1)

          
            log_ranks = np.log10(ranks)
            log_counts = np.log10(counts_sorted)

            
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_counts)
            
            if(r_value == 0.0):
                slope = 0.0
            
            slopes.append(slope)
            r_values.append(r_value)
        self.images['zipf_slope'] = slopes
        self.images['zipf_r_value'] = r_values


    def zipf_difference(self, channel='all'):  
        '''
        Calculate the Zipf's difference slope and R-value for the pixel value distribution of each image and store the values 
        in the self.images DataFrame under the columns 'zipf_slope' and 'zipf_r_value', respectively.
        
        Unlike the zipf_rank method, which calculates the slope and R-value based on the frequency of pixel values,
        this method calculates the slope and R-value based on the frequency of pixel value differences between neighboring pixels.

        Function is based on Zipf-like statistics and Zipf's Law, which claims that in many natural 
        processes the frequency of something is inversely proportional to its rank. 

        Parameters:
        - channel (str): The image channel to use for the calculation. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'. If 'all' is selected, the original RGB image will be used for the calculation. If a specific channel is selected, only that channel will be used for the calculation. If H, S or V are chosen, image is first converted to HSV format.
        ''' 
        slopes = []
        r_values = []
        for name in self.images['image_path']:
            image = self.select_channel(name, channel=channel)

            shifts = [(-1, -1), (-1, 0), (-1, 1),
                      ( 0, -1),          ( 0, 1),
                      ( 1, -1), ( 1, 0), ( 1, 1)
                      ]
        
            differences = []

            for dx, dy in shifts:
                shifted = np.roll(image, shift=(dx, dy), axis=(0, 1))
                diff = np.abs(image - shifted)
                differences.append(diff)

            diffs = np.concatenate([d.flatten() for d in differences])

            values, counts = np.unique(diffs, return_counts=True)
            valid_mask = (values > 0) & (values <= 255)
            
            
            
            values = values[valid_mask]
            counts = counts[valid_mask]

            if(len(values) < 2):
                slopes.append(0.0)
                r_values.append(0.0)
                continue

            log_values = np.log10(values)
            log_counts = np.log10(counts)

            slope, intercept, r_value, p_value, std_err = stats.linregress(log_values, log_counts)
            
            if(r_value == 0.0):
                slope = 0.0
            
            slopes.append(slope)
            r_values.append(r_value)


        
        self.images['zipf_diff_slope'] = slopes
        self.images['zipf_diff_r_value'] = r_values


    def count_unique_colors(self,bits_per_channel,use_mask):
        '''
        Count the number of unique colors in each image and store the values in the self.images DataFrame under the column 'unique_colors'.
        The method quantizes the colors of the image to reduce the number of unique colors, making the computation more efficient and counting only the most relevant colors.
        
        Parameters:
        - bits_per_channel (int): The number of bits to use for quantization per color channel. 
        - use_mask (bool): Whether to apply an edge mask to the image before counting unique colors. Can be useful to count colors in edge regions and avoid counting colors in flat background areas. 
        
        '''
        
        
        unique_colors_array = []
        colors_count_array = []
        
        for name in self.images['image_path']:
            image = self.load_image(name)

        
                
            
            colors = self.quantized_color_set(image,bits_per_channel)

            if(use_mask):
                mask = self.edge_mask(image)
                colors = colors[mask>0]

            unique_colors, counts = np.unique(colors, return_counts=True)
            colors_count_array.append(counts)
            unique_colors_array.append(unique_colors)
        
        return unique_colors_array,colors_count_array




    def fft_texture_features(self,img_path):
        img = self.load_image_gs(img_path)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log1p(np.abs(fshift))
        h, w = magnitude_spectrum.shape
        cy, cx = h//2, w//2
        r = min(cx, cy)
        
        #energy in low, mid and high frequency bands
        low = magnitude_spectrum[cy-r//4:cy+r//4, cx-r//4:cx+r//4].mean()
        mid = magnitude_spectrum[cy-r//2:cy+r//2, cx-r//2:cx+r//2].mean()
        high = magnitude_spectrum.mean()

        return np.array([low, mid, high])
    
    def fft_radial_profile(self,img_path, num_bins=10):
        img = self.load_image_gs(img_path)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log1p(np.abs(fshift))

        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r = r.astype(int)

        radial_profile = np.zeros(num_bins)
        for i in range(num_bins):
            mask = (r >= i * (r.max() / num_bins)) & (r < (i + 1) * (r.max() / num_bins))
            radial_profile[i] = magnitude_spectrum[mask].mean() if np.any(mask) else 0

        return radial_profile



    def get_fft_features(self):
        features_array = []
        for index, row in self.images.iterrows():
            features = self.fft_texture_features(row['image_path'])
            radial = self.fft_radial_profile(row['image_path'], num_bins=3)
            features = np.concatenate((features, radial))
            features_array.append(features)
        
        
        df_fft = pd.DataFrame(features_array, columns=['fft_low', 'fft_mid', 'fft_high','radial_bin1','radial_bin2','radial_bin3'])
        
        df_fft['class'] = self.images['class'].values
        df_fft['image_path'] = self.images['image_path'].values
        return df_fft

    
    def haralick_features(self,image_path,get_embeddings=False):
        
        
        img = self.load_image_gs(image_path)
        
        
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if get_embeddings:
            img = np.stack([img]*3, axis=-1)
            img = self.get_feature_embeddings(img).mean(axis=2).astype(np.uint8)

        # Compute GLCM at multiple angles/distances
        glcm = graycomatrix(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            symmetric=True, normed=True)
        
        features = []
        for prop in ('contrast', 'correlation', 'energy', 'homogeneity','entropy'):
            vals = graycoprops(glcm, prop)
            features.extend(vals.mean(axis=1))  # average over angles

        return np.array(features)
    
    def get_haralick_features(self,get_embeddings=False):

        

        features_array = []
        for index, row in self.images.iterrows():
            features = self.haralick_features(row['image_path'],get_embeddings=get_embeddings)
            features_array.append(features)
        
        scaler = MinMaxScaler()
        df_haralick = pd.DataFrame(scaler.fit_transform(features_array), columns=['contrast', 'correlation', 'energy', 'homogeneity','entropy'])
        
        df_haralick['class'] = self.images['class'].values
        df_haralick['image_path'] = self.images['image_path'].values
        return df_haralick
    
    

            
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
    


    
    

    


    # -------------------- OVERLAP METRICS -------------------------



    
    def compute_normalized_matrices(self, X, y):
        '''
        Compute the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat).
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
    
    
    
    def compute_m_sep_direct(self, S_w_hat, S_b_hat):
        '''
        Auxiliary function to compute M_sep directly from the normalized within-class scatter matrix S_w_hat and the normalized between-class scatter matrix S_b_hat.
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
            


    def m_sep_measure(self,emb_type='CNN', layer_index=-1, reduction_type=None, reduction_method=None):
        '''
        Compute the M_sep measure of class separability in the embedding space.

        M_sep is calculated using the normalized within-class scatter matrix (S_w_hat) and the normalized between-class scatter matrix (S_b_hat) in the embedding space.

        Parameters:
        - emb_type (str): The type of embeddings to use for the calculation.
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating M_sep. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 

        Returns:
        - float: The calculated M_sep value representing class separability in the embedding space.
        '''
        embs = self.embed_images(emb_type=emb_type, layer_index=layer_index)

        if embs is None:
            return None

        if reduction_type is not None:
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_compoments=2)
            self.feature_embeddings = embs
        

        S_w_hat, S_b_hat = self.compute_normalized_matrices(embs, self.images['class'].values)
        m_sep = self.compute_m_sep_direct(S_w_hat, S_b_hat)
        
        layer_index_str = str(layer_index) if layer_index >= 0 else "final" 

        self.overlap_measures_dic['m_sep_' + emb_type + '_layer' + str(layer_index_str)] = m_sep
        return m_sep



    

    def tabular_measure(self, layer_index=-1, reduction_type=None, reduction_method=None, emb_type='CNN', measure='kdn'):
        '''
        Calculate overlap measures using the pycol complexity libray.

        Measure is stored in the self.overlap_measures_dic dictionary with a key composed of the measure name, embedding type, and layer index.

        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the overlap measures. Options are 'pca', 'tsne', or 'custom'. 
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 
        - emb_type (str): The type of embeddings to use for the calculation. 
        - measure (str): The specific overlap measure to calculate. Options are 'n2', 'kdn', or 'lsc'. Each measure captures different aspects of class overlap and complexity in the feature space.
        '''

        embs = self.embed_images(emb_type=emb_type, layer_index=layer_index)
        if embs is None:
            return None
        
        if reduction_type is not None:
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_compoments=2)
            self.feature_embeddings = embs
    
        
        dataset_dic = {'X': embs, 'y': self.images['class'].values}

        if(measure=='n2'):
            comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).N2(imb=True)
        if(measure=='kdn'):
            comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).kDN(imb=True)
        if(measure=='lsc'):
            comp_value = pycol_complexity.Complexity(file_type='array',dataset=dataset_dic).LSC(imb=True)


        layer_index_str = str(layer_index) if layer_index >= 0 else "final" 
        self.overlap_measures_dic[measure + '_' + emb_type + '_layer' + str(layer_index_str)] = comp_value

        return comp_value
    



    def knn_density_estimation(self, query_points, reference_points,k_neighbors=5):

        if len(reference_points) < k_neighbors:
            k = len(reference_points)
        else:
            k = k_neighbors
            
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn.fit(reference_points)
        
        distances, _ = knn.kneighbors(query_points)
        
        d = reference_points.shape[1]
        
        volumes = (2 * distances[:, -1]) ** d  
        volumes = np.maximum(volumes, 1e-10)
        
        n_ref = len(reference_points)
        densities = k / (n_ref * volumes)
        
        return densities
    
    def compute_pairwise_similarity(self, embeddings_i, embeddings_j,n_samples=50):
        
        inxs = np.random.choice(len(embeddings_i), min(n_samples, len(embeddings_i)), replace=False)
        monte_carlo_samples = embeddings_i[inxs]
        probalities = self.knn_density_estimation(monte_carlo_samples, embeddings_j)      
        similarity = np.mean(probalities)
        
        return similarity
    
    def compute_similarity_matrix_S(self, data, n_samples=50):
        '''
        Compute similarity matrix S where S[i, j] represents the average similarity between samples from class i and samples from class j in the embedding space.
        '''
    
        class_labels = self.class_labels
        num_classes = self.num_classes


        similarity_matrix_S = np.zeros((num_classes, num_classes))
        
        print("Computing similarity matrix S...")
        for i in range(num_classes):
            for j in range(num_classes):
                
                embeddings_i = data[self.images['class'] == class_labels[i]]
                embeddings_j = data[self.images['class'] == class_labels[j]]
                similarity_matrix_S[i, j] = self.compute_pairwise_similarity(embeddings_i, embeddings_j,n_samples=n_samples)
        print("Similarity matrix S computed.")
        
        return similarity_matrix_S

    def compute_adjacency_matrix_W(self, similarity_matrix_S):
        
        size = similarity_matrix_S.shape[0]
        adjacency_matrix_W = np.zeros((size, size))
        
        print("Computing adjacency matrix W...")
        for i in range(size):
            for j in range(size):
                if i == j:
                    adjacency_matrix_W[i, j] = 1.0  
                else:
                    numerator = np.sum(np.abs(similarity_matrix_S[i, :] - similarity_matrix_S[j, :]))
                    denominator = np.sum(np.abs(similarity_matrix_S[i, :] + similarity_matrix_S[j, :]))
                    
                    if denominator == 0:
                        adjacency_matrix_W[i, j] = 0.0
                    else:
                        adjacency_matrix_W[i, j] = 1.0 - (numerator / denominator)
        
        adjacency_matrix_W = (adjacency_matrix_W + adjacency_matrix_W.T) / 2
        
        return adjacency_matrix_W
    
    def compute_laplacian_matrix_L(self, adjacency_matrix_W):
    
        degree_matrix_D = np.diag(np.sum(adjacency_matrix_W, axis=1))
        
        laplacian_matrix_L = degree_matrix_D - adjacency_matrix_W
        
        
        return laplacian_matrix_L, degree_matrix_D
    
    def compute_spectrum(self, laplacian_matrix_L):

        eigenvalues, eigenvectors = eigh(laplacian_matrix_L)
        
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    

    def compute_csg_complexity(self, eigenvalues):
        '''
        Auxiliary function to compute the CSG complexity score based on the eigenvalues of the graph.
        '''


        eig_size = len(eigenvalues)
        
        
        normalized_eigengaps = np.zeros(eig_size-1)
        for i in range(eig_size-1):
            delta_lambda = eigenvalues[i+1] - eigenvalues[i]
            normalized_eigengaps[i] = delta_lambda / (eig_size - i)
        
        
        cumulative_max = np.zeros_like(normalized_eigengaps)
        current_max = 0
        for i in range(len(normalized_eigengaps)):
            current_max = max(current_max, normalized_eigengaps[i])
            cumulative_max[i] = current_max
        
       
        csg_score = np.sum(cumulative_max)
        
        
        return csg_score





    def csg_measure(self, layer_index=-1,emb_type='CNN',n_samples=50,  reduction_type=None,reduction_method=None):
        '''
        Calculate the CSG complexity measure based on the spectrum of the graph. 
        
        Parameters:
        - layer_index (int): The index of the layer from which to extract embeddings. If -1 is specified, the final layer embeddings will be used.
        - emb_type (str): The type of embeddings to use for the calculation.
        - n_samples (int): The number of samples to use for the Monte Carlo estimation of pairwise similarities.
        - reduction_type (str): The type of dimensionality reduction to apply to the embeddings before calculating the CSG measure. Options are 'pca', 'tsne', or 'custom'. If None, no dimensionality reduction is applied.
        - reduction_method (callable): A custom dimensionality reduction method to apply to the embeddings if reduction_type is 'custom'. 

        Returns:
        - float: The calculated CSG complexity score for the dataset based on the specified embedding
        '''
        
        embs = self.embed_images(emb_type=emb_type, layer_index=layer_index)
        if embs is None:
            return None

        if reduction_type is not None:
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_compoments=2)
            self.feature_embeddings = embs

        similarity_matrix_S = self.compute_similarity_matrix_S(embs, n_samples=n_samples)
        
        W = self.compute_adjacency_matrix_W(similarity_matrix_S)
        L, D = self.compute_laplacian_matrix_L(W)
        eigenvalues, eigenvectors = self.compute_spectrum(L)

        csg_score = self.compute_csg_complexity(eigenvalues)

        
        layer_index_str = str(layer_index) if layer_index >= 0 else "final" 
        self.overlap_measures_dic['csg_' + emb_type + '_layer' + str(layer_index_str)] = csg_score

        return csg_score
        
    def plot_overlap_measures(self,cls='average'):
        '''
        Plot the overlap measures stored in the self.overlap_measures_dic dictionary as a bar chart.
        If cls is set to 'average', the average values of the measures will be plotted.

        Parameters:
        - cls (str): The class for which to plot the measures. If 'average', the average values of the measures will be plotted. If a specific class is specified, only the values for that class will be plotted, assuming the measures are stored as lists or arrays with values for each class.
        
        '''
        
        
        measures = list(self.overlap_measures_dic.keys())
        values = list(self.overlap_measures_dic.values())

        for i in range(len(values)):
            if isinstance(values[i], (list, np.ndarray)):
                if(cls=='average'):
                    values[i] = np.mean(values[i])
                else:
                    class_indices = np.where(self.images['class'] == cls)[0]
                    values[i] = values[i][class_indices]
                    

        plt.figure(figsize=(10, 6))
        plt.bar(measures, values)
        plt.xlabel('Value')
        plt.title('Image Overlap Measures')
        plt.show()
    
    def plot_intrinsic_measures(self):
        '''
        Plot the intrinsic measures stored in the self.intrinsic_measures_dic dictionary as a bar chart.
        '''


        intrinsic_measures = [col for col in self.images.columns if col not in ['image_path', 'class']]
        self.intrinsic_measures_dic = {}

        for measure in intrinsic_measures:
            self.intrinsic_measures_dic[measure] = self.images[measure].mean()

        measures = list(self.intrinsic_measures_dic.keys())
        values = list(self.intrinsic_measures_dic.values())

        plt.figure(figsize=(10, 6))
        plt.bar(measures, values)
        plt.xlabel('Value')
        plt.title('Image Intrinsic Measures')
        plt.show()

    def plot_tsne(self,embs=None,save_image=False,name="tsne_plot.png",folder="./"):
        '''
        Plot a t-SNE visualization of the feature embeddings. If embeddings are not provided, it will use the feature embeddings stored in self.feature_embeddings or calculate them if they are not already calculated.
        
        Parameters:
        - embs (np.ndarray): The feature embeddings to use for the t-SNE visualization. If None, the method will use self.feature_embeddings or calculate them if they are not already calculated.
        - save_image (bool): Whether to save the t-SNE plot as an image file.
        '''

        if(embs is not None):
            embeddings = embs.flatten().reshape(len(self.images), -1)
        else:
            #check if embeddings are already calculated
            if not hasattr(self, 'feature_embeddings'):
                
                embeddings = self.get_feature_embeddings_all().flatten().reshape(len(self.images), -1)

            else:
                embeddings = self.feature_embeddings.flatten().reshape(len(self.images), -1)

        if(embeddings.shape[1]>2):
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(embeddings)
        else:
            tsne_results = embeddings
            


        plt.figure(figsize=(8, 6))
        classes = self.images['class'].unique()
        

        for cls in classes:
            subset = tsne_results[self.images['class'] == cls]
            plt.scatter(subset[:, 0], subset[:, 1], label=cls, alpha=0.7)
        


        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title("t-SNE Feature Embeddings")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if(save_image):
            plt.savefig(folder + name, dpi=300)
        else:
            plt.show()

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
    

    def visualize_metrics_per_class(self, metric_name):
        '''
        Visualize the average values of a specific intrinsic measure for each class as a bar plot.

        Parameters:
        - metric_name (str): The name of the intrinsic measure to visualize. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure.
        '''

        class_means = self.get_all_values_per_class()
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_means['class'], class_means[metric_name])
        
        plt.xlabel('Class')
        plt.ylabel(metric_name)
        plt.title(f'Average {metric_name} per Class')
    
        plt.show()
    
    
    
    

        


dataset = "shapes_dataset"
folder = "./" + dataset +  "/train/"

classes = ["Circle","Square"]

complexity = ImageComplexity(folder,keep_classes=classes,number_per_class=300)

kDN = complexity.tabular_measure(emb_type='efficient_net',measure='kdn',reduction_type='pca')

print(kDN)



 #REDUCE DATASET SIZE EXAMPLE
'''
dataset = "shapes_dataset"
folder = "./" + dataset +  "/train/"

classes = ["Circle","Square","Triangle"]

complexity_train = ImageComplexity(folder,keep_classes=classes)

complexity_train.jpeg_compression_ratio()
complexity_train.sample_dataset(n_samples_per_class=5000,sample_type='jpeg_compression')

complexity_train.embed_images(emb_type='efficient_net')
complexity_train.feature_embeddings = complexity_train.dim_reduction(complexity_train.feature_embeddings,method='pca',n_compoments=10)
reduction_method = complexity_train.reduction_method.transform



X_train = complexity_train.feature_embeddings
y_train = complexity_train.images['class'].values

print(complexity_train.images.shape)

folder = "./" + dataset +  "/test/"

complexity_test = ImageComplexity(folder,keep_classes=classes)
complexity_test.embed_images(emb_type='efficient_net')
complexity_test.feature_embeddings = complexity_test.dim_reduction(complexity_test.feature_embeddings,method='custom',custom_method=reduction_method)


X_test = complexity_test.feature_embeddings
y_test = complexity_test.images['class'].values

accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)
print("XGB Accuracy:", accuracy_xgb)
'''


#------------------------- Viz Examples -------------------------------------
'''

complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca')
complexity_train.tabular_measure(emb_type='efficient_net',measure='kdn',reduction_type='pca')
complexity_train.compute_m_sep(emb_type='efficient_net', reduction_type='pca')
complexity_train.plot_overlap_measures()

complexity_train.plot_tsne(embs=complexity_train.feature_embeddings)


#complexity_train.calculate_energy()


#complexity_train.jpeg_compression_ratio()
#complexity_train.calculate_entropy()
#complexity_train.edge_density_canny()

#complexity_train.visualize_metrics_per_class('entropy')

#complexity_train.sample_dataset(n_samples_per_class=5000,sample_type='jpeg_compression')

'''


'''
complexity_train.embed_images(emb_type='efficient_net')

complexity_train.feature_embeddings = complexity_train.dim_reduction(complexity_train.feature_embeddings,method='pca',n_compoments=10)
reduction_method = complexity_train.reduction_method.transform



X_train = complexity_train.feature_embeddings
y_train = complexity_train.images['class'].values

print(complexity_train.images.shape)

folder = "./" + dataset +  "/test/"

complexity_test = ImageComplexity(folder,keep_classes=classes)
complexity_test.embed_images(emb_type='efficient_net')
complexity_test.feature_embeddings = complexity_test.dim_reduction(complexity_test.feature_embeddings,method='custom',custom_method=reduction_method)


X_test = complexity_test.feature_embeddings
y_test = complexity_test.images['class'].values

accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)
print("XGB Accuracy:", accuracy_xgb)
'''
#________________________________________________________

'''
#Example of usage
dataset = "CovidDataset"
folder = "./" + dataset +  "/train/"

#classes = ["Circle","Square","Triangle"]
classes = ["COVID19","PNEUMONIA"]

depth = 1
epochs = 1

complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=400)
#complexity_train.define_feature_embedding_model(network_type="CNN",depth=depth)
#complexity_train.train_model(epochs=epochs,network_type="CNN")
metric_train = complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca')

X_train = complexity_train.feature_embeddings
y_train = complexity_train.images['class'].values

folder = "./" + dataset +  "/test/"
complexity_test = ImageComplexity(folder,keep_classes=classes,number_per_class=400)

#complexity_test.model_to_train = complexity_train.model_to_train
#complexity_test.model_all_layers = complexity_train.model_all_layers
#complexity_test.model = complexity_train.model

reduction_method = complexity_train.reduction_method.transform
metric_test = complexity_test.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='custom', reduction_method=reduction_method)




X_test = complexity_test.feature_embeddings
y_test = complexity_test.images['class'].values

accuracy_svm = svm_classifier(X_train,y_train,X_test,y_test)
accuracy_nn = nn_classifier(X_train,y_train,X_test,y_test)
accuracy_knn = knn_classifier(X_train,y_train,X_test,y_test)
accuracy_xgb = xgb_classifier(X_train,y_train,X_test,y_test)


print("Train CSG Score:", metric_train)
print("Test CSG Score:", metric_test)

print("SVM Accuracy:", accuracy_svm)
print("NN Accuracy:", accuracy_nn)
print("KNN Accuracy:", accuracy_knn)
print("XGB Accuracy:", accuracy_xgb)
'''

#_______________________________________________________

'''
complexity.get_hsv()
complexity.sample_dataset(n_samples_per_class=100)
'''


'''
dataset = "apple_v_banana"
folder = "./" + dataset +  "/train/"
projection = "CNN_random"
layer = -1
depth = 3
epochs = 10
'''


'''
dataset_array = ["apple_v_banana"]
projection_array = ["CNN_tsne","CNN_random"]
layer=-1
depth_array = [2,3,4]
epochs_array = [5,10,20]

for dataset in dataset_array:
    for projection in projection_array:
        for depth in depth_array:
            for epochs in epochs_array:
                folder = "./" + dataset +  "/train/"

                if(layer == -1):
                    layer_name = "fin"
                else:
                    layer_name = str(layer)

                complexity = ImageComplexity(folder)
                complexity.define_feature_embedding_model(network_type="CNN",depth=depth)
                complexity.train_model(epochs=epochs,network_type="CNN")

                m_sep = complexity.csg_measure(emb_type=projection, layer_index=-1)



                name = dataset + "-" + projection + "-l" + layer_name + "-e" + str(epochs) + "-d" + str(depth)
                complexity.plot_tsne(complexity.feature_embeddings,save_image=True,name=name + ".png",folder="./plot_projection/")

                #save complexity results
                with open("./results/csg/" + name + "_complexity.txt", "w") as f:
                    f.write(f"CSG Score: {m_sep['csg_score']}\n")
                    f.write(f"Spectral Span: {m_sep['spectral_span']}\n")
                    f.write(f"Spectral Gap: {m_sep['spectral_gap']}\n")
                    f.write(f"Area Under Curve: {m_sep['area_under_curve']}\n")
                    f.write(f"Max Eigengap: {m_sep['max_eigengap']}\n")
                    f.write(f"Max Eigengap Position: {m_sep['max_eigengap_position']}\n")
                    f.write(f"Normalized Max Eigengap: {m_sep['normalized_max_eigengap']}\n")
                    f.write(f"Number of Classes: {m_sep['num_classes']}\n")
'''


#complexity = ImageComplexity("./CovidDataset/train/COVID19/COVID19(0).jpg")print("Sucessfuly loaded")


'''
for layer in range(1,len(complexity.model_all_layers)):
    print(f"Computing complexity for layer {layer}...")
    
    #measure = complexity.csg_measure(layer_index=layer,emb_type='CNN_tsne')['csg_score']
    measure = complexity.tabular_measure(layer_index=layer,measure='kdn',emb_type='CNN_tsne')
    
    layers_complexity.append(measure)
'''

