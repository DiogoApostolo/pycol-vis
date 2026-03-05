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


from complexity import Complexity


from scipy.linalg import eigh
from embedding_models import ConvAutoencoder, EfficientNetLite0EmbeddingModel, MobileNetV3EmbeddingModel


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


from classifiers import svm_classifier, nn_classifier, knn_classifier, xgb_classifier

class ImageComplexity:
    def __init__(self, folder, keep_classes = 'all', number_per_class= -1, use_keras_dataset=False):
        self.use_keras_dataset = use_keras_dataset
        self.images = self.load_images(folder,keep_classes,number_per_class)
        self.image_shape = self.get_average_image_shape()
        self.num_classes = len(self.images['class'].unique())
        
        self.is_trained = False
        self.overlap_measures_dic= {}
        print("Dataset loaded")

    def get_average_image_shape(self):
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
        data = []
        
        if(keep_classes == 'all'):
            keep_classes = os.listdir(folder)

        for class_name in keep_classes:
            class_path = os.path.join(folder, class_name)
            
            if os.path.isdir(class_path):
                count = 0
                for image_name in os.listdir(class_path):
                    if(number_per_class != -1 and count >= number_per_class):
                        break
                    image_path = os.path.join(class_path, image_name)
                    data.append([image_path, class_name])
                    count += 1

        df = pd.DataFrame(data, columns=["image_path", "class"])
        return df
        
        
    

    def sample_dataset(self, n_samples_per_class,sample_type='complexity'):
        if(sample_type=='random'):
            sampled_images = self.images.groupby('class').apply(lambda x: x.sample(n=n_samples_per_class, random_state=42)).reset_index(drop=True)
        elif(sample_type=='complexity'):
            #chose diverse images from the image df based on the complexity columns
            H_mean = self.images['H_mean']
            S_mean = self.images['S_mean']
            V_mean = self.images['V_mean']

            complexity_features = np.array(list(zip(H_mean, S_mean, V_mean)))

            scaler = MinMaxScaler()
            complexity_features_scaled = scaler.fit_transform(complexity_features)

            # Use KMeans to cluster and select representative samples
            
            sampled_images_list = []
            for class_name, group in self.images.groupby('class'):
                group_features = complexity_features_scaled[group.index]
                
                if len(group) <= n_samples_per_class:
                    sampled_images_list.append(group)
                    continue
                
                kmeans = KMeans(n_clusters=n_samples_per_class, random_state=42)
                kmeans.fit(group_features)
                cluster_centers = kmeans.cluster_centers_
                
                closest, _ = pairwise_distances_argmin_min(cluster_centers, group_features)
                
                sampled_images_list.append(group.iloc[closest])

            #replace self.images with sampled images
            sampled_images = pd.concat(sampled_images_list).reset_index(drop=True)
            self.images = sampled_images

        elif(sample_type=='jpeg_compression'):
            sampled_images = self.images.groupby('class').apply(lambda x: x.nsmallest(n_samples_per_class, 'jpeg_compression_ratio')).reset_index(drop=True)
            self.images = sampled_images


    def cnn_embedding_model(self,image_shape,num_classes,depth=1):
        inputs = Input(shape=image_shape)
        layers = []
        
        x_1 = Conv2D(128, (3, 3), activation='relu')(inputs)
        x_1 = MaxPooling2D((2, 2), name='feature_map')(x_1)

        layers.append(x_1)
        x_conv = x_1
        if(depth>=2):
            x_2 = Conv2D(64, (3, 3), activation='relu')(x_1)
            x_2 = MaxPooling2D((2, 2))(x_2)
            layers.append(x_2)
            x_conv = x_2

        if(depth>=3):
            x_3 = Conv2D(32, (3, 3), activation='relu')(x_2)
            x_3 = MaxPooling2D((2, 2))(x_3)
            layers.append(x_3)
            x_conv = x_3
        if(depth>=4):
            x_4 = Conv2D(16, (3, 3), activation='relu')(x_3)
            x_4 = MaxPooling2D((2, 2))(x_4)
            layers.append(x_4)
            x_conv = x_4

        
        x_flatten = Flatten(name='flatten_layer')(x_conv)
        x_dense_1 = Dense(64, activation = 'relu')(x_flatten)
        x_dropout = Dropout(0.5)(x_dense_1)
        x_dense_2 = Dense(num_classes, activation = 'softmax')(x_dropout)

        layers.append(x_flatten)
        layers.append(x_dense_1)
        layers.append(x_dropout)


        self.model_to_train = Model(inputs=inputs, outputs=x_dense_2)
        self.model_all_layers = [Model(inputs=inputs, outputs=x) for x in layers]
        self.model = Model(inputs=inputs, outputs=x_conv)
        self.model_to_train.compile(optimizer='adam', loss='categorical_crossentropy')
        
        



    def train_model(self,network_type="CNN",epochs=20):
        train_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        if(network_type=="CNN"):
            train_generator = train_datagen.flow_from_dataframe(
                dataframe=self.images,
                x_col="image_path",
                y_col="class",
                target_size=self.image_shape[:2],
                batch_size=32,
                class_mode='categorical',
                shuffle=True
            )
            
            self.model_to_train.fit(train_generator, epochs=epochs)
            self.is_trained = True
            return
        if(network_type=="CAE"):
            #load images for CAE
            for name in self.images['image_path']:
                image = self.load_image(name, convert_rgb=True)
                image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))
                image = image.astype('float32') / 255.0
                if 'x_data' not in locals():
                    x_data = np.expand_dims(image, axis=0)
                else:
                    x_data = np.vstack((x_data, np.expand_dims(image, axis=0)))


            self.model_to_train.fit(x_data, x_data, epochs=epochs,batch_size=32)
            self.is_trained = True
            return
        


    def cae_embedding_model(self,image_shape):
        self.model_to_train = ConvAutoencoder(input_shape=image_shape, latent_dim=64)
        self.model_to_train.compile(optimizer='adam', loss='mse')
        self.model = self.model_to_train.encoder  

     

    def define_feature_embedding_model(self,network_type="CNN",depth=1):
        if(network_type=="CNN"):
            self.cnn_embedding_model(self.image_shape,self.num_classes,depth=depth)
        if(network_type=="CAE"):
            self.cae_embedding_model(self.image_shape)
        


    def sobel_edges(self,channel, direction = 'all'):
        
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
    
    '''
    Convert an image to HSV color space.
    '''
    def convert_to_hsv(self,image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    
    '''
    Calculate the average color of an image.
    '''
    def calculate_color_average(self,image):
        avg_color_per_row = np.average(image, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        return [avg_color[0], avg_color[1], avg_color[2]]
    
    '''
    Calculate the average HSV values for each image
    '''
    def calculate_hsv_average(self,image):
        avg_hsv_per_row = np.average(image, axis=0)
        avg_hsv = np.average(avg_hsv_per_row, axis=0)
        return [avg_hsv[0], avg_hsv[1], avg_hsv[2]]
    
    '''
    '''
    def get_hsv(self):
        H_mean, S_mean, V_mean = [], [], []

        for name in self.images['image_path']:
            h, s, v = self.calculate_hsv_average(self.convert_to_hsv((self.load_image(name))))
            
            H_mean.append(h)
            S_mean.append(s)
            V_mean.append(v)
        
        self.images['H_mean'] = H_mean
        self.images['S_mean'] = S_mean 
        self.images['V_mean'] = V_mean

    def get_hsv_per_class(self):
        #Verify if HSV values are already calculated
        if 'H_mean' not in self.images.columns or 'S_mean' not in self.images.columns or 'V_mean' not in self.images.columns:
            self.get_hsv()
        
        return self.images.groupby('class')[['H_mean', 'S_mean', 'V_mean']].mean().reset_index()
    
    
    '''
    Calculate the average RGB values for each image and store them in the DataFrame.
    '''
    def get_rgb(self):
        
        R_means, G_means, B_means = [], [], []
        for name in self.images['image_path']:
            r, g, b = self.calculate_color_average((self.load_image(name)))
            
            R_means.append(r)
            G_means.append(g)
            B_means.append(b)
            
        self.images['R_mean'] = R_means
        self.images['G_mean'] = G_means
        self.images['B_mean'] = B_means

    def get_rgb_per_class(self):
        #Verify if RGB values are already calculated
        if 'R_mean' not in self.images.columns or 'G_mean' not in self.images.columns or 'B_mean' not in self.images.columns:
            self.get_rgb()
        
        return self.images.groupby('class')[['R_mean', 'G_mean', 'B_mean']].mean().reset_index()


    def get_feature_embeddings_all(self,layer_index=-1,batch_size=32):

        




        embeddings = []
        num_images = len(self.images)

        for start_idx in range(0, num_images, batch_size):
            end_idx = min(start_idx + batch_size, num_images)
            batch_images = []

            for i in range(start_idx, end_idx):
                image = self.load_image(self.images['image_path'].iloc[i])
                image = cv2.resize(image, (self.model.input_shape[2], self.model.input_shape[1]))
                batch_images.append(image)

            batch_images = np.array(batch_images)

            if(layer_index==-1):
                features = self.model.predict(batch_images)
            else:
                features = self.model_all_layers[layer_index].predict(batch_images)

            embeddings.extend(features)

    
        self.feature_embeddings = np.array(embeddings)
        return np.array(embeddings)

    def get_feature_embeddings(self,image,layer_index=-1):
        image = cv2.resize(image, (self.model.input_shape[2], self.model.input_shape[1]))
        if(layer_index==-1):
            features = self.model.predict(np.expand_dims(image, axis=0))
        else:
            features = self.model_all_layers[layer_index].predict(np.expand_dims(image, axis=0))
        return features[0]

    '''
    Load an image from the specified path.
    '''
    def load_image(self, image_path, convert_rgb=True):
        image = cv2.imread(image_path)
        
        if convert_rgb:
            return self.convert_to_rgb(image)
        
        return image
    '''
    Load an image from the specified path in grayscale.
    '''
    def load_image_gs(self,image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    '''
    Convert an image to RGB format if it is not already.
    '''
    def convert_to_rgb(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

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


    def calculate_entropy(self):
        entropy_array = []
        for image in self.images['image_path']:
            image = self.load_image_gs(image)
            histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
            histogram = histogram / histogram.sum()  # Normalize to get probabilities
            histogram = histogram[histogram > 0]  # Remove zero entries
            entropy_value = -np.sum(histogram * np.log2(histogram))
            entropy_array.append(entropy_value)
        
        self.images['entropy'] = entropy_array

    def calculate_entropy_per_class(self):
        #Verify if entropy values are already calculated
        if 'entropy' not in self.images.columns:
            self.calculate_entropy()
        
        return self.images.groupby('class')[['entropy']].mean().reset_index()

    def calculate_energy(self):
        energy_array = []
        for image in self.images['image_path']:
            image = self.load_image_gs(image)
            f = np.fft.fft2(image)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift)
            energy = np.sum(magnitude_spectrum ** 2) / (image.shape[0] * image.shape[1])
            energy_array.append(energy)
        
        self.images['energy'] = energy_array

    def n_regions_count(self, scale_factor=0.02, color_factor=0.1, area_factor=0.001):
        
        n_regions_array = []
        for img_path in self.images['image_path']:
            image = self.load_image(img_path, convert_rgb=False)    
            
            
            h, w = image.shape[:2]
            total_pixels = h * w

            # --- Dynamic parameters ---
            spatial_radius = int(min(h, w) * scale_factor)       # proportional to smallest image dimension
            color_radius = int(255 * color_factor)               # proportional to color range
            min_region_size = int(total_pixels * area_factor)    # proportional to total area
            
            shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)

            gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num_labels, labels = cv2.connectedComponents(binary)

            unique_labels, counts = np.unique(labels, return_counts=True)

            region_sizes = counts[1:]  # skip background
            valid_regions = region_sizes[region_sizes >= min_region_size]

            num_valid = len(valid_regions)
            n_regions_array.append(num_valid)

        self.images['mean_shift_regions'] = n_regions_array 


    def frequency_factor(self):
        #FrequencyFactor, it is the ratio between the frequency corresponding to the 99% of
        # the image energy and the Nyquist frequency

        if 'energy' not in self.images.columns:
            self.calculate_energy()

        nyquist_frequency = 0.5  # Normalized Nyquist frequency
        frequency_factors = self.images['energy'] / nyquist_frequency
        self.images['frequency_factor'] = frequency_factors

    def calculate_jpeg_compression_ratio(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        ratios = []
        rmses = []
        count = 0
        for name in self.images['image_path']:
            
            original_image = self.select_channel(name, channel=channel)
            

            if(is_edge_processing == True):
                
                original_image = self.edge_processing(original_image, method=edge_method, direction= direction)
                

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encoded_img = cv2.imencode('.jpg', original_image, encode_param)
            
            if(channel=='all'):
                jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
            else:   
                jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)


            cv2.imwrite("./temp.png", original_image)
            cv2.imwrite("./temp.jpg", original_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            original_size = os.path.getsize("./temp.png")
            jpeg_size = os.path.getsize("./temp.jpg")
            compression_ratio =  jpeg_size / original_size

            diff = (original_image.astype(np.float32) - jpeg_image.astype(np.float32)) ** 2
            mse = np.mean(diff)
            rmse = np.sqrt(mse)
            
            
            
            
            
            
            
            ratios.append(compression_ratio)
            rmses.append(rmse)

            count = count + 1
            if(count%500==0):
                print(count)
           
        self.images['jpeg_compression_ratio'] = ratios
        self.images['jpeg_rmse'] = rmses
    
    def jpeg_compression_ratio_per_class(self, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
        #Verify if compression ratios are already calculated
        if 'jpeg_compression_ratio' not in self.images.columns or 'jpeg_rmse' not in self.images.columns:
            self.jpeg_compression_ratio(quality=quality, channel=channel, is_edge_processing=is_edge_processing, edge_method=edge_method, direction=direction)
        
        return self.images.groupby('class')[['jpeg_compression_ratio', 'jpeg_rmse']].mean().reset_index()


    def zipf_rank(self, channel='all'):
        slopes = []
        r_values = []
        for name in self.images['image_path']:
            image = self.select_channel(name, channel=channel)
            values, counts = np.unique(image, return_counts=True)

            # Sort counts in descending order (most frequent = rank 1)
            counts_sorted = np.sort(counts)[::-1]
            ranks = np.arange(1, len(counts_sorted) + 1)

            # Convert to log scale
            log_ranks = np.log10(ranks)
            log_counts = np.log10(counts_sorted)

            # Fit a linear regression: log(f) = a + b*log(r)
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_counts)
            
            if(r_value == 0.0):
                slope = 0.0
            
            slopes.append(slope)
            r_values.append(r_value)
        self.images['zipf_slope'] = slopes
        self.images['zipf_r_value'] = r_values


    def zipf_difference(self, channel='all'):   
        slopes = []
        r_values = []
        for name in self.images['image_path']:
            image = self.select_channel(name, channel=channel)

            shifts = [(-1, -1), (-1, 0), (-1, 1),
                ( 0, -1),          ( 0, 1),
                ( 1, -1), ( 1, 0), ( 1, 1)]
        
            differences = []

            # Compute absolute differences with all neighbors
            for dx, dy in shifts:
                shifted = np.roll(image, shift=(dx, dy), axis=(0, 1))
                diff = np.abs(image - shifted)
                differences.append(diff)

            diffs = np.concatenate([d.flatten() for d in differences])

            # Count occurrences of difference magnitudes (1–255)
            values, counts = np.unique(diffs, return_counts=True)
            valid_mask = (values > 0) & (values <= 255)
            
            
            
            values = values[valid_mask]
            counts = counts[valid_mask]

            if len(values) < 2:
                slopes.append(0.0)
                r_values.append(0.0)
                continue

            # Log transform
            log_values = np.log10(values)
            log_counts = np.log10(counts)

            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_values, log_counts)
            
            if(r_value == 0.0):
                slope = 0.0
            
            slopes.append(slope)
            r_values.append(r_value)


        
        self.images['zipf_diff_slope'] = slopes
        self.images['zipf_diff_r_value'] = r_values


    def edge_density_canny(self, low_threshold=0.11, high_threshold=0.27):
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

        self.images['edge_density'] = density_array
        
    
    
    def quantized_color_set(self, image, bits_per_channel):

        shift = 8 - bits_per_channel
        img_quantized = np.right_shift(image, shift).astype(np.uint16)

        # 3. Combine channels into a single index value per pixel
        # For example, if bits_per_channel=4: 
        #   R: 4 bits, G: 4 bits, B: 4 bits → total 12 bits
        color_indices = (
            (img_quantized[:, :, 0] << (2 * bits_per_channel)) +
            (img_quantized[:, :, 1] << bits_per_channel) +
            img_quantized[:, :, 2]
        )

        # 4. Count unique color indices
        return color_indices
    
    
    
    
    def j_distance(self,colors1, colors2):
        
        if colors1.size == 0 or colors2.size == 0:
            return 0.0
        
        intersection = np.intersect1d(colors1, colors2).size
        union = np.union1d(colors1, colors2).size
        return intersection / union
    
    def keep_relevant_colors(self, color_array, color_count, percent=0.9):
        

    
            order = np.argsort(color_count)[::-1]
            sorted_counts = color_count[order]
            sorted_colors = color_array[order]
            cum = np.cumsum(sorted_counts)
            total = cum[-1]

            cutoff_idx = np.searchsorted(cum, percent * total)
            return sorted_colors[:cutoff_idx + 1]

    def is_similar(self,s,t):
        if(s>=t):
            return 1
        else:
            return 0
    
    
    def edge_distance_function(self,img1_path,img2_path,low_threshold=100,high_threshold=200):
        
        img1 = self.load_image_gs(img1_path)
        img2 = self.load_image_gs(img2_path)

        #resize
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        edges1 = cv2.Canny(img1, low_threshold, high_threshold)
        edges2 = cv2.Canny(img2, low_threshold, high_threshold)

        e1 = edges1 > 0
        e2 = edges2 > 0

        intersection = np.logical_and(e1, e2).sum()
        union = np.logical_or(e1, e2).sum()

    
        distance = 2 * intersection / (e1.sum() + e2.sum()) if (e1.sum() + e2.sum()) > 0 else 0
        return distance
    
    def perceptual_similarity(self,img1_path,img2_path,use_mask=True):
        
        
        

        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        



        ssim_r,ssim_r_map = ssim(img1[:,:,0], img2[:,:,0],full=True)
        ssim_g,ssim_g_map = ssim(img1[:,:,1], img2[:,:,1],full=True)
        ssim_b,ssim_b_map = ssim(img1[:,:,2], img2[:,:,2],full=True)
        
        if(use_mask==True):
            mask = self.edge_mask(img1)
            
            ssim_r = np.mean(ssim_r_map[mask > 0])
            ssim_g = np.mean(ssim_g_map[mask > 0])
            ssim_b = np.mean(ssim_b_map[mask > 0])
        
        ssim_mean = (ssim_r + ssim_g + ssim_b) / 3
        return ssim_mean
    
    
    def hsv_hist_similarity(self,img1_path,img2_path):
        img1 = self.convert_to_hsv(self.load_image(img1_path))
        img2 = self.convert_to_hsv(self.load_image(img2_path))

        hist1 = cv2.calcHist([img1], [0, 1], None, [50,60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1], None, [50,60], [0, 180, 0, 256])
    
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score
    
    def edge_mask(self,img):
        # 1. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Detect edges (Canny)
        edges = cv2.Canny(gray, 100, 200)

        # 3. Optionally thicken edges to cover adjacent pixels
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)

        # 4. Convert binary edge map to mask (white=keep)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        return mask

    
    def hist_similarity(self,img1_path,img2_path):
        
        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)



        hist1 = cv2.calcHist([img1], [0, 1, 2], self.edge_mask(img1), [16,16,16], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], self.edge_mask(img2), [16,16,16], [0, 256, 0, 256, 0, 256])
    
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score
    
    def fft_texture_features(self,img_path):
        img = self.load_image_gs(img_path)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log1p(np.abs(fshift))

        # Compute energy in low, mid, and high frequencies
        h, w = magnitude_spectrum.shape
        cy, cx = h//2, w//2
        r = min(cx, cy)
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

    def color_moments(self,image_path,cs='LAB'):
        
        img = self.load_image(image_path, convert_rgb=False)
        

        if cs == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0

        elif cs == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            img[:, 0] /= 179.0  # H
            img[:, 1] /= 255.0  # S
            img[:, 2] /= 255.0  # V

        elif cs == 'LAB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)
            img[:, 0] /= 100.0                 # L
            img[:, 1] = (img[:, 1] + 128) / 255.0  # a
            img[:, 2] = (img[:, 2] + 128) / 255.0  # b

        else:
            raise ValueError("color_space must be one of ['RGB', 'HSV', 'LAB']")


        # Split channels
        channels = cv2.split(img)
        features = []
        for ch in channels:
            mean = np.mean(ch)
            std = np.std(ch)
            #sk = np.mean(skew(ch)) 
            features.extend([mean, std])

        return np.array(features)

    
    
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
            
    
    def haralick_similarity(self,img1_path,img2_path):
        img1_features = self.haralick_df[self.haralick_df['image_path']==img1_path]['contrast'].values[0]
        img2_features = self.haralick_df[self.haralick_df['image_path']==img2_path]['contrast'].values[0]

        

        return abs(img1_features-img2_features)
    
    
    
    def color_moments_similarity(self,img1_path,img2_path):
        img1_color_moments = self.color_moments(img1_path)
        img2_color_moments = self.color_moments(img2_path)

        similarity = 1 / (1 + np.linalg.norm(img1_color_moments - img2_color_moments))

        return similarity
    
    def similarity_overlap(self,distance_metric="edge",use_mask=False):
        
        classes = np.unique(self.images['class'])
        overlap_array = [[] for _ in range (len(classes))]
        
        if(distance_metric=='edge'):
            distance_function = self.edge_distance_function
        elif(distance_metric=='ssim'):
            distance_function = self.perceptual_similarity
        elif(distance_metric=='hist'):
            distance_function = self.hist_similarity
        elif(distance_metric=='hs_hist'):
            distance_function = self.hsv_hist_similarity
        elif(distance_metric=='color_moments'):
            distance_function = self.color_moments_similarity
        elif(distance_metric=='haralick'):
            distance_function = self.haralick_similarity
            self.haralick_df = self.get_haralick_features()
        else:
            return 
        instance_count = 0
        for name in self.images['image_path']:
            
            

            

            c_count = 0
            for c in classes:
                
                diff_class_indices = np.where(self.images['class'] == c)[0]

                class_sims = [distance_function(name,self.images.iloc[s]['image_path']) for s in diff_class_indices if name != self.images.iloc[s]['image_path']]

                overlap_array[c_count].append(np.mean(class_sims))
                c_count +=1


            instance_count +=1

            print(instance_count)
    
    
        overlap_array = np.array(overlap_array)
        for i in range(len(classes)):
            self.images['class_similarity' + str(i)] = overlap_array[i]
    
    
    def count_unique_colors(self,bits_per_channel,use_mask):
        unique_colors_array = []
        colors_count_array = []
        
        for name in self.images['image_path']:
            image = self.load_image(name)

        
                
            
            colors = self.quantized_color_set(image,bits_per_channel)

            if(use_mask):
                mask = self.edge_mask(image)
                colors = colors[mask>0]

            # Count how many times each color is present in the image
            unique_colors, counts = np.unique(colors, return_counts=True)
            colors_count_array.append(counts)
            unique_colors_array.append(unique_colors)
        
        return unique_colors_array,colors_count_array
    
    def color_overlap(self,bits_per_channel=4,use_mask=False):
        
        
        unique_colors_array, colors_count_array = self.count_unique_colors(bits_per_channel,use_mask)

        row = 0
        
    
        classes = np.unique(self.images['class'])
        inter_class_colors = [[] for _ in range (len(classes))]
        for name in self.images['image_path']:
            
            
            this_colors = unique_colors_array[row]
            this_counts = colors_count_array[row]
        
            c_count = 0
            for c in classes:
                
                diff_class_indices = np.where(self.images['class'] == c)[0]

                
            
                other_class_sims = [self.j_distance(self.keep_relevant_colors(this_colors,this_counts), 
                                                    self.keep_relevant_colors(unique_colors_array[s],colors_count_array[s])
                                                    ) for s in diff_class_indices if name != self.images.iloc[s]['image_path']]
                
                sims = other_class_sims
                #sims = [self.is_similar(s,0.15) for s in other_class_sims]

                inter_class_colors[c_count].append(np.mean(sims))
                c_count +=1

            row +=1

        inter_class_colors = np.array(inter_class_colors)
        for i in range(len(classes)):
            self.images['class_colors_' + str(i)] = inter_class_colors[i]

    def zipf_difference_per_class(self, channel='all'):
        #Verify if compression ratios are already calculated
        if 'zipf_diff_slope' not in self.images.columns or 'zipf_diff_r_value' not in self.images.columns:
            self.zipf_difference(channel=channel)
        
        return self.images.groupby('class')[['zipf_diff_slope', 'zipf_diff_r_value']].mean().reset_index()
    
    
    def zipf_rank_per_class(self, channel='all'):
        #Verify if compression ratios are already calculated
        if 'zipf_slope' not in self.images.columns or 'zipf_r_value' not in self.images.columns:
            self.zipf_rank(channel=channel)
        
        return self.images.groupby('class')[['zipf_slope', 'zipf_r_value']].mean().reset_index()

    def _knn_density_estimation(self, query_points, reference_points,k_neighbors=5):

        if len(reference_points) < k_neighbors:
            k = len(reference_points)
        else:
            k = k_neighbors
            
        # Fit KNN on reference points
        knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
        knn.fit(reference_points)
        
        # Find distances to k-nearest neighbors for each query point
        distances, _ = knn.kneighbors(query_points)
        
        # Volume of hypercube in d-dimensional space
        d = reference_points.shape[1]
        volumes = (2 * distances[:, -1]) ** d  # diameter = 2 * radius
        
        # Avoid division by zero
        volumes = np.maximum(volumes, 1e-10)
        
        # Density estimation: k / (n * volume)
        n_ref = len(reference_points)
        densities = k / (n_ref * volumes)
        
        return densities
    
    def compute_pairwise_similarity(self, embeddings_i, embeddings_j,n_samples=50):
        #project the images into 2D using t-SNE
        

        
        mc_samples = embeddings_i[np.random.choice(len(embeddings_i), min(n_samples, len(embeddings_i)), replace=False)]
        
        # Estimate P(phi(x)|C_j) for each Monte Carlo sample from C_i
        p_j_given_i = self._knn_density_estimation(mc_samples, embeddings_j)
        
        # Monte Carlo approximation: E_{P(phi(x)|C_i)}[P(phi(x)|C_j)]
        similarity = np.mean(p_j_given_i)
        
        return similarity
    
    def compute_similarity_matrix_S(self, data, n_samples=50):

    
        class_labels = self.images['class'].unique()
        K = len(class_labels)
        similarity_matrix_S = np.zeros((K, K))
        
        print("Computing similarity matrix S...")
        for i in range(K):
            for j in range(K):
                
                embeddings_i = data[self.images['class'] == class_labels[i]]
                embeddings_j = data[self.images['class'] == class_labels[j]]
                similarity_matrix_S[i, j] = self.compute_pairwise_similarity(embeddings_i, embeddings_j,n_samples=n_samples)

        
        return similarity_matrix_S

    def compute_adjacency_matrix_W(self, similarity_matrix_S):
        
        K = similarity_matrix_S.shape[0]
        adjacency_matrix_W = np.zeros((K, K))
        
        print("Computing adjacency matrix W...")
        for i in range(K):
            for j in range(K):
                if i == j:
                    adjacency_matrix_W[i, j] = 1.0  # Self-similarity
                else:
                    # Bray-Curtis similarity: 1 - (sum|S_ik - S_jk|) / (sum|S_ik + S_jk|)
                    numerator = np.sum(np.abs(similarity_matrix_S[i, :] - similarity_matrix_S[j, :]))
                    denominator = np.sum(np.abs(similarity_matrix_S[i, :] + similarity_matrix_S[j, :]))
                    
                    if denominator == 0:
                        adjacency_matrix_W[i, j] = 0.0
                    else:
                        adjacency_matrix_W[i, j] = 1.0 - (numerator / denominator)
        
        # Ensure symmetry
        adjacency_matrix_W = (adjacency_matrix_W + adjacency_matrix_W.T) / 2
        
        return adjacency_matrix_W
    
    def compute_laplacian_matrix_L(self, adjacency_matrix_W):
    
        # Degree matrix: D_ii = sum_j W_ij
        degree_matrix_D = np.diag(np.sum(adjacency_matrix_W, axis=1))
        
        # Laplacian matrix: L = D - W
        laplacian_matrix_L = degree_matrix_D - adjacency_matrix_W
        
        
        return laplacian_matrix_L, degree_matrix_D
    
    def compute_spectrum(self, laplacian_matrix_L):

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigh(laplacian_matrix_L)
        
        # Sort by eigenvalues (ascending)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    

    def compute_csg_complexity(self, eigenvalues):
        
        K = len(eigenvalues)
        
        
        normalized_eigengaps = np.zeros(K-1)
        for i in range(K-1):
            delta_lambda = eigenvalues[i+1] - eigenvalues[i]
            normalized_eigengaps[i] = delta_lambda / (K - i)
        
        
        cumulative_max = np.zeros_like(normalized_eigengaps)
        current_max = 0
        for i in range(len(normalized_eigengaps)):
            current_max = max(current_max, normalized_eigengaps[i])
            cumulative_max[i] = current_max
        
       
        csg_score = np.sum(cumulative_max)
        
        self.normalized_eigengaps = normalized_eigengaps
        self.cumulative_max = cumulative_max
        self.csg_score = csg_score
        
        return csg_score, normalized_eigengaps, cumulative_max
    

    

    
    def compute_normalized_matrices(self, X, y):
        
        n_samples, n_features = X.shape
        classes = np.unique(y)
        
        
        # Global mean
        global_mean = np.mean(X, axis=0)
        
        # Initialize scatter matrices
        S_w_hat = np.zeros((n_features, n_features))
        S_b_hat = np.zeros((n_features, n_features))
        
        total_samples = n_samples
        
        for cls in classes:
            # Get samples for current class
            class_mask = (y == cls)
            X_class = X[class_mask]
            m_i = len(X_class)  # Number of samples in class i
            
                
            # Class mean
            class_mean = np.mean(X_class, axis=0)
            
            # Normalized within-class scatter for this class
            centered_class = X_class - class_mean
            S_w_hat += (1 / m_i) * centered_class.T @ centered_class
            
            # Normalized between-class scatter for this class
            mean_diff = class_mean - global_mean
            weight = m_i / total_samples
            S_b_hat += weight * np.outer(mean_diff, mean_diff)
        
        return S_w_hat, S_b_hat
    
    
    
    def compute_m_sep_direct(self, S_w_hat, S_b_hat):
        
        try:
            # Solve generalized eigenvalue problem: S_w_hat^{-1} S_b_hat v = λ v
            eigenvalues, eigenvectors = eigh(S_b_hat, S_w_hat)
            
            # Find largest eigenvalue (M_sep value)
            max_idx = np.argmax(eigenvalues)
            m_sep = eigenvalues[max_idx]
            
            
            return m_sep
            
        except np.linalg.LinAlgError:
            
            S_w_pinv = np.linalg.pinv(S_w_hat)
            matrix = S_w_pinv @ S_b_hat
            eigenvalues = np.linalg.eigvals(matrix)
            m_sep = np.max(np.real(eigenvalues))
            return m_sep
            


    def compute_m_sep(self,emb_type='CNN_tsne', layer_index=-1, reduction_type=None, reduction_method=None):
        
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





    def compute_spectral_metrics(self, eigenvalues):
    
        K = len(eigenvalues)
        
        # Basic spectral metrics
        spectral_span = eigenvalues[-1] - eigenvalues[0]
        spectral_gap = eigenvalues[1] - eigenvalues[0]  # λ₁ - λ₀
        
        # Area under eigenvalue curve
        area_under_curve = np.sum(eigenvalues)
        
        # Find the largest eigengap and its position
        eigengaps = np.diff(eigenvalues)
        max_eigengap_idx = np.argmax(eigengaps)
        max_eigengap = eigengaps[max_eigengap_idx]
        
        # Compute CSG
        csg_score, normalized_eigengaps, cumulative_max = self.compute_csg_complexity(eigenvalues)
        
        metrics = {
            'csg_score': csg_score,
            'spectral_span': spectral_span,
            'spectral_gap': spectral_gap,
            'area_under_curve': area_under_curve,
            'max_eigengap': max_eigengap,
            'max_eigengap_position': max_eigengap_idx,
            'normalized_max_eigengap': max_eigengap / (K - max_eigengap_idx),
            'num_classes': K,
            'eigenvalues': eigenvalues,
            'normalized_eigengaps': normalized_eigengaps,
            'cumulative_max': cumulative_max
        }
        
        return metrics


    def embed_images(self, emb_type, layer_index=-1):
        
        
        #check if models are loaded if the chosen emb_type requires it
        # Return with warning if model is not loaded
        if(emb_type in ['CNN','CNN_tsne','CNN_random'] and not hasattr(self, 'model')):
            print("Model not loaded. Please load a model before extracting CNN embeddings.")
            return None 

        if(emb_type == 'current'):
            if(self.feature_embeddings is None):
                print("No current embeddings found.")
                return None
            return self.feature_embeddings
        
        
        if(emb_type == 'raw'):

            feature_embeddings = self.images['image_path'].apply(lambda x:  cv2.resize(self.load_image(x), (self.model.input_shape[2], self.model.input_shape[1])).flatten()).tolist()
            self.feature_embeddings = np.array(feature_embeddings)
            
            
        elif(emb_type == 'tsne'):
            feature_embeddings = self.images['image_path'].apply(lambda x:  cv2.resize(self.load_image(x), (self.model.input_shape[2], self.model.input_shape[1])).flatten()).tolist()
            feature_embeddings = np.array(feature_embeddings)
            tsne = TSNE(n_components=2, random_state=42)
            self.feature_embeddings = tsne.fit_transform(feature_embeddings)

        elif(emb_type == 'CNN'):
            self.feature_embeddings = self.get_feature_embeddings_all(layer_index=layer_index).reshape(len(self.images), -1)
            

        elif(emb_type == 'CNN_tsne'):
            self.feature_embeddings = self.get_feature_embeddings_all(layer_index=layer_index).reshape(len(self.images), -1)
            tsne = TSNE(n_components=2, random_state=42)
            self.feature_embeddings = tsne.fit_transform(self.feature_embeddings)
        
        elif(emb_type == 'CNN_pca'):
            self.feature_embeddings = self.get_feature_embeddings_all(layer_index=layer_index).reshape(len(self.images), -1)
            pca = PCA(n_components=50)
            self.feature_embeddings = pca.fit_transform(self.feature_embeddings)
        
        elif(emb_type == 'random'):
            feature_embeddings = self.images['image_path'].apply(lambda x:  cv2.resize(self.load_image(x), (self.model.input_shape[2], self.model.input_shape[1])).flatten()).tolist()
            feature_embeddings = np.array(feature_embeddings)
            
            n_features = feature_embeddings.shape[1]
            n_components = 50 

            r = np.random.normal(0, 1/np.sqrt(n_components), (n_features, n_components))
            self.feature_embeddings = feature_embeddings @ r
        elif(emb_type == 'CNN_random'):
            self.feature_embeddings = self.get_feature_embeddings_all(layer_index=layer_index).reshape(len(self.images), -1)
            n_features = self.feature_embeddings.shape[1]
            n_components = 50 

            r = np.random.normal(0, 1/np.sqrt(n_components), (n_features, n_components))
            self.feature_embeddings = self.feature_embeddings @ r
        elif(emb_type == 'efficient_net'):
            print("Extracting EfficientNet-Lite0 embeddings...")
            model = EfficientNetLite0EmbeddingModel()
            feature_embeddings = self.images['image_path'].apply(lambda x:  model(self.load_image(x)).tolist())
            
            
            
            


            
            self.feature_embeddings = np.array(feature_embeddings.tolist()).astype(np.float32)
            print("EfficientNet-Lite0 embeddings extracted.")

        elif(emb_type == 'efficient_net_pca'):
            model = EfficientNetLite0EmbeddingModel()
            feature_embeddings = self.images['image_path'].apply(lambda x:  model(self.load_image(x)).tolist())

            #ensure it's a numpy array with float values
            feature_embeddings = np.array(feature_embeddings.tolist()).astype(np.float32)
            #feature_embeddings = self.normalize_embs(feature_embeddings)


            pca = PCA(n_components=100)
            self.feature_embeddings = pca.fit_transform(feature_embeddings)
        
        elif(emb_type == 'mobile_net_pca'):
            model = MobileNetV3EmbeddingModel()
            feature_embeddings = self.images['image_path'].apply(lambda x:  model(self.load_image(x)).tolist())

            #ensure it's a numpy array with float values
            feature_embeddings = np.array(feature_embeddings.tolist()).astype(np.float32)
            #feature_embeddings = self.normalize_embs(feature_embeddings)

            pca = PCA(n_components=50)
            self.feature_embeddings = pca.fit_transform(feature_embeddings) 
            
        return self.feature_embeddings
    

    def dim_reduction(self,emb,method='pca',n_compoments=50,custom_method=None): 
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
        #normalize
        embs_min = np.array(embs.min(axis=0))
        embs_max = np.array(embs.max(axis=0))

        #check if max equals min to avoid division by zero
        zro_mask = (embs_max - embs_min) == 0
        embs_max[zro_mask] = 1
        embs_min[zro_mask] = 0

        embs = (embs - embs_min) / (embs_max - embs_min)

        return embs

    def tabular_measure(self, layer_index=-1, reduction_type=None, reduction_method=None, emb_type='CNN_tsne', measure='kdn'):
        

        embs = self.embed_images(emb_type=emb_type, layer_index=layer_index)
        if embs is None:
            return None
        
        if reduction_type is not None:
            embs = self.dim_reduction(embs,method=reduction_type,custom_method=reduction_method,n_compoments=2)
            self.feature_embeddings = embs
    
        
        dataset_dic = {'X': embs, 'y': self.images['class'].values}

        if(measure=='n2'):
            comp_value = Complexity(file_type='array',dataset=dataset_dic).N2(imb=True)
        if(measure=='kdn'):
            comp_value = Complexity(file_type='array',dataset=dataset_dic).kDN(imb=True)
        if(measure=='lsc'):
            comp_value = Complexity(file_type='array',dataset=dataset_dic).LSC(imb=True)


        layer_index_str = str(layer_index) if layer_index >= 0 else "final" 
        self.overlap_measures_dic[measure + '_' + emb_type + '_layer' + str(layer_index_str)] = comp_value

        return comp_value

    def csg_measure(self, layer_index=-1,emb_type='CNN_tsne',n_samples=50,summarize_results=True,  reduction_type=None,reduction_method=None):
        
        
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

        spectral_metrics = self.compute_spectral_metrics(eigenvalues)
        self.spectral_metrics = spectral_metrics
        

        layer_index_str = str(layer_index) if layer_index >= 0 else "final" 
        self.overlap_measures_dic['csg_' + emb_type + '_layer' + str(layer_index_str)] = spectral_metrics['csg_score']

        if(summarize_results == True):
            return spectral_metrics['csg_score']
        else:
            return spectral_metrics
        
    def plot_overlap_measures(self,cls='average'):
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
        #intrinsic measures are all the columns added to the image dataframe 
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
        

        if(embs is not None):
            embeddings = embs.flatten().reshape(len(self.images), -1)
        else:
            #check if embeddings are already calculated
            if not hasattr(self, 'feature_embeddings'):
                self.get_feature_embeddings_all()
                embeddings = self.get_feature_embeddings_all().flatten().reshape(len(self.images), -1)

            else:
                embeddings = self.feature_embeddings.flatten().reshape(len(self.images), -1)

        if(embeddings.shape[1]>2):
            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(embeddings)
        else:
            tsne_results = embeddings
            
        #normalize
        #tsne_min = tsne_results.min(axis=0)
        #tsne_max = tsne_results.max(axis=0)
        #tsne_results = (tsne_results - tsne_min) / (tsne_max - tsne_min)

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
        
        #check the columns self.images already has
        existing_columns = self.images.columns.tolist()
        
        existing_columns.pop(0)
        existing_columns.pop(0)
        return self.images.groupby('class')[existing_columns].mean().reset_index()
    

    
    def visualize_metrics_per_class(self, metric_name):
        class_means = self.get_all_values_per_class()
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_means['class'], class_means[metric_name])
        
        plt.xlabel('Class')
        plt.ylabel(metric_name)
        plt.title(f'Average {metric_name} per Class')
    
        plt.show()
    
    
    
    def plot_feature_distribution(self, df, feature_x, feature_y, class_col="class"):
        
        plt.figure(figsize=(8, 6))
        classes = df[class_col].unique()

        for cls in classes:
            subset = df[df[class_col] == cls]
            plt.scatter(subset[feature_x], subset[feature_y], label=cls, alpha=0.7)
        


        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.title(f"Feature Distribution: {feature_x} vs {feature_y}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.show()


dataset = "shapes_dataset"
folder = "./" + dataset +  "/train/"

classes = ["Circle","Square","Triangle"]

complexity_train = ImageComplexity(folder,keep_classes=classes)


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

#------------------------- Viz Examples -------------------------------------
'''

complexity_train.csg_measure(emb_type="efficient_net",n_samples=50, reduction_type='pca')
complexity_train.tabular_measure(emb_type='efficient_net',measure='kdn',reduction_type='pca')
complexity_train.compute_m_sep(emb_type='efficient_net', reduction_type='pca')
complexity_train.plot_overlap_measures()

complexity_train.plot_tsne(embs=complexity_train.feature_embeddings)


#complexity_train.calculate_energy()


#complexity_train.calculate_jpeg_compression_ratio()
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

