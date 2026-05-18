from .intrinsic_measures import *


class IntrinsicAPI:

    def __init__(self, parent):
        self.parent = parent

    def edge_density_canny(self, low_threshold=0.11, high_threshold=0.27):
        '''
        Calculate the edge density of an image using the Canny edge detection algorithm and add the values to the images dataframe in a new column 'edge_density_canny'.

        Parameters:
        - low_threshold (float): The lower threshold for the Canny edge detection algorithm.
        - high_threshold (float): The upper threshold for the Canny edge detection algorithm.
        '''

        edge_density = edge_density_canny(image_paths=self.parent.images['image_path'],low_threshold=low_threshold,high_threshold=high_threshold)
        self.parent.images['edge_density_canny'] = edge_density

    def edge_density_sobel(self, threshold=0.2):
        '''
        Calculate the edge density of an image using the Sobel edge detection algorithm and add the values to the images dataframe in a new column 'edge_density_sobel'.

        Parameters:
        - threshold (float): A value between 0 and 1 to determine the edge density threshold. The Sobel edge image is normalized to the range [0, 1], and edges are considered present where the normalized value exceeds the threshold.
        '''

        edge_density = edge_density_sobel(image_paths=self.parent.images['image_path'],threshold=threshold)
        self.parent.images['edge_density_sobel'] = edge_density


    def hsv_std(self):
        '''
        Calculate the standard deviation of the color channels of each image in the HSV color space and store the values in the self.images DataFrame under the columns 'H_std', 'S_std', and 'V_std'.
        '''

        H_std, S_std, V_std = hsv_std(image_paths=self.parent.images['image_path'])
        self.parent.images['H_std'] = H_std
        self.parent.images['S_std'] = S_std
        self.parent.images['V_std'] = V_std

    def hsv_mean(self):
        '''
        Calculate the average color of each image in the HSV color space and store the values in the self.images DataFrame under the columns 'H_mean', 'S_mean', and 'V_mean'.
        '''

        H_mean, S_mean, V_mean = hsv_mean(image_paths=self.parent.images['image_path'])
        self.parent.images['H_mean'] = H_mean
        self.parent.images['S_mean'] = S_mean
        self.parent.images['V_mean'] = V_mean

    def rgb_mean(self):
        '''
        Calculate the average color of each image in the RGB color space and store the values in the self.images DataFrame under the columns 'R_mean', 'G_mean', and 'B_mean'.
        '''

        R_mean, G_mean, B_mean = rgb_mean(image_paths=self.parent.images['image_path'])
        self.parent.images['R_mean'] = R_mean
        self.parent.images['G_mean'] = G_mean
        self.parent.images['B_mean'] = B_mean

    def rgb_std(self):
        '''
        Calculate the standard deviation of the RGB channels for each image and store the values in the self.images DataFrame under the columns 'R_std', 'G_std', and 'B_std'.
        '''

        R_std, G_std, B_std = rgb_std(image_paths=self.parent.images['image_path'])
        self.parent.images['R_std'] = R_std
        self.parent.images['G_std'] = G_std
        self.parent.images['B_std'] = B_std

    def entropy_measure(self):
        '''
        Calculate the entropy of each image and store the values in the self.images DataFrame under the column 'entropy'.
        '''

        entropy = entropy_measure(image_paths=self.parent.images['image_path'])
        self.parent.images['entropy'] = entropy


    def energy_measure(self):
        '''
        Calculate the energy of each image and store the values in the self.images DataFrame under the column 'energy'.
        '''

        energy = energy_measure(image_paths=self.parent.images['image_path'])
        self.parent.images['energy'] = energy


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

        n_regions = n_regions(image_paths=self.parent.images['image_path'], scale_factor=scale_factor, color_factor=color_factor, area_factor=area_factor)
        self.parent.images['n_regions'] = n_regions


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

        jpeg_metrics = jpeg_compression_ratio( image_paths=self.parent.images['image_path'],quality=quality,channel=channel,is_edge_processing=is_edge_processing,edge_method=edge_method,direction=direction )
        
        #unzip the list of tuples into two separate lists
        ratio, jpeg_rmse = zip(*jpeg_metrics)

        self.parent.images['jpeg_compression_ratio'] = ratio
        self.parent.images['jpeg_rmse'] = jpeg_rmse


    def zipf_rank(self, channel='all'):
        '''
        Calculate the Zipf rank slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_slope' and 'zipf_r_value'.

        The method computes the frequency of pixel values, ranks them, and performs a linear regression on the log-log scale to determine the slope and R-value of the distribution, which can provide insights into the complexity and structure of the image.


        Parameters:
            - channel (str): The image channel to use for calculating the Zipf rank. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
        '''

        zipf = zipf_rank(image_paths=self.parent.images['image_path'], channel=channel )
        
        #unzip the list of tuples into two separate lists
        zipf_slope, zipf_r_value = zip(*zipf)

        self.parent.images['zipf_slope'] = zipf_slope
        self.parent.images['zipf_r_value'] = zipf_r_value


    def zipf_difference(self, channel='all'):
        '''
        Calculate the Zipf difference slope and r-value for each image and store the values in the self.images DataFrame under the columns 'zipf_diff_slope' and 'zipf_diff_r_value'.

        Parameters:
            - channel (str): The image channel to use for calculating the Zipf rank. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
        '''

        zipf = zipf_difference(image_paths=self.parent.images['image_path'],channel=channel)

        #unzip the list of tuples into two separate lists
        zipf_diff_slope, zipf_diff_r_value = zip(*zipf)

        self.parent.images['zipf_diff_slope'] = zipf_diff_slope
        self.parent.images['zipf_diff_r_value'] = zipf_diff_r_value


    def count_unique_colors(self, bits_per_channel=8, use_mask=False):
        '''
        Count the number of unique colors in each image and store the values in the self.images DataFrame under the column 'unique_colors', with optional quantization and edge masking. The method quantizes the colors of the image to reduce the number of unique colors, 
        making the computation more efficient and counting only the most relevant colors. If use_mask is True, an edge mask is applied to the image before counting unique colors, 
        which can help to focus on the most important regions of the image and reduce noise from irrelevant areas.

        Parameters:
        - bits_per_channel (int): The number of bits to use for quantization per color channel.
        - use_mask (bool): Whether to apply an edge mask to the image before counting unique colors. If True, an edge mask is applied to the image to focus on important regions and reduce noise from irrelevant areas. Default is False.
        '''

        colors_count, unique_colors = count_unique_colors(image_paths=self.parent.images['image_path'],bits_per_channel=bits_per_channel,use_mask=use_mask)
        self.parent.images['unique_colors'] = colors_count
        self.parent.unique_colors_array = unique_colors


    def fft_measures(self):
        '''
        Calculate the FFT-based measures for each image and store the values in the self.images DataFrame under the columns 'fft_entropy', 'fft_energy', and 'fft_bandwidth'.
        '''

        df_fft = fft_measures(image_paths=self.parent.images['image_path'])

        df_fft['class'] = self.parent.images['class'].values
        df_fft['image_path'] = self.parent.images['image_path'].values

        self.parent.images = self.parent.images.merge(df_fft, on=['class', 'image_path'])

    def haralick_measures(self):
        '''
        Calculate the Haralick texture features for each image and store the values in the self.images DataFrame under the columns 'contrast_haralick', 'correlation_haralick', 'energy_haralick', and 'homogeneity_haralick'.
        '''

        df_haralick = haralick_measures(image_paths=self.parent.images['image_path'])
    
        df_haralick['class'] = self.parent.images['class'].values
        df_haralick['image_path'] = self.parent.images['image_path'].values

        self.parent.images = self.parent.images.merge(df_haralick, on=['class', 'image_path'])