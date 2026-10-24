# ==========================================
# intrinsic_measures.py
# ==========================================

import cv2
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from ..utils.utils import load_image, load_image_gs, convert_to_hsv, select_channel, quantized_color_set, edge_mask

from .intrinsic_utils import (
    sobel_edges,
    edge_processing,
    calculate_color_average,
    calculate_color_std,
    fft_texture_features,
    haralick_features
)


def edge_density_canny(image_paths, low_threshold=0.11, high_threshold=0.27):
    '''
    Calculate the edge density of an image using the Canny edge detection algorithm specified in the image_paths lists. The Canny edge detection algorithm is applied to each image, 
    and the edge density is calculated as the proportion of edge pixels to the total number of pixels in the image. The low_threshold and high_threshold parameters control the sensitivity of the edge detection, 
    with values between 0 and 1 representing the lower and upper thresholds for the Canny algorithm, respectively. 
    Images are specified in the image_paths list, and the resulting edge density values are returned as a list.

    Parameters:
    - image_paths (list): List of image file paths.
    - low_threshold (float): The lower threshold for the Canny edge detection algorithm.
    - high_threshold (float): The upper threshold for the Canny edge detection algorithm.


    Returns:
    - list: A list of edge density values for each image.
    '''

    if(low_threshold < 0 or low_threshold > 1):
        raise ValueError("low_threshold must be between 0 and 1.")
    
    if(high_threshold < 0 or high_threshold > 1):
        raise ValueError("high_threshold must be between 0 and 1.")
    
    if(low_threshold >= high_threshold):
        raise ValueError("low_threshold must be less than high_threshold.")
    
    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    
    

    density_array = []

    for name in image_paths:

        image = load_image(name, convert_rgb=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        low = int(low_threshold * 255)
        high = int(high_threshold * 255)

        edges = cv2.Canny(gray, low, high)

        density = np.sum(edges > 0) / edges.size
        density_array.append(density)

    

    return density_array


def edge_density_sobel(image_paths, threshold=0.2):
    '''
    Calculate the edge density of an image using the Sobel edge detection algorithm. Images are first processed with
      the Sobel operator to detect edges, and then the edge density is calculated as the proportion of edge pixels to the total number of pixels in the image. 
      The Sobel edge image is normalized to the range [0, 1], and edges are considered present where the normalized value exceeds the specified threshold.
      Images are specified in the image_paths list, and the resulting edge density values are returned as a list.
    Parameters:
    - image_paths (list): List of image file paths.
    - threshold (float): A value between 0 and 1 to determine the edge density threshold. The Sobel edge image is normalized to the range [0, 1], and edges are considered present where the normalized value exceeds the threshold.

    Returns:
    - list: A list of edge density values for each image.

    '''

    if(threshold < 0 or threshold > 1):
        raise ValueError("threshold must be between 0 and 1.")
    
    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    
    

    density_array = []

    for name in image_paths:

        image = load_image(name, convert_rgb=False)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sobel = sobel_edges(gray, direction='all')

        sobel_normalized = sobel / 255.0

        edges = sobel_normalized > threshold

        density = np.sum(edges) / edges.size

        density_array.append(density)

    return density_array


def hsv_std(image_paths):
    '''
    Calculate the standard deviation of the color channels of each image in the HSV color space specified in the image_paths lists.

    Parameters:
    - image_paths (list): List of image file paths.
    Returns:
    - tuple: A tuple containing lists of standard deviation values for each channel of each image.
    
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    

    H_std, S_std, V_std = [], [], []

    for name in image_paths:

        h, s, v = calculate_color_std(convert_to_hsv(load_image(name)))

        H_std.append(h)
        S_std.append(s)
        V_std.append(v)

    return H_std, S_std, V_std



def hsv_mean(image_paths):
    '''
    Calculate the average color of each image in the HSV color space specified in the image_paths lists.  
    Parameters:
    - image_paths (list): List of image file paths.
    Returns:
    - tuple: A tuple containing lists of average color values for each channel of each image.
    
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    H_mean, S_mean, V_mean = [], [], []

    for name in image_paths:

        h, s, v = calculate_color_average(convert_to_hsv(load_image(name)))

        H_mean.append(h)
        S_mean.append(s)
        V_mean.append(v)

    return H_mean, S_mean, V_mean



def rgb_mean(image_paths):
    '''
    Calculate the average color of each image in the RGB color space specified in the image_paths lists.  

    Parameters:
    - image_paths (list): List of image file paths.
    Returns:
    - tuple: A tuple containing lists of average color values for each channel of each image.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    R_means, G_means, B_means = [], [], []

    for name in image_paths:

        r, g, b = calculate_color_average(load_image(name))

        R_means.append(r)
        G_means.append(g)
        B_means.append(b)

    return R_means, G_means, B_means



def rgb_std(image_paths):

    '''
    Calculate the standard deviation of the RGB channels for each image specified in the image_paths lists.  

    Parameters:
    - image_paths (list): List of image file paths.
    Returns:
    - tuple: A tuple containing lists of standard deviation values for each channel of each image.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    R_std, G_std, B_std = [], [], []

    for name in image_paths:

        r, g, b = calculate_color_std(load_image(name))

        R_std.append(r)
        G_std.append(g)
        B_std.append(b)

    return R_std, G_std, B_std

   

def entropy_measure(image_paths):

    '''
    Calculate the entropy of each image

    Parameters:
    - image_paths (list): List of image file paths.

    Returns:
    - list: A list of entropy values for each image.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")


    entropy_array = []

    for image_path in image_paths:

        image = load_image_gs(image_path)

        histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        histogram = histogram / histogram.sum()
        histogram = histogram[histogram > 0]

        entropy_value = -np.sum(histogram * np.log2(histogram))
        entropy_array.append(entropy_value)

    return entropy_array



def energy_measure(image_paths):

    '''
    Calculate the energy of each image as the average of the squared pixel values, which provides a measure of the intensity and texture of the image. 
    Images are specified in the image_paths list, and the resulting energy values are returned as a list.

    Parameters:
    - image_paths (list): List of image file paths.

    Returns:
    - list: A list of energy values for each image.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    energy_array_spacial = []

    for image_path in image_paths:
        image = load_image_gs(image_path)
        energy_spacial = np.sum(image.astype(np.float64) ** 2) / (image.shape[0] * image.shape[1])
        energy_array_spacial.append(energy_spacial)

    return  energy_array_spacial


def n_regions(image_paths, scale_factor=0.02, color_factor=0.1, area_factor=0.001):
    '''
    Calculate the number of regions in each image using mean shift segmentation. The mean shift algorithm is applied to each image to segment it into regions based on color and spatial proximity.
    Images are specified in the image_paths list, and the resulting number of regions for each image is returned as a list. 
    
    
    Parameters:
    - image_paths (list): List of image file paths.
    - scale_factor (float): A value to determine the spatial radius for mean shift segmentation based on the image dimensions.
    - color_factor (float): A value to determine the color radius for mean shift segmentation based on the image dimensions.
    - area_factor (float):  A value to determine the minimum region size for mean shift segmentation based on the image dimensions.

    Returns:
    - list: A list of region counts for each image.
    '''

    if(scale_factor <= 0 or scale_factor >= 1):
        raise ValueError("scale_factor must be between 0 and 1.")
    if(color_factor <= 0 or color_factor >= 1):
        raise ValueError("color_factor must be between 0 and 1.")
    if(area_factor <= 0 or area_factor >= 1):
        raise ValueError("area_factor must be between 0 and 1.")
    
    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")


    n_regions_array = []

    for img_path in image_paths:

        image = load_image(img_path, convert_rgb=False)

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
        region_sizes = counts[1:]
        valid_regions = region_sizes[region_sizes >= min_region_size]
        num_valid = len(valid_regions)
        n_regions_array.append(num_valid)

    return n_regions_array



def jpeg_compression_ratio(image_paths, quality=90, channel='all', is_edge_processing=False, edge_method='sobel', direction='all'):
    
    '''
    Calculate the JPEG compression ratio for each image in the image_paths list. 
    
    The method compresses each image using JPEG compression at the specified quality level and calculates the compression ratio as the size of the compressed image divided by the size of the original image.

    User may choose to first apply edge processing to the image before compression, which may affect the compression ratio. 
    If edge processing is applied, the user can specify the method and direction for edge detection.

    Parameters:
    - image_paths (list): List of image file paths.
    - quality (int): The quality level for JPEG compression (0 to 100).
    - channel (str): The image channel to use for compression. Options are 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.
    - is_edge_processing (bool): Whether to apply edge processing to the image before compression.
    - edge_method (str): The method to use for edge processing if is_edge_processing is True.
    - direction (str): The direction of edges to calculate for edge processing. Options are 'x' for horizontal edges, 'y' for vertical edges, and 'all' for both.

    Returns:
    - list: A list of tuples containing the compression ratio and RMSE values for each image.
    '''

    if(quality < 0 or quality > 100):
        raise ValueError("quality must be between 0 and 100.")
    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    if(channel not in ['all', 'R', 'G', 'B', 'H', 'S', 'V']):
        raise ValueError("channel must be one of 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.")
    if(is_edge_processing):
        if(edge_method not in ['sobel']):
            raise ValueError("edge_method must be one of 'sobel'.")
        if(direction not in ['x', 'y', 'all']):
            raise ValueError("direction must be one of 'x', 'y', or 'all'.")
    
    
    ratios = []
    rmses = []

    for name in image_paths:

        original_image = select_channel(name, channel=channel)

        if(is_edge_processing):
            original_image = edge_processing(original_image, method=edge_method, direction=direction)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        result, encoded_img = cv2.imencode('.jpg', original_image, encode_param)

        if(not result):
            raise RuntimeError("JPEG encoding failed")

        if(channel == 'all'):
            jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        else:
            jpeg_image = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)

        jpeg_size = len(encoded_img)
        original_size = original_image.size * original_image.itemsize

        compression_ratio = jpeg_size / original_size

        ratios.append(compression_ratio)

        diff = (original_image.astype(np.float32) - jpeg_image.astype(np.float32)) ** 2

        mse = np.mean(diff)
        rmse = np.sqrt(mse)

        rmses.append(rmse)

    return list(zip(ratios, rmses))


def zipf_rank(image_paths, channel='all'):
    '''
    Calculate the Zipf's law slope and R-value for the pixel value distribution of each image specified in the image_paths list. 
    
    The method computes the frequency of pixel values, ranks them, and performs a linear regression on the log-log scale to determine the slope and R-value of the distribution, which can provide insights into the complexity and structure of the image.

    Parameters:
    - image_paths (list): List of image file paths.
    - channel (str): The image channel to use for the calculation.

    Returns:
    - pd.DataFrame: A DataFrame with two new columns: 'zipf_slope' and 'zipf_r_value', containing the Zipf's law slope and R-value for each image, respectively.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    if(channel not in ['all', 'R', 'G', 'B', 'H', 'S', 'V']):
        raise ValueError("channel must be one of 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.")

    slopes = []
    r_values = []

    for name in image_paths:

        image = select_channel(name, channel=channel)

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

    return list(zip(slopes, r_values))


def zipf_difference(image_paths, channel='all'):
    '''
    Calculate the Zipf's difference slope and R-value for the pixel value distribution of each image specified in the image_paths list.

    Parameters:
    - image_paths (list): List of image file paths.
    - channel (str): The image channel to use for the calculation.

    Returns:
    - list: A list of tuples containing the Zipf's difference slope and R-value for each image, respectively.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    if(channel not in ['all', 'R', 'G', 'B', 'H', 'S', 'V']):
        raise ValueError("channel must be one of 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.")

    slopes = []
    r_values = []

    for name in image_paths:

        image = select_channel(name, channel=channel)

        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
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

    return list(zip(slopes, r_values))


def count_unique_colors(image_paths, bits_per_channel=8, use_mask=False):
    '''
    Count the number of unique colors in each image specified in the image_paths list, with optional quantization and edge masking. The method quantizes the colors of the image to reduce the number of unique colors, 
    making the computation more efficient and counting only the most relevant colors. If use_mask is True, an edge mask is applied to the image before counting unique colors, 
    which can help to focus on the most important regions of the image and reduce noise from irrelevant areas.

   

    Parameters:
    - image_paths (list): List of image file paths.
    - bits_per_channel (int): The number of bits to use for quantization per color channel.
    - use_mask (bool): Whether to apply an edge mask to the image before counting unique colors. If True, an edge mask is applied to the image to focus on important regions and reduce noise from irrelevant areas. Default is False.

    Returns:
    - list: A list of the count of unique colors for each image.
    - list: A list of arrays, where each array contains the unique colors for the corresponding image.
    '''

    if(bits_per_channel <= 0 or bits_per_channel > 8):
        raise ValueError("bits_per_channel must be between 1 and 8.")
    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")
    

    unique_colors_array = []
    colors_count_array = []

    for name in image_paths:

        image = load_image(name)

        colors = quantized_color_set(image, bits_per_channel)

        if(use_mask):

            mask = edge_mask(image)

            colors = colors[mask > 0]

        unique_colors, counts = np.unique(colors, return_counts=True)

        colors_count_array.append(len(unique_colors))
        unique_colors_array.append(unique_colors)

   
    return  colors_count_array, unique_colors_array


def fft_measures(image_paths):
    '''
    Compute the FFT features for all images in the dataset specified in the image_paths list. 
    
    
    The method applies the Fast Fourier Transform (FFT) to each image to extract texture features, which can provide insights into the frequency components and patterns present in the images.

    Parameters:
    - image_paths (list): List of image file paths.

    Returns:
    - pd.DataFrame: A DataFrame with the FFT features for each image, containing the columns 'fft_low', 'fft_mid', and 'fft_high'.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    features_array = []

    for img_path in image_paths:

        features = fft_texture_features(img_path)

        features_array.append(features)

    df_fft = pd.DataFrame(
        features_array,
        columns=['fft_low', 'fft_mid', 'fft_high']
    )


   
    return df_fft


def haralick_measures(image_paths):
    '''
    Compute the Haralick texture features for all images in the dataset specified in the image_paths list.

    Parameters:
    - image_paths (list): List of image file paths.

    Returns:
    - pd.DataFrame: A DataFrame with the Haralick features for each image, containing the columns 'contrast_haralick', 'correlation_haralick', 'energy_haralick', and 'homogeneity_haralick'.
    '''

    if(len(image_paths) == 0):
        raise ValueError("image_paths list cannot be empty.")

    features_array = []

    for img_path in image_paths:

        features = haralick_features(img_path)

        features_array.append(features)

    scaler = MinMaxScaler()

    df_haralick = pd.DataFrame(
        scaler.fit_transform(features_array),
        columns=[
            'contrast_haralick',
            'correlation_haralick',
            'energy_haralick',
            'homogeneity_haralick'
        ]
    )

    return df_haralick