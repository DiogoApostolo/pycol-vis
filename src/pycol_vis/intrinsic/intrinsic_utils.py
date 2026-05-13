
import cv2
import numpy as np
from scipy import stats
from skimage.feature import graycomatrix, graycoprops

from ..utils.utils import load_image, load_image_gs, convert_to_hsv, select_channel, quantized_color_set, edge_mask


def sobel_edges(channel, direction='all'):
    '''
    Calculate the Sobel edges for a given image channel and direction.

    Parameters:
    - channel (np.ndarray): The image channel for which to calculate the Sobel edges.
    - direction (str): The direction of edges to calculate. Options are 'x' for horizontal edges, 'y' for vertical edges, and 'all' for both.
    
    Returns:
    - np.ndarray: A NumPy array containing the calculated Sobel edges for the specified channel and direction.
    '''

    if(direction == 'x'):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_scale = cv2.convertScaleAbs(sobel_x)

    if(direction == 'y'):
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        sobel_scale = cv2.convertScaleAbs(sobel_y)

    if(direction == 'all'):
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)

        sobel_all = cv2.magnitude(sobel_x, sobel_y)
        sobel_scale = cv2.convertScaleAbs(sobel_all)

    return sobel_scale


def edge_processing(channel, method='sobel', direction='all'):
    '''
    Apply edge processing to an image channel.
    '''

    if(method == 'sobel'):
        edge_image = sobel_edges(channel, direction=direction)

    return edge_image


def calculate_color_average(image):
    '''
    Calculate the average color of an image.

    Parameters:
    - image (np.ndarray): The image for which to calculate the average color.

    Returns:
    - list: A list containing the average values for each channel.
    '''

    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    return [avg_color[0], avg_color[1], avg_color[2]]


def calculate_color_std(image):
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


def fft_texture_features(img_path):
    '''
    Auxiliary function to get the FFT features for a give image.
    '''

    img = load_image_gs(img_path)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = np.log1p(np.abs(fshift))

    h, w = magnitude_spectrum.shape
    cy, cx = h//2, w//2
    r = min(cx, cy)

    low = magnitude_spectrum[cy-r//4:cy+r//4, cx-r//4:cx+r//4].mean()
    mid = magnitude_spectrum[cy-r//2:cy+r//2, cx-r//2:cx+r//2].mean()
    high = magnitude_spectrum.mean()

    return np.array([low, mid, high])


def haralick_features(image_path):
    '''
    Auxiliary function to calculate the Haralick texture features for a give image.
    '''

    img = load_image_gs(image_path)

    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    glcm = graycomatrix(
        img,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        symmetric=True,
        normed=True
    )

    labels = ('contrast', 'correlation', 'energy', 'homogeneity')

    features = []

    for prop in labels:
        vals = graycoprops(glcm, prop)
        features.append(vals.mean())

    return np.array(features)