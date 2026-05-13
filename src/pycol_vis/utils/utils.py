

import cv2
import numpy as np
import os
import pandas as pd


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


import torch






def get_average_image_shape(images):
        '''
        Calculate the average image shape (height, width) across all images in the dataset.
        
        Parameters:
        - images (pd.DataFrame): A DataFrame containing the image paths and corresponding class labels, with a column named 'image_path' that contains the paths to the images.

        
        Returns:
        -tuple: A tuple containing the average width, average height, and number of channels (3 for RGB) for the dataset.
        '''
        total_height = 0
        total_width = 0
        count = 0

        print(images.shape)

        for name in images['image_path']:
            image = load_image(name, convert_rgb=False)
            h, w = image.shape[:2]
            total_height += h
            total_width += w
            count += 1

        avg_height = total_height // (count)
        avg_width = total_width // (count)

        return (avg_width, avg_height, 3)
        


def load_images(folder,keep_classes,number_per_class): 
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

    #check if folder exists
    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist.")
    
    if(keep_classes == 'all'):
        keep_classes = os.listdir(folder)

    for class_name in keep_classes:
        class_path = os.path.join(folder, class_name)
        
        if(os.path.isdir(class_path)):
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
    



def load_image(image_path, convert_rgb=True):
    '''
    Load an image from the specified path.

    Parameters:
    - image_path (str): The path to the image file.
    - convert_rgb (bool): Whether to convert the image to RGB format. 

    Returns:
    - np.ndarray: The loaded image as a NumPy array. 
    '''
    image = cv2.imread(image_path)
    
    if(convert_rgb):
        return convert_to_rgb(image)
    
    return image



def load_image_gs(image_path):
    '''
    Load an image from the specified path in grayscale.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - np.ndarray: The loaded image as a NumPy array.
    '''
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def convert_to_rgb(image):
    '''
    Convert an image to RGB format.

    Parameters:
    - image (np.ndarray): The image to convert to RGB format.

    Returns:
    - np.ndarray: The converted image as a NumPy array.
    '''
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def convert_to_hsv(image): 
        '''
        Convert an image to the HSV color space.

        Parameters:
        - image (np.ndarray): The image to convert to the HSV color space.
        Returns:
        - np.ndarray: The converted image in HSV color space as a NumPy array.
        '''
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)



def sample_dataset(images, n_samples_per_class,sample_type='complexity'):
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


    if sample_type == 'random':
        sampled_images = images.groupby('class', group_keys=True).apply(lambda x: x.sample(n=n_samples_per_class, random_state=42),include_groups=False).reset_index(level=0).reset_index(drop=True)
    
    elif sample_type == 'jpeg_compression':
        sampled_images = images.groupby('class', group_keys=True).apply(lambda x: x.nsmallest(n_samples_per_class, 'jpeg_compression_ratio'),include_groups=False).reset_index(level=0).reset_index(drop=True)

    return sampled_images

def select_channel(name, channel='all'):
    


    if(channel=='all'):
        original_image = load_image(name,convert_rgb=False)
    elif(channel=='R'):
        original_image = load_image(name,convert_rgb=False)[:,:,0]
    elif(channel=='G'):
        original_image = load_image(name,convert_rgb=False)[:,:,1]
    elif(channel=='B'):
        original_image = load_image(name,convert_rgb=False)[:,:,2]
    elif(channel=='H'):
        original_image = convert_to_hsv(load_image(name,convert_rgb=False))[:,:,0]
    elif(channel=='S'):
        original_image = convert_to_hsv(load_image(name,convert_rgb=False))[:,:,1]
    elif(channel=='V'):
        original_image = convert_to_hsv(load_image(name,convert_rgb=False))[:,:,2]
    else:
        raise ValueError("Channel must be one of 'all', 'R', 'G', 'B', 'H', 'S', or 'V'.")
    return original_image






def quantized_color_set( image, bits_per_channel):

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



def edge_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200)

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(edges, kernel, iterations=1)

    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
    return mask

#-------------------------------


