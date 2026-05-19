
import matplotlib
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

matplotlib.use("QtAgg")

from .utils.utils import load_image
from sklearn.decomposition import PCA


import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from .utils.utils import load_image

import pandas as pd
import os

def plot_overlap_measures(overlap_measures, labels=None, cls='average'):
    '''
    Plot the overlap measures stored in a dictionary as a bar chart.
    If cls is set to 'average', the average values of the measures will be plotted.

    Parameters:
    - overlap_measures (dict): Dictionary containing overlap measure names and values.
    - labels (array-like): Class labels corresponding to the measure values.
    - cls (str): The class for which to plot the measures. If 'average',
      the average values of the measures will be plotted.
    '''
    

    if(cls != 'average' and labels is None):
        raise ValueError("labels must be provided when cls is not 'average'")
    if(cls != 'average' and cls not in labels):
        raise ValueError(f"cls '{cls}' not found in labels")
    if(not isinstance(overlap_measures, dict)):
        raise ValueError("overlap_measures must be a dictionary")
    if(len(overlap_measures) == 0):
        raise ValueError("overlap_measures dictionary cannot be empty")
    
    



    measures = list(overlap_measures.keys())
    values = list(overlap_measures.values())

    for i in range(len(values)):
        if(isinstance(values[i], (list, np.ndarray))):
            if(cls=='average'):
                values[i] = np.mean(values[i])
            else:
                if(labels is None):
                    raise ValueError("labels must be provided when cls is not 'average'")
                class_indices = np.where(labels == cls)[0]
                values[i] = np.mean(np.array(values[i])[class_indices])

    plt.figure(figsize=(10, 6))
    plt.bar(measures, values)
    plt.xlabel('Measures')
    plt.ylabel('Value')
    plt.title('Image Overlap Measures')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_intrinsic_measures(images_df):
    '''
    Plot the intrinsic measures stored in the dataframe as a bar chart.

    Parameters:
    - images_df (pd.DataFrame): DataFrame containing image metrics.
    '''

    if(not isinstance(images_df, pd.DataFrame)):
        raise ValueError("images_df must be a pandas DataFrame")
    if('image_path' not in images_df.columns or 'class' not in images_df.columns):
        raise ValueError("images_df must contain 'image_path' and 'class' columns")
    if(len(images_df) == 0):
        raise ValueError("images_df cannot be empty")
    

    intrinsic_measures = [col for col in images_df.columns if col not in ['image_path', 'class']]
    intrinsic_measures_dic = {}

    for measure in intrinsic_measures:
        intrinsic_measures_dic[measure] = images_df[measure].mean()

    measures = list(intrinsic_measures_dic.keys())
    values = list(intrinsic_measures_dic.values())

    plt.figure(figsize=(10, 6))
    plt.bar(measures, values)
    plt.xlabel('Measures')
    plt.ylabel('Value')
    plt.title('Image Intrinsic Measures')  
    plt.xticks(rotation=45)
    plt.tight_layout()  
    plt.show()


def plot_tsne(embeddings, labels, save_image=False, name="tsne_plot.png", folder="./"):
    '''
    Plot a t-SNE visualization of feature embeddings. If embs is not provided, it will use the feature embeddings stored in self.feature_embeddings. If no embeddings are found, it will raise a ValueError. 
    The plot will be saved to the specified folder with the given name if save_image is True.

    Parameters:
    - embeddings (np.ndarray): Feature embeddings.
    - labels (array-like): Class labels.
    - save_image (bool): Whether to save the t-SNE plot as an image file. Default is False.
    - name (str): The name of the image file to save the plot. Default is "tsne_plot.png".
    - folder (str): The folder path where the image file will be saved if save_image is True. Default is "./".
    '''

    if(embeddings is None or len(embeddings) == 0):
        raise ValueError("embeddings cannot be None or empty.")
    if(labels is None or len(labels) == 0):
        raise ValueError("labels cannot be None or empty.")
    if(len(embeddings) != len(labels)):
        raise ValueError("Length of embeddings and labels must match.")
    if(save_image and (not isinstance(name, str) or not isinstance(folder, str))):
        raise ValueError("name and folder must be strings when save_image is True.")
    if(save_image and not os.path.exists(folder)):
        raise ValueError(f"Folder '{folder}' does not exist. Please create it before saving the image.")
    

    embeddings = np.array(embeddings)
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    if(embeddings.shape[1] > 2):
        tsne = TSNE(n_components=2, random_state=42) 
        tsne_results = tsne.fit_transform(embeddings)

    else:
        tsne_results = embeddings

    plt.figure(figsize=(8, 6))
    classes = np.unique(labels)

    for cls in classes:
        subset = tsne_results[np.array(labels) == cls]
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


def visualize_metrics_per_class(images_df, metric_name):
    '''
    Visualize the average values of a specific intrinsic measure for each class as a bar plot.

    Parameters:
    - images_df (pd.DataFrame): DataFrame containing image metrics.
    - metric_name (str): The name of the intrinsic measure to visualize. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure. The method will calculate the average value of the specified measure for each class and create a bar plot to visualize the differences between classes.
    '''

    if(not isinstance(images_df, pd.DataFrame)):
        raise ValueError("images_df must be a pandas DataFrame")
    if('image_path' not in images_df.columns or 'class' not in images_df.columns):
        raise ValueError("images_df must contain 'image_path' and 'class' columns")
    if(metric_name not in images_df.columns):
        raise ValueError(f"metric_name '{metric_name}' not found in images_df columns")
    if(len(images_df) == 0):
        raise ValueError("images_df cannot be empty")
    if(metric_name in ['image_path', 'class']):
        raise ValueError("metric_name cannot be 'image_path' or 'class'")
    if(metric_name not in images_df.columns):
        raise ValueError(f"metric_name '{metric_name}' not found in images_df columns")
    

    existing_columns = images_df.columns.tolist()

    existing_columns.remove('image_path')
    existing_columns.remove('class')

    class_means = images_df.groupby('class')[existing_columns].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.bar(class_means['class'], class_means[metric_name])
    plt.xlabel('Class')
    plt.ylabel(metric_name)
    plt.title(f'Average {metric_name} per Class')
    plt.tight_layout()
    plt.show()


def visualize_measure_distribution(images_df, measure="entropy", n=10, figsize=(15, 6), seed=None, by_class=False):
    '''
    Method to visualize the images in a dataset
    based on their measured complexity.

    Images are presented in 3 Rows,
    corresponding to High, Medium and Low Complexity.

    Parameters:
    - images_df (pd.DataFrame): DataFrame containing image paths and metrics.
    - measure (str): The name of the intrinsic measure to use for visualizing the distribution. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure.
    - n (int): The number of images to display for each complexity level (High, Medium, Low). Default is 10.
    - figsize (tuple): The size of the figure for the plot. Default is (15, 6).
    - seed (int): The random seed for reproducibility when selecting images to display. Default is None, which means that the selection will be random each time the method is called.
    - by_class (bool): Whether to visualize the distribution of the measure separately for each class. If True, the method will create separate plots for each class, showing the distribution of the specified measure within each class. If False, a single plot will be created showing the overall distribution of the measure across all classes. Default is False.
    '''

    if(not isinstance(images_df, pd.DataFrame)):
        raise ValueError("images_df must be a pandas DataFrame")
    if('image_path' not in images_df.columns or 'class' not in images_df.columns):
        raise ValueError("images_df must contain 'image_path' and 'class' columns")
    if(measure not in images_df.columns):
        raise ValueError(f"measure '{measure}' not found in images_df columns")
    if(len(images_df) == 0):
        raise ValueError("images_df cannot be empty")
    if(n <= 0):
        raise ValueError("n must be a positive integer")
    if(seed is not None and not isinstance(seed, int)):
        raise ValueError("seed must be an integer or None")
    if(measure in ['image_path', 'class']):
        raise ValueError("measure cannot be 'image_path' or 'class'")
    if(measure not in images_df.columns):
        raise ValueError(f"measure '{measure}' not found in images_df columns")
    if(figsize is not None and (not isinstance(figsize, tuple) or len(figsize) != 2)):
        raise ValueError("figsize must be a tuple of length 2 or None")
    

    if(seed is not None):
        np.random.seed(seed)

    if(measure not in images_df.columns):
        raise ValueError("Measure " + measure + " not found in dataframe.")

    if(by_class == False):
        classes = [None]

    else:
        classes = images_df['class'].unique()

    for cls in classes:

        if(cls is None):
            df_subset = images_df.copy()
            title_suffix = ""

        else:
            df_subset = images_df[images_df['class'] == cls]
            title_suffix = " - Class '" + cls + "'"

        total_images = len(df_subset)
        n_per_row = min(n, total_images // 3)

        if(n_per_row == 0):
            print("Not enough images to display.")
            continue

        df_sorted = df_subset.sort_values(by=measure)

        low_images = df_sorted.iloc[:n_per_row].sample(n=n_per_row, random_state=seed)
        med_images = df_sorted.iloc[total_images//3 : total_images//3 + n_per_row].sample(n=n_per_row, random_state=seed)
        high_images = df_sorted.iloc[-n_per_row:].sample(n=n_per_row, random_state=seed)

        image_rows = [low_images, med_images, high_images]

        row_labels = ["Low", "Medium", "High"]
        fig, axes = plt.subplots(nrows=3, ncols=n_per_row, figsize=figsize)

        if(n_per_row == 1):
            axes = axes[:, np.newaxis]

        for row_idx, row_df in enumerate(image_rows):
            for col_idx, (_, img_row) in enumerate(row_df.iterrows()):

                img = load_image(img_row['image_path'], convert_rgb=True)

                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis('off')
                axes[row_idx, col_idx].set_title(f"{img_row[measure]:.3f}", fontsize=9)

            fig.text(0.04, 0.5/3 + row_idx/3, row_labels[row_idx], va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')

        plt.suptitle(f"Visualization of '{measure}' measure{title_suffix}", fontsize=14)
        plt.tight_layout(rect=[0.06, 0.03, 1, 0.95])
        plt.show()


def visualize_specific_images(images_df, image_list, measure="entropy", figsize=(15, 3)):
    '''
    Visualize a specific list of images along with their corresponding values for a specified intrinsic measure. The method will create a plot displaying the selected images and annotate each image with its value for the specified measure.

    Parameters:
    - images_df (pd.DataFrame): DataFrame containing image metrics.
    - image_list (list): A list of image paths to visualize. Each path should correspond to an image in the dataset that has a calculated value for the specified measure in the self.images DataFrame.
    - measure (str): The name of the intrinsic measure to display for each image. This should correspond to a column in the self.images DataFrame that contains the calculated values for that measure. The method will annotate each image with its value for this measure.
    - figsize (tuple): The size of the figure for the plot. Default is (15, 3).
    '''

    if(not isinstance(images_df, pd.DataFrame)):
        raise ValueError("images_df must be a pandas DataFrame")
    if('image_path' not in images_df.columns or 'class' not in images_df.columns):
        raise ValueError("images_df must contain 'image_path' and 'class' columns")
    if(measure not in images_df.columns):
        raise ValueError(f"measure '{measure}' not found in images_df columns")
    if(len(images_df) == 0):
        raise ValueError("images_df cannot be empty")
    if(not isinstance(image_list, list) or len(image_list) == 0):
        raise ValueError("image_list must be a non-empty list of image paths")
    if(measure in ['image_path', 'class']):
        raise ValueError("measure cannot be 'image_path' or 'class'")
    if(measure not in images_df.columns):
        raise ValueError(f"measure '{measure}' not found in images_df columns")
    if(figsize is not None and (not isinstance(figsize, tuple) or len(figsize) != 2)):
        raise ValueError("figsize must be a tuple of length 2 or None")
    if(not all(isinstance(path, str) for path in image_list)):
        raise ValueError("All items in image_list must be strings representing image paths")
    if(not all(path in images_df['image_path'].values for path in image_list)):
        raise ValueError("All image paths in image_list must be present in the 'image_path' column of images_df")
    if(not all(os.path.exists(path) for path in image_list)):
        raise ValueError("All image paths in image_list must exist on the filesystem")
    if(len(image_list) > 20):
        print("Warning: Visualizing a large number of images may lead to a cluttered plot. Consider reducing the number of images for better visualization.")
    

    

    n_images = len(image_list)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    if(n_images == 1):
        axes = [axes]

    for i, path in enumerate(image_list):

        img = load_image(path, convert_rgb=True)

        axes[i].imshow(img)
        axes[i].axis("off")

        if(measure is not None and measure in images_df.columns):
            row = images_df[images_df["image_path"] == path]

            if(not row.empty):

                value = row.iloc[0][measure]
                axes[i].set_title(f"{value:.3f}", fontsize=20)

    plt.suptitle("Selected Images Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()