from pycol_vis.image_metrics import ImageComplexity
import os

'''
Visualization use case example. In this example we show how the visualize_measure_distribution method can be used
analyse a dataset.

Images are grouped according to low, medium, and high values of a selected measure, enabling a qualitative inspection of how the measure correlates with visual properties of the images.

Download the dataset at https://www.kaggle.com/datasets/marquis03/fruits-100

OR

Use the Fruit_dataset.zip in this repo


'''

if __name__ == "__main__":


    dataset = "Fruit_dataset"
    folder = "./" + dataset +  "/train/"

    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://www.kaggle.com/datasets/marquis03/fruits-100 and place it in the correct location.")

    classes = ["apple","banana"]

    complexity_train = ImageComplexity(folder,keep_classes=classes)

    complexity_train.entropy_measure()
    complexity_train.visualize_measure_distribution(by_class="True")


    list_of_images = [folder + "apple\\0.jpg",folder + "apple\\2.jpg"]


    complexity_train.entropy_measure()
    complexity_train.visualize_specific_images(image_list=list_of_images)