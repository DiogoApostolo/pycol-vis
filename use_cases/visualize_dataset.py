from pycol_vis.image_metrics import ImageComplexity


'''
Visualization use case example. In this example we show how the visualize_measure_distribution method can be used
analyse a dataset.

Images are grouped according to low, medium, and high values of a selected measure, enabling a qualitative inspection of how the measure correlates with visual properties of the images.

Download the dataset at https://www.kaggle.com/datasets/marquis03/fruits-100?select=train

OR

Use the Fruit_dataset.zip in this repo


'''


dataset = "Fruit_dataset"
folder = "./" + dataset +  "/train/"

classes = ["apple","banana"]

complexity_train = ImageComplexity(folder,keep_classes=classes,number_per_class=200)

complexity_train.entropy_measure()
complexity_train.visualize_measure_distribution(by_class="True")