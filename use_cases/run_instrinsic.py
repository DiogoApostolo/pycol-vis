from pycol_vis.image_metrics import ImageComplexity

import os

if __name__ == "__main__":

    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia and place it in the correct location.")

    classes = ["PNEUMONIA","NORMAL"]

    complexity = ImageComplexity(folder,keep_classes=classes,number_per_class=50)
    

    complexity.all_intrinsic_measures()

    print("Intrinsic measures calculated:")

    print(complexity.images)