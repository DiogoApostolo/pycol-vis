from pycol_vis import ImageComplexity

import os
import time
if __name__ == "__main__":

    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    if not os.path.exists(folder):
        raise ValueError("Folder " + folder + " does not exist. Please download the dataset from https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia and place it in the correct location.")

    classes = ["PNEUMONIA","NORMAL"]

    complexity = ImageComplexity(folder,keep_classes=classes,number_per_class=50)
    
    #time the execution of the overlap measures calculation
    start_time = time.time()
    complexity.all_intrinsic_measures()

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("Intrinsic measures calculated:")

    print(complexity.images)