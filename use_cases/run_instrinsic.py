from pycol_vis.image_metrics import ImageComplexity


if __name__ == "__main__":

    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    classes = ["PNEUMONIA","NORMAL"]

    complexity = ImageComplexity(folder,keep_classes=classes,number_per_class=50)
    

    complexity.all_intrinsic_measures()

    print("Intrinsic measures calculated:")

    print(complexity.images)