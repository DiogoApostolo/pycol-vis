from pycol_vis import ImageComplexity


if __name__ == "__main__":

    dataset = "CovidDataset"
    folder = "./" + dataset +  "/train/"

    classes = ["PNEUMONIA","NORMAL"]

    complexity = ImageComplexity(folder,keep_classes=classes,number_per_class=50)
    

    complexity.all_overlap_measures()

    print("Overlap measures calculated:")

    print(complexity.overlap_measures_dic)