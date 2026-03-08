[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# ImComPy: Python Image Complexity Library

The Python Image Complexity Library (`ImComPy`) assembles a set of data complexity measures associated with image data. 

Dataset complexity poses a significant challenge in classification tasks, especially in real-world applications where a combination of factors such as class overlap, data imbalance, noise, and dimensionality can jeopardize a machine learning algorithm's performance. 

The seminal work of [1] has leveraged a set of measures devoted to estimating the difficulty level of a tabular classification problem. However, since these complexity measures were designed for tabular datasets, they cannot be directly applied to images. Furthermore, while comprehensive software packages for complexity analysis exist for tabular data such as [pycol](https://github.com/DiogoApostolo/pycol/tree/new_main) , [dcol](https://github.com/nmacia/dcol) , [ECoL](https://github.com/lpfgarcia/ECoL), [ImbCoL](https://github.com/victorhb/ImbCoL), [SCoL](https://github.com/lpfgarcia/SCoL), and [mfe](https://github.com/rivolli/mfe) no equivalent, standardized toolkit exists for image datasets. 

The lack of dedicated image measures and the absence of supporting software, have created a significant gap in our understanding of image complexity, despite the importance of image data in areas such as healthcare, security, remote sensing, and autonomous systems. Our work aims to address this gap directly by introducing a comprehensive package for this purpose. In particular, the ImComPy package distinguishes itself by categorizing image metrics into two distinct complexity families: 

* **Intrinsic**: comprised of metrics to quantify the difficulty of individual images, based image properties such as color, entropy and edge density.
* **Overlap**: focusing on class separability and complexity between classes, of a binary or multiclass image dataset.

## Implemented Measures

The following Table shows the measures implemented in our package divided by family:

| Category  | Name                                                   | Acronym     | Range                | Reference |
|-----------|--------------------------------------------------------|-------------|----------------------|-----------|
| Overlap   | Cumulative Spectral Gradient                           | CSG         | 0–∞                  | [2]  |
| Overlap   | Area Under Laplacian Spectrum                          | AULS        | 0–∞                  | [3]  |
| Overlap   | Cumulative Maximum Scaled Area Under Laplacian Spectrum| cmsAULS     | 0–∞                  | [3]  |
| Overlap   | Class Separability                                     | m-sep       | 0–1                  | [4]  |
| Overlap   | In-Class Variability                                   | m-var       | 0–1                  | [4]  |
| Intrinsic | JPEG Compression Ratio                                 | JPEG        | 0–1                  | [5]  |
| Intrinsic | Fractal Compression                                    | Fractal     | 0–1                  | [5]  |
| Intrinsic | Entropy                                                | H           | 0–1                  | [6]  |
| Intrinsic | Canny Edge Density                                     | CED         | 0–1                  | [7]  |
| Intrinsic | Sobel Edge Density                                     | SED         | 0–1                  | [7]  |
| Intrinsic | Color Average/STD                                      | Color Avg.  | [0–1, 0–1, 0–1]      | [6]  |
| Intrinsic | Unique Colors                                          | #Colors     | 1–∞                  | [7]  |
| Intrinsic | Zipf Rank/Difference                                   | Zipf        | 0–1                  | [5]  |
| Intrinsic | Haralick Features                                      | haralick    | —                    | -  |
| Intrinsic | FFT Features                                           | fft         | —                    | — |

#### Overlap:
* **Cumulative Spectral Gradient (CSG):** Graph-based measure obtained using spectral clustering. Minimum cutting cost of the Similarity Matrix S. 
* **Area Under Laplacian Curve (AULS):**  effect of the Area Under Laplacian Spectrum
* **Cumulative Maximum Scaled Area Under Laplacian Spectrum (cmAULS):** Combines CSG and AULS
* **Class Separability (m-sep):** Inter Class Separability based on LDA Measure
* **In-Class Variability (m-var):** Intra Class Separability based on LDA Measure



#### Instrinsic:
* **JPEG Compression Ratio:** The compression Ratio Achieved by compressing an image to JPEG format (quality is defined as a parameter)
* **Fractal Compression:**
* **Entropy:** The Shannon Entropy of a given image
* **Canny/Sobel Edge Density:** The density of edges of a given image, calculated used either Canny or Sobel Filters. More edge density indicates higher complexity.
* **Color Average/STD:** The average and standard deviation of the colors of a given image, for each individual channel of the image. Image can be converted into different formats.
* **Unique Colors:** The unique colors present in a given image. Image is first quantized to reduce the color space, leaving only the most relevant colors.
* **Zipf Rank/Difference:** Complexity based on Zipf-like statistics and Zipf's Law, which claims that in many natural processes the frequency of something is inversely proportional to its rank. 
* **Haralick Features:** Group of of measures based on haralick features obtained using graycomatrix and graycoprops.
* **FFT Features:** Group of of measures based on fft features. Image is converted to frequency space and the energy in low, mid and high frequency bands is calculated as well as the radial profile. 



## Installation Instructions

All packages required to run ImComPy are listed in the requirements.txt file found in this github repository. To install all needed pacakges run:

`pip install -r requirements.txt`

## Datasets

Various datasets are used in the use cases presented in this page. Below is a list of some of the datasets used to test our package which are also necessary to run the use case files:

* [Shapes dataset](https://data.mendeley.com/datasets/wzr2yv7r53/1): Dataset is composed of 2D 9 geometric shapes, each shape is drawn randomly on a 200x200 RGB image.
* [COVID Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia): Covid Dataset with 3 classes COVID19, PNEUMONIA and NORMAL
* [Fruits Dataset](https://www.kaggle.com/datasets/marquis03/fruits-100?select=train): A dataset contains 100 classes of fruit images.
* [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset): A dataset of handwritten digits
* [Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist): A dataset of 28x28 pixel images of 10 fashion categories (e.g., shirts, shoes, bags)

This package expects the datasets to be stored in the following structure:

- Folder
  - Class_1
    - img1.png
    - img2.png
  - Class_2
    - img1.png
    - img2.png

### Basic Usage

This section shows some basic syntax of how to use our package. Specifically, how to load a dataset, parameterize the setup and how to run the different complexity measures.

```python

# Load the Dataset Stored in the Fruits folder, keeping only the apple and banana class and 100 samples (selected randomly) from each class.
comp = ImageComplexity('Fruits',
           keep_classes=['apple',
           'banana'],
           number_per_class=100)

#Calculate the CSG overlap Measure and the JPEG Compression measure and print them to the user
print(comp.csg_measure())
print(comp.jpeg_compression_ratio())


#Example of the CSG parameters, specifying a specific embedding and how many samples to use to estimate probability.
comp.csg_measure(
    emb_type="mobile_net",
    n_samples=50
)



```


### Use Cases

A collection of Use Cases is shown are provided in the use_cases folder. The provided use cases use the datasets described previously and display how our package can be used in practice to extract valuable insights from image datasets. 

In particular de use case folder includes the following files:

* **model_selection.py:** A Use Case showing how the overlap measures in our package can be used to inform model selection
* **sample_selection.py:** A Use Case showing how the intrinsic measures can be used to reduce the dataset size, selecting only the most relevant samples
* **feature_selection.py:** A Use Case showing how the overlap measures can be used to reduce the embedding feature space, without losing classfication performance.
* **viz_options.py:** A Use Case displaying the different visualization options present in our package

## References

[1] Ho, T. K., & Basu, M. (2002).  
Complexity Measures of Supervised Classification Problems.  
IEEE Transactions on Pattern Analysis and Machine Intelligence, 24(3), 289–300.  
https://doi.org/10.1109/34.990132

[2] Branchaud-Charron, F., Achkar, A., & Jodoin, P.-M. (2019).  
Spectral Metric for Dataset Complexity Assessment.  
arXiv:1905.07299. https://arxiv.org/abs/1905.07299

[3] Li, G., Togo, R., Ogawa, T., & Haseyama, M. (2022).  
Dataset complexity assessment based on cumulative maximum scaled area under Laplacian spectrum.  
Multimedia Tools and Applications, 81(22), 32287–32303.  
https://doi.org/10.1007/s11042-022-13027-3

[4] Cho, H., & Lee, S. (2021).  
Data Quality Measures and Efficient Evaluation Algorithms for Large-Scale High-Dimensional Data.  
Applied Sciences, 11(2), 472.  
https://doi.org/10.3390/app11020472

[5] Machado, P., Romero, J., Nadal, M., Santos, A., Correia, J., & Carballal, A. (2015).  
Computerized measures of visual complexity.  
Acta Psychologica, 160, 43–57.  
https://doi.org/10.1016/j.actpsy.2015.06.005

[6] Rahane, A. A., & Subramanian, A. (2020).  
Measures of Complexity for Large Scale Image Datasets.  
arXiv:2008.04431. https://arxiv.org/abs/2008.04431

[7] Corchs, S. E., Ciocca, G., Bricolo, E., & Gasparini, F. (2016).  
Predicting Complexity Perception of Real World Images.  
PLOS ONE, 11(6).  
https://doi.org/10.1371/journal.pone.0157986









