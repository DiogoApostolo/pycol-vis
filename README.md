[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# ImComPy: Python Image Complexity Library

The Python Image Complexity Library (`ImComPy`) assembles a set of data complexity measures associated with image data. 

Dataset complexity poses a significant challenge in classification tasks, especially in real-world applications where a combination of factors such as class overlap, data imbalance, noise, and dimensionality can jeopardize a machine learning algorithm's performance. 

The seminal work of [1] \cite{hoBasu} has leveraged a set of measures devoted to estimating the difficulty level of a tabular classification problem. However, since these complexity measures were designed for tabular datasets, they cannot be directly applied to images. Furthermore, while comprehensive software packages for complexity analysis exist for tabular data such as [pycol](https://github.com/DiogoApostolo/pycol/tree/new_main) , [dcol](https://github.com/nmacia/dcol) , [ECoL](https://github.com/lpfgarcia/ECoL), [ImbCoL](https://github.com/victorhb/ImbCoL), [SCoL](https://github.com/lpfgarcia/SCoL), and [mfe](https://github.com/rivolli/mfe) no equivalent, standardized toolkit exists for image datasets. 

The lack of dedicated image measures and the absence of supporting software, have created a significant gap in our understanding of image complexity, despite the importance of image data in areas such as healthcare, security, remote sensing, and autonomous systems. Our work aims to address this gap directly by introducing a comprehensive package for this purpose. In particular, the ImComPy package distinguishes itself by categorizing image metrics into two distinct complexity families: 

* Intrinsic: comprised of metrics to quantify the difficulty of individual images, based image properties such as color, entropy and edge density.
* Overlap: focusing on class separability and complexity between classes, of a binary or multiclass image dataset.

## Implemented Measures

The following Table shows the measures implemented in our package divided by family:

| Category  | Name                                                   | Acronym     | Range                | Reference |
|-----------|--------------------------------------------------------|-------------|----------------------|-----------|
| Overlap   | Cumulative Spectral Gradient                           | CSG         | 0–∞                  | [2] \cite{image_complexity2} |
| Overlap   | Area Under Laplacian Spectrum                          | AULS        | 0–∞                  | [3] \cite{AugCSG} |
| Overlap   | Cumulative Maximum Scaled Area Under Laplacian Spectrum| cmsAULS     | 0–∞                  | [3] \cite{AugCSG} |
| Overlap   | Class Separability                                     | m-sep       | 0–1                  | [4] \cite{lda_measure} |
| Overlap   | In-Class Variability                                   | m-var       | 0–1                  | [4] \cite{lda_measure} |
| Intrinsic | JPEG Compression Ratio                                 | JPEG        | 0–1                  | [\cite{image_complexity5} |
| Intrinsic | Fractal Compression                                    | Fractal     | 0–1                  | \cite{image_complexity5} |
| Intrinsic | Entropy                                                | H           | 0–1                  | \cite{image_complexity1} |
| Intrinsic | Canny Edge Density                                     | CED         | 0–1                  | \cite{image_complexity3} |
| Intrinsic | Sobel Edge Density                                     | SED         | 0–1                  | \cite{image_complexity3} |
| Intrinsic | Color Average/STD                                      | Color Avg.  | [0–1, 0–1, 0–1]      | \cite{image_complexity1} |
| Intrinsic | Unique Colors                                          | #Colors     | 1–∞                  | \cite{image_complexity3} |
| Intrinsic | Zipf Rank/Difference                                   | Zipf        | 0–1                  | \cite{image_complexity5} |
| Intrinsic | Haralick Features                                      | haralick    | —                    | -  |
| Intrinsic | FFT Features                                           | fft         | —                    | — |

#### Overlap:
* **Cumulative Spectral Gradient (CSG):** 
* **Area Under Laplacian Curve (AULS):** 
* **Cumulative Maximum Scaled Area Under Laplacian Spectrum (cmAULS):** 
* **Class Separability (m-sep):** 
* **In-Class Variability (m-var):** 



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
* COVID Dataset:
* Fruits Dataset:
* MNIST:
* MNIST-Fashion:

## Use Cases










