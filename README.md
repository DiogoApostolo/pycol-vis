[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# ImComPy: Python Image Complexity Library

The Python Image Complexity Library (`ImComPy`) assembles a set of data complexity measures associated with image data. 

Dataset complexity poses a significant challenge in classification tasks, especially in real-world applications where a combination of factors such as class overlap, data imbalance, noise, and dimensionality can jeopardize a machine learning algorithm's performance. 

The seminal work of \cite{hoBasu} has leveraged a set of measures devoted to estimating the difficulty level of a tabular classification problem. However, since these complexity measures were designed for tabular datasets, they cannot be directly applied to images. Furthermore, while comprehensive software packages for complexity analysis exist for tabular data such as [pycol](https://github.com/DiogoApostolo/pycol/tree/new_main) , [dcol](https://github.com/nmacia/dcol) , [ECoL](https://github.com/lpfgarcia/ECoL), [ImbCoL](https://github.com/victorhb/ImbCoL), [SCoL](https://github.com/lpfgarcia/SCoL), and [mfe](https://github.com/rivolli/mfe) no equivalent, standardized toolkit exists for image datasets. 

The lack of dedicated image measures and the absence of supporting software, have created a significant gap in our understanding of image complexity, despite the importance of image data in areas such as healthcare, security, remote sensing, and autonomous systems. Our work aims to address this gap directly by introducing a comprehensive package for this purpose. In particular, the ImComPy package distinguishes itself by categorizing image metrics into two distinct complexity families: 


* Intrinsic: comprised of metrics to quantify the difficulty of individual images, based image properties such as color, entropy and edge density.
* Overlap: focusing on class separability and complexity between classes, of a binary or multiclass image dataset.

| Category  | Name                                                   | Acronym     | Range                | Reference |
|-----------|--------------------------------------------------------|-------------|----------------------|-----------|
| Overlap   | Cumulative Spectral Gradient                           | CSG         | 0–∞                  | \cite{image_complexity2} |
| Overlap   | Area Under Laplacian Spectrum                          | AULS        | 0–∞                  | \cite{AugCSG} |
| Overlap   | Cumulative Maximum Scaled Area Under Laplacian Spectrum| cmsAULS     | 0–∞                  | \cite{AugCSG} |
| Overlap   | Class Separability                                     | m-sep       | 0–1                  | \cite{lda_measure} |
| Overlap   | In-Class Variability                                   | m-var       | 0–1                  | \cite{lda_measure} |
| Intrinsic | JPEG Compression Ratio                                 | JPEG        | 0–1                  | \cite{image_complexity5} |
| Intrinsic | Fractal Compression                                    | Fractal     | 0–1                  | \cite{image_complexity5} |
| Intrinsic | Entropy                                                | H           | 0–1                  | \cite{image_complexity1} |
| Intrinsic | Canny Edge Density                                     | CED         | 0–1                  | \cite{image_complexity3} |
| Intrinsic | Sobel Edge Density                                     | SED         | 0–1                  | \cite{image_complexity3} |
| Intrinsic | Color Average                                          | Color Avg.  | [0–1, 0–1, 0–1]      | \cite{image_complexity1} |
| Intrinsic | Unique Colors                                          | #Colors     | 1–∞                  | \cite{image_complexity3} |
| Intrinsic | Zipf Rank                                              | Zipf        | 0–1                  | \cite{image_complexity5} |
| Intrinsic | Haralick Features                                      | haralick    | —                    | \cite{image_complexity3} |
| Intrinsic | FFT Features                                           | fft         | —                    | — |
