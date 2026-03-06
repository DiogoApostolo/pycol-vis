[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)

# ImComPy: Python Image Complexity Library

The Python Image Complexity Library (`ImComPy`) assembles a set of data complexity measures associated with image data. 

Dataset complexity poses a significant challenge in classification tasks, especially in real-world applications where a combination of factors such as class overlap, data imbalance, noise, and dimensionality can jeopardize a machine learning algorithm's performance. 

The seminal work of \cite{hoBasu} has leveraged a set of measures devoted to estimating the difficulty level of a tabular classification problem. However, since these complexity measures were designed for tabular datasets, they cannot be directly applied to images. Furthermore, while comprehensive software packages for complexity analysis exist for tabular data such as [pycol](https://github.com/DiogoApostolo/pycol/tree/new_main) , [dcol](https://github.com/nmacia/dcol) , [ECoL](https://github.com/lpfgarcia/ECoL), [ImbCoL](https://github.com/victorhb/ImbCoL), [SCoL](https://github.com/lpfgarcia/SCoL), and [mfe](https://github.com/rivolli/mfe) no equivalent, standardized toolkit exists for image datasets. 
