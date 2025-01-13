## Overview

### Data

Folder `data` contains the data used in the study, coming from `nih-dataset` repository, which
is one of the biggest open source available data repositories for medical images.

### Data Analysis

Folder `data-analysis` contains the Jupyter Notebooks used to analyze the data and generate the plots
that can be used in the study later. Thanks to this, we can easily find outliers, missing data, and other
data issues that may affect the study.

#### Run Data Analysis

To run the Jupyter Notebooks, you need to have Python installed on your machine.

```bash
$ cd data-analysis
$ python -m notebook
```