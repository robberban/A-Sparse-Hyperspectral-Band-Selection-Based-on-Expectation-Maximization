# A-Sparse-Hyperspectral-Band-Selection-Based-on-Expectation-Maximization

## Experiment 1: MNIST Classification Sparsification

### Objective
The primary aim of this experiment is to validate the effectiveness of sparse loss in a classification context, utilizing the well-known MNIST dataset as a benchmark.

### Location
The experiment is housed within the `MNIST_Sparse_test` directory.

### How to Run
To execute the experiment, navigate to the `MNIST_Sparse_test` folder and run the `MNIST_Test.py` script. This can typically be done through a command line interface with the following command:

```bash
python MNIST_Test.py
```

## Experiment 2: Spectral Band Selection

### Objective
This experiment aims to perform band selection on hyperspectral data, a critical step in hyperspectral image processing and analysis. The experiment leverages the HT2013 dataset, which must be downloaded and prepared before the experiment can be executed.

### Dataset Preparation
1. **Download the HT2013 Dataset:** First, ensure you have access to and download the HT2013 hyperspectral dataset. It should be placed in a directory where the experiment script can access it.

2. **Data Splitting:** The HT2013 dataset must be properly split according to the experiment's requirements. Ensure that the data is ready for processing by following the necessary splitting criteria and methods.

### How to Run
After preparing the HT2013 dataset, you can proceed with the spectral band selection experiment:

```bash
python BS_trainModel_publc_HD_EM.py
```


