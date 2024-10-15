# Optimizing Crop Type Mapping for Fairness - Code Repository

This repository contains the code used for the MSc thesis titled *Optimizing Crop Type Mapping for Fairness*, which explores methods to improve crop type classification while addressing class imbalance and the disparity between small and large agricultural parcels using the BreizhCrops dataset. The full thesis can be accessed at .

## Overview

To replicate the results presented in the thesis, this repository should be combined with the full BreizhCrops dataset. This repository only provides supplementary code and other content specifically relating to the thesis.

## Repository Contents

- **`breizhcrops.py`**: A modified version of the dataset class, extended with functionalities to handle undersampled datasets using index files.
- **`index_files_without_thin_parcels`**: Index files from the full dataset, with thin parcels excluded to reduce noise in experiments.
- **`AEs/`**: Contains the auxiliary experiments; further details can be found in the README within this folder.
- **`dataset_generation.ipynb`**: A notebook used to generate candidate datasets and calculate their complexity values.
- **`RO.ipynb`**: A script implementing random oversampling (RO) to balance datasets.
- **`RO-R.ipynb`**: A novel script for random oversampling with resampling (RO-R), which increases the number of small parcels through a more targeted oversampling approach.
- **`summary_baseline_training_datasets.csv`**: A summary file of the experimental datasets (ED 1-ED 10).
- **`train.py`**: The main script for running experiments. It takes as input a folder containing dataset index files, along with parameters specifying the method to use and an output folder for saving results. This script is based on the BreizhCrops example training script.
- **`get_results.py`**: A utility script that selects the best validation model and tests it on the test set, outputting the final performance results.
- **`analysis_functions.py`**: A collection of functions to compile results, including calculating fairness metrics, drawing confusion matrixes, and other results.
- **`results/`**: Contains the main results from the research. The `results_per_dataset-model/` subfolder holds method-model results, including train logs, final model states, and summaries of validation and testing results. Due to storage constraints, it does not include plots or CSVs of sample-wise results. Note that dataset ID `0` refers to *AD-5* in the thesis, included as it was initially planned to explore all methods on the full dataset (though computational and time limitations restricted all methods from being tested on this large dataset). This folder also includes a notebook that aggregates results across datasets to create the plots presented in the thesis.

For more information please contact author at ilyagorb99@gmail.com
