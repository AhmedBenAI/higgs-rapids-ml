# Accelerated Machine Learning for Higgs Boson Classification Using RAPIDS

This project builds a full machine learning pipeline for classifying Higgs boson events using the HIGGS dataset from the UCI Machine Learning Repository.
It demonstrates how GPU acceleration with NVIDIA RAPIDS and Dask improves preprocessing and training performance compared to a CPU based workflow.

The repository contains the exact code and methodology used in the final coursework notebook `notebooks/AML.ipynb`.

## 1. Project Goals

- Build a scalable machine learning workflow for Higgs boson signal vs background classification
- Perform exploratory data analysis and preprocessing for a large, high dimensional dataset
- Train and compare several models: Random Forest, XGBoost, and Support Vector Machine
- Apply PCA for dimensionality reduction and analyze its effect on performance
- Compare CPU (pandas plus scikit learn) and GPU (cuDF plus cuML) versions of the pipeline

## 2. Dataset

- Name: HIGGS dataset
- Source: UCI Machine Learning Repository
- Samples: 11 million events
- Features: 28 numerical features
  - 21 low level detector features
  - 7 high level handcrafted features
- Target:
  - 1: signal (Higgs boson production)
  - 0: background

In practice most experiments in the notebook use a subset of the full 11 million rows in order to keep experiments manageable while still large enough to show the benefit of GPU acceleration.

## 3. Preprocessing Pipeline

The preprocessing steps implemented in the notebook include:

- Skewness correction using log transforms on heavily skewed features
- Outlier handling using a combination of transformation and Z score filtering
- Feature scaling using Z score normalization
- Dimensionality reduction using Principal Component Analysis (PCA)
- Train, validation, test splitting with an 80 percent, 10 percent, 10 percent split

These steps address high dimensionality, skewed distributions, and outliers which are typical for Higgs style datasets.

## 4. Models

The following models are trained and evaluated:

- Random Forest
- Support Vector Machine with RBF kernel
- XGBoost

Hyperparameters are tuned using validation performance, taking into account the cost of training on millions of examples.

Both CPU and GPU variants are used where possible, so that the speedup from RAPIDS can be measured directly.

## 5. GPU Acceleration with RAPIDS

The project uses the RAPIDS ecosystem to move the tabular pipeline from CPU to GPU:

- cuDF for GPU dataframes
- cuML for GPU accelerated machine learning
- Dask to support larger than memory datasets and parallel computation

The notebook compares:

- Data loading and preprocessing time on CPU vs GPU
- Model training time on CPU vs GPU
- Model performance metrics on both versions

## 6. Evaluation Metrics

The following metrics are used to evaluate the classifiers:

- ROC AUC
- Precision
- Recall
- F1 score

These metrics provide a clear view of class separation and performance on both signal and background classes.

## 7. Repository Structure

```text
higgs-rapids-project/
├── notebooks/
│   ├── AML.ipynb        # Original coursework notebook with full code and results
│   └── AML_clean.ipynb  # Same notebook with outputs cleared for a lighter version
├── requirements.txt
├── .gitignore
└── README.md
```

All real code and markdown for the project lives inside `notebooks/AML.ipynb`.  
The cleaned version `AML_clean.ipynb` is useful if you want a smaller file for GitHub while keeping the original locally.

## 8. Environment Setup

1. Create and activate an environment (example with conda):

   ```bash
   conda create -n higgs-rapids python=3.10
   conda activate higgs-rapids
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install RAPIDS for your CUDA version following the official instructions on the RAPIDS web site.

## 9. How to Run

The entire workflow is contained in the notebook.

```text
notebooks/AML.ipynb
```

Open it in Jupyter or VS Code and execute the cells from top to bottom.
For a lighter version without stored outputs you can use `AML_clean.ipynb`.

## 10. Notes

- This repository is intended as a portfolio quality example of GPU accelerated machine learning on a realistic high energy physics dataset.
- The notebook includes end to end steps: exploratory data analysis, preprocessing, model training, evaluation, and CPU vs GPU comparison.
