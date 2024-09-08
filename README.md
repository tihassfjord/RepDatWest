# Machine Learning Assignments

This repository contains code and data for my machine learning coursework. Currently, it includes the first assignment focused on building a logistic regression classifier using Stochastic Gradient Descent (SGD) and evaluating its performance.

## Assignment 1: Logistic Regression with Stochastic Gradient Descent

### Description

In this assignment, I developed a binary classifier to distinguish between Pop and Classical songs using logistic regression. The dataset was preprocessed by scaling key features like `liveness` and `loudness`, which were selected as the primary predictors. The model was trained using Stochastic Gradient Descent (SGD), and I experimented with various learning rates to optimize the model's performance.

The final model achieved a training accuracy of 92.10% and a test accuracy of 91.85%, demonstrating that feature scaling had a significant impact on improving model performance.

### Repository Structure
assignment1/
│
├── code/
│   ├── assignment1.ipynb            # Jupyter notebook with full code and analysis
│   ├── assignment1_2.ipynb          # Additional notebook for model experiments
│
├── data/
│   ├── SpotifyFeatures.csv          # Filtered dataset used for model training
│
├── report/
│   ├── assignment1_torivar_hassfjord.pdf # Final report for assignment 1
│   ├── FYS_2021_Assignment1.pdf          # Assignment description
│
├── .gitignore
├── README.md                         # Overview of the project
├── requirements.txt                  # Required dependencies


### Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt


