# Machine Learning Assignments

This repository contains code, data, and reports for my machine learning coursework. The assignments explore different machine learning models and algorithms, with a focus on practical implementation, experimentation, and analysis.

## Assignment 1: Logistic Regression with Stochastic Gradient Descent

### Description

In this assignment, I developed a binary classifier to distinguish between Pop and Classical songs using logistic regression. The classifier was trained using **Stochastic Gradient Descent (SGD)** and various learning rates were tested to optimize the model. Feature scaling was applied to improve performance.

The final model achieved:
- **Training Accuracy:** 92.10%
- **Test Accuracy:** 91.85%

### Contents

- **`assignment1_torivar_hassfjord.ipynb`:** Contains data preprocessing, model training, and evaluation code.
- **`assignment1_various.ipynb`:** Additional experiments with different learning rates and decision boundary visualizations.
- **`report/`:** Final report detailing the results and analysis.

## Assignment 2: Linear Regression with Gradient Descent

### Description

This assignment focuses on implementing **linear regression** using **gradient descent** to minimize the **Mean Squared Error (MSE)** loss function. I explored different learning rates and initialization strategies, and visualized the **loss surface** to study convergence. Additionally, I analyzed the use of **Mean Absolute Error (MAE)** and its challenges in optimization.

The assignment also includes a comparison of different optimization techniques, such as **momentum**, and visualizes their impact on the learning process.

### Contents

- **`assignment2/code/assignment2_final.ipynb`:** The main notebook for implementing and analyzing linear regression with gradient descent.
- **`assignment2/data/data_problem2.csv`:** Dataset used for training and testing the model.
- **`assignment2/data/loss_surface.png`:** Visual representation of the loss surface for gradient descent optimization.
- **`assignment2/data/momentum_example.png`:** Visualization of the effect of momentum on gradient descent.
- **`assignment2/report/`:** Contains the final report (`assignment2.pdf`) and additional figures and documentation.

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
