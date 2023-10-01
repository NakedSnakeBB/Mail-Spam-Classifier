# Spam Mail Detection Model

Author: Soumedhik Bharati

## Introduction
This project focuses on building a Spam Mail Detection model using Python. The model aims to classify emails into spam and non-spam categories. In this README, we will provide a step-by-step explanation of the code and its functionality.

## Dataset
The dataset is loaded from the CSV file 'emailsmol.csv'.

## Data Preprocessing
- **Missing Value Handling:** The code counts the number of missing values in each column of the dataset.
- **Feature Selection:** Selects the first 3000 columns as input features (X).
- **Target Variable:** Selects the 'Prediction' column as the target variable (Y).

## Model Building
- **Splitting Data:** Splits the dataset into training and testing sets (80% train, 20% test).
- **Model Selection:** Initializes a Gaussian Naive Bayes model and trains it on the training data.
- **Prediction:** Makes predictions on the test data.

## Evaluation
- **Weighted F1 Score:** Calculates the weighted F1 score of the model's predictions.
- **Accuracy Score:** Calculates the accuracy score of the model's predictions.
- **Confusion Matrix:** Generates a confusion matrix and visualizes it using a heatmap.

## Results
- **Weighted F1 Score:** 0.9485438246750387
- **Accuracy Score:** 0.9478260869565217

## Conclusion
This Spam Mail Detection Model can effectively classify emails into spam and non-spam categories. It can be used for various applications to filter out unwanted emails.

For more details, refer to the code file provided.

## Dependencies
Make sure you have the following libraries installed before running the code:
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

```bash
pip install numpy pandas scikit-learn matplotlib seaborn


