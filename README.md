# Credit Card Fraud Detection System

A machine learning implementation for detecting fraudulent credit card transactions using **Logistic Regression** with **Stochastic Gradient Descent** (SGD).

## Overview

This project addresses the problem of credit card fraud detection, where fraudulent transactions represent an extremely small fraction of total transactions (0.17%). The model is trained on the Kaggle Credit Card Fraud Detection dataset and achieves strong performance using Precision, Recall, and F1-Score as evaluation metrics.

## Key Features

- **Imbalance Handling:** Uses oversampling to balance the minority fraud class in the training set.
- **Algorithm:** Logistic Regression trained via Stochastic Gradient Descent.
- **Evaluation:** Confusion Matrix, Precision, Recall, F1-Score, and Accuracy.
- **Pure Python:** No external dependencies required.

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples:** 284,807 transactions
- **Features:** 28 anonymized PCA features (V1–V28), Time, Amount
- **Target:** Class (0 = Legitimate, 1 = Fraudulent)
- **Imbalance:** Only 492 fraud cases (0.172%)

## Requirements

- Python 3.9 or higher

## Installation

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```
#Usage
1. Download the dataset from Kaggle and place creditcard.csv in the project folder.

2. Run the script:

```bash
python fraud_detection.py
```
#Results
Metric	Value
Accuracy	0.9972
Precision	0.91
Recall	0.80
F1-Score	0.85

#Confusion Matrix
```text
TP: 142  |  FP: 14
FN: 35   |  TN: 85,213
```
Project Structure
```text
credit-card-fraud-detection/
├── fraud_detection.py   # Main script
├── requirements.txt     # Dependencies (none required)
└── README.md            # Project documentation
```
#Author
##Saddam Hussain
Cyber Security Intern – Arch Technologies
