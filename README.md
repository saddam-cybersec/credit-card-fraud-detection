# Credit Card Fraud Detection System

A machine learning pipeline built to detect fraudulent credit card transactions using a **Random Forest classifier** with customizable probability threshold tuning.

## Overview
This project addresses severe class imbalance in credit card fraud detection (only 0.17% fraud). It handles data preprocessing, feature scaling, oversampling, and trains a Random Forest model to accurately flag frauds while minimizing false positives in production.

## Key Features
- **Data Pipeline**: Custom CSV loading and manual Standard Scaling of `Time` and `Amount` features.
- **Imbalance Handling**: Custom oversampling of the minority fraud class to perfectly balance the training dataset.
- **Algorithm**: `scikit-learn` Random Forest Classifier for non-linear pattern recognition.
- **Threshold Tuning**: Configurable probability threshold (`0.80` currently) allows tuning between catching more frauds (Recall) or reducing false alarms (Precision).

## Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Download Instructions**: You must download the `creditcard.csv` file from the Kaggle link above and place it in the **root** folder of this project directory (next to `fraud_detection.py`).
- **Samples**: 284,807 transactions
- **Features**: 28 anonymized PCA features (V1–V28), Time, Amount
- **Target**: Class (0 = Legitimate, 1 = Fraudulent)
- **Imbalance**: Only 492 fraud cases (0.172%) in the original dataset.

## Installation & Usage
1. Clone this repository to your local machine.
2. Install the required library:
   ```bash
   pip install -r requirements.txt
   ```
3. Download creditcard.csv from the Kaggle link above and place it in the project folder.
4. Run the script:
```bash
python fraud_detection.py
```
## Results (with Threshold = 0.80)
Evaluated on a hold-out test set of 85,443 transactions.
``` text
Metric        |  	  Value
Accuracy      |  	  0.9994
Precision     |     0.9900
Recall        |     0.6689
F1-Score      |     0.7984
```
## Confusion Matrix
```text
TP: 99   |  FP: 1
FN: 49   |  TN: 85294
```
## Project Structure
``` text
credit-card-fraud-detection/
├── creditcard.csv        # The Kaggle dataset (must be downloaded manually)
├── fraud_detection.py    # Main Python script
├── requirements.txt      # Dependency list
└── README.md             # Project documentation
```
## Author
## Saddam Hussain
Cyber Security Intern – Arch Technologies
