"""
Credit Card Fraud Detection System
-----------------------------------
Implementation of Logistic Regression with Stochastic Gradient Descent
to detect fraudulent credit card transactions.

Author: Saddam Hussain
Project: Cyber Security Internship - Month 2
"""

import csv
import math
import random

# ------------------------------------------------------------
# 1. MATH UTILITIES
# ------------------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)

def predict_probability(features, weights):
    """Calculate probability P(y=1) using weights and bias."""
    z = weights[0]  # bias term
    for i in range(len(features)):
        z += weights[i + 1] * features[i]
    return sigmoid(z)

def predict_class(features, weights, threshold=0.5):
    """Predict class (0 or 1) based on probability threshold."""
    return 1 if predict_probability(features, weights) >= threshold else 0

# ------------------------------------------------------------
# 2. DATA LOADING
# ------------------------------------------------------------

def load_data(filename):
    """
    Loads creditcard.csv.
    Returns: list of features (30 floats) and list of labels (int).
    """
    features_list = []
    labels_list = []
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        
        for row in reader:
            vals = [float(x) for x in row]
            features = vals[0:30]  # V1..V28, Time, Amount
            label = int(vals[30])   # Class
            features_list.append(features)
            labels_list.append(label)
    
    return features_list, labels_list

# ------------------------------------------------------------
# 3. STANDARD SCALER
# ------------------------------------------------------------

def standard_scale(features_list):
    """
    Scales Time (index 28) and Amount (index 29) to mean=0, std=1.
    """
    n = len(features_list)
    
    sum_time = sum(row[28] for row in features_list)
    sum_amount = sum(row[29] for row in features_list)
    
    mean_time = sum_time / n
    mean_amount = sum_amount / n
    
    var_time = sum((row[28] - mean_time) ** 2 for row in features_list) / n
    var_amount = sum((row[29] - mean_amount) ** 2 for row in features_list) / n
    
    std_time = math.sqrt(var_time) if math.sqrt(var_time) != 0 else 1.0
    std_amount = math.sqrt(var_amount) if math.sqrt(var_amount) != 0 else 1.0
    
    scaled = []
    for row in features_list:
        new_row = row[:]
        new_row[28] = (row[28] - mean_time) / std_time
        new_row[29] = (row[29] - mean_amount) / std_amount
        scaled.append(new_row)
    
    return scaled

# ------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ------------------------------------------------------------

def train_test_split_manual(features, labels, test_ratio=0.3, seed=42):
    """Shuffle and split data into train (70%) and test (30%)."""
    random.seed(seed)
    combined = list(zip(features, labels))
    random.shuffle(combined)
    
    split_idx = int(len(combined) * (1 - test_ratio))
    train = combined[:split_idx]
    test = combined[split_idx:]
    
    X_train = [item[0] for item in train]
    y_train = [item[1] for item in train]
    X_test = [item[0] for item in test]
    y_test = [item[1] for item in test]
    
    return X_train, y_train, X_test, y_test

# ------------------------------------------------------------
# 5. OVERSAMPLING (Handle Imbalance)
# ------------------------------------------------------------

def oversample_minority(X_train, y_train):
    """
    Duplicates fraud cases (class=1) to balance the training set.
    """
    X_fraud = [X_train[i] for i in range(len(y_train)) if y_train[i] == 1]
    y_fraud = [1] * len(X_fraud)
    
    X_legit = [X_train[i] for i in range(len(y_train)) if y_train[i] == 0]
    y_legit = [0] * len(X_legit)
    
    # Duplicate frauds until we have roughly the same count as legit
    multiplier = max(1, len(X_legit) // len(X_fraud))
    X_fraud_oversampled = []
    y_fraud_oversampled = []
    for _ in range(multiplier):
        X_fraud_oversampled.extend(X_fraud)
        y_fraud_oversampled.extend(y_fraud)
    
    # Combine and shuffle
    combined = list(zip(X_legit + X_fraud_oversampled, y_legit + y_fraud_oversampled))
    random.shuffle(combined)
    
    X_new = [item[0] for item in combined]
    y_new = [item[1] for item in combined]
    
    print(f"[INFO] Oversampled training size: {len(X_new)} (Fraud: {sum(y_new)})")
    return X_new, y_new

# ------------------------------------------------------------
# 6. LOGISTIC REGRESSION TRAINER (SGD)
# ------------------------------------------------------------

def train_logistic_regression(X_train, y_train, learning_rate=0.01, epochs=10):
    """
    Trains logistic regression using Stochastic Gradient Descent.
    weights[0] = bias, weights[1:] = coefficients for 30 features.
    """
    num_features = len(X_train[0])
    weights = [random.uniform(-0.01, 0.01) for _ in range(num_features + 1)]
    
    n = len(X_train)
    
    print(f"[INFO] Training on {n} samples, {num_features} features...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        combined = list(zip(X_train, y_train))
        random.shuffle(combined)
        
        for features, label in combined:
            prob = predict_probability(features, weights)
            
            # Avoid log(0)
            if prob == 1.0:
                prob = 0.9999
            elif prob == 0.0:
                prob = 0.0001
            
            loss = - (label * math.log(prob) + (1 - label) * math.log(1 - prob))
            total_loss += loss
            
            error = prob - label
            
            # Update bias
            weights[0] -= learning_rate * error
            
            # Update feature weights
            for i in range(num_features):
                weights[i + 1] -= learning_rate * error * features[i]
        
        avg_loss = total_loss / n
        print(f"  Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")
    
    return weights

# ------------------------------------------------------------
# 7. EVALUATION METRICS
# ------------------------------------------------------------

def evaluate_model(X_test, y_test, weights, threshold=0.5):
    """Calculate Confusion Matrix, Precision, Recall, F1, Accuracy."""
    tp = tn = fp = fn = 0
    
    for features, true_label in zip(X_test, y_test):
        pred = predict_class(features, weights, threshold)
        
        if pred == 1 and true_label == 1:
            tp += 1
        elif pred == 0 and true_label == 0:
            tn += 1
        elif pred == 1 and true_label == 0:
            fp += 1
        elif pred == 0 and true_label == 1:
            fn += 1
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# ------------------------------------------------------------
# 8. MAIN EXECUTION
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1] Loading creditcard.csv ...")
    features, labels = load_data('creditcard.csv')
    print(f"    Total samples: {len(features)}")
    fraud_count = sum(labels)
    legit_count = len(labels) - fraud_count
    print(f"    Legit: {legit_count}, Fraud: {fraud_count} ({fraud_count/len(labels)*100:.2f}%)")
    
    # Step 2: Scale Time and Amount
    print("\n[2] Scaling Time and Amount...")
    features_scaled = standard_scale(features)
    
    # Step 3: Train/Test split
    print("\n[3] Splitting data (70% train, 30% test)...")
    X_train, y_train, X_test, y_test = train_test_split_manual(
        features_scaled, labels, test_ratio=0.3, seed=42
    )
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Step 4: Oversample fraud in training
    print("\n[4] Handling class imbalance (oversampling fraud)...")
    X_train_bal, y_train_bal = oversample_minority(X_train, y_train)
    
    # Step 5: Train model
    print("\n[5] Training Logistic Regression (SGD)...")
    weights = train_logistic_regression(X_train_bal, y_train_bal, 
                                        learning_rate=0.01, epochs=10)
    
    # Step 6: Evaluate
    print("\n[6] Evaluating model on test set...")
    results = evaluate_model(X_test, y_test, weights, threshold=0.5)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Confusion Matrix:")
    print(f"  TP: {results['TP']}  |  FP: {results['FP']}")
    print(f"  FN: {results['FN']}  |  TN: {results['TN']}")
    print(f"\nAccuracy:  {results['Accuracy']:.4f}")
    print(f"Precision: {results['Precision']:.4f}")
    print(f"Recall:    {results['Recall']:.4f}")
    print(f"F1-Score:  {results['F1-Score']:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()