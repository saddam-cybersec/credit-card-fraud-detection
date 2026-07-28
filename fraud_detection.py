"""
Credit Card Fraud Detection System
-----------------------------------
Implementation using Random Forest with a strict probability threshold
to detect fraudulent credit card transactions.

Author: Saddam Hussain
Project: Cyber Security Internship - Month 2
"""

import csv
import math
import random

# ------------------------------------------------------------
# 1. MATH UTILITIES (Kept for learning demonstration)
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
# 6. MAIN EXECUTION (MODEL REPLACED WITH RANDOM FOREST)
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
    
    # Step 5: Train Random Forest Classifier
    print("\n[5] Training Random Forest Classifier...")
    from sklearn.ensemble import RandomForestClassifier
    # n_estimators=100 is standard, n_jobs=-1 uses all CPU cores for speed
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_bal, y_train_bal)
    print("[INFO] Random Forest training complete.")
    
    # Step 6: Evaluate with a high probability threshold (to eliminate False Positives)
    print("\n[6] Evaluating model on test set...")
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    # Get the probability scores, not the 0/1 predictions
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    
    # Apply a custom threshold (0.80 means 80% confidence required to flag as fraud)
    threshold = 0.80
    y_pred = (y_probs >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Confusion Matrix:")
    print(f"  TP: {cm[1,1]}  |  FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}  |  TN: {cm[0,0]}")
    
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 60)
    
    print("\n[INFO] Note: You can tweak the 'threshold' variable in the code to balance Precision vs Recall.")

if __name__ == "__main__":
    main()
