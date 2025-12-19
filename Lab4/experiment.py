# =========================================================
# Experiment: Spam Email Classification using Logistic Regression
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# Step 1: Load the Dataset
# ---------------------------------------------------------

feature_names = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
    'word_freq_people', 'word_freq_report', 'word_freq_addresses',
    'word_freq_free', 'word_freq_business', 'word_freq_email',
    'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font',
    'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
    'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415',
    'word_freq_85', 'word_freq_technology', 'word_freq_1999',
    'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs',
    'word_freq_meeting', 'word_freq_original', 'word_freq_project',
    'word_freq_re', 'word_freq_edu', 'word_freq_table',
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
    'char_freq_$', 'char_freq_#',
    'capital_run_length_average', 'capital_run_length_longest',
    'capital_run_length_total',
    'spam'
]

df = pd.read_csv("spambase.data", header=None, names=feature_names)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ---------------------------------------------------------
# Step 2: Separate Features and Target
# ---------------------------------------------------------

X = df.drop('spam', axis=1)
y = df['spam']

# ---------------------------------------------------------
# Step 3: Train-Test Split
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Step 4: Feature Scaling
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# Step 5: Train Logistic Regression Model
# ---------------------------------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# Step 6: Predictions
# ---------------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ---------------------------------------------------------
# Step 7: Model Evaluation
# ---------------------------------------------------------

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy * 100, "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))

# ---------------------------------------------------------
# Step 8: Confusion Matrix Visualization
# ---------------------------------------------------------

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Spam", "Spam"],
            yticklabels=["Not Spam", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.show()
