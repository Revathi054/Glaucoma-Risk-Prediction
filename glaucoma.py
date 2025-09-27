# glaucoma_model.py
# Glaucoma - Damage in Optic Nerve (Classification using ML)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ------------------ Load Data ------------------
df = pd.read_csv("glaucoma.csv")

# Fill missing values
mode_value = df['Medication Usage'].mode()[0]
df['Medication Usage'] = df['Medication Usage'].fillna(mode_value)

# Drop unused columns
df = df.drop(columns=[
    'Patient ID', 'Medical History', 'Medication Usage',
    'Visual Field Test Results', 'Optical Coherence Tomography (OCT) Results',
    'Visual Symptoms'
])

# ------------------ Encoding ------------------
# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=[
    'Gender', 'Visual Acuity Measurements', 'Family History',
    'Cataract Status', 'Angle Closure Status', 'Diagnosis'
])

# Encode target column
label_encoder = LabelEncoder()
df_encoded['Glaucoma Type'] = label_encoder.fit_transform(df_encoded['Glaucoma Type'])

# Scale numerical features
scaler = MinMaxScaler()
df_encoded[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']] = \
    scaler.fit_transform(df_encoded[['Age', 'Intraocular Pressure (IOP)', 'Cup-to-Disc Ratio (CDR)', 'Pachymetry']])

# ------------------ Train-Test Split ------------------
X = df_encoded.drop(['Glaucoma Type'], axis=1)
y = df_encoded['Glaucoma Type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Before Oversampling:", y_train.value_counts())

# ------------------ Oversampling with SMOTE ------------------
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After Oversampling:\n", y_train_resampled.value_counts())

# ------------------ Logistic Regression ------------------
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_resampled, y_train_resampled)
y_pred = logreg.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ------------------ SVM Classifier ------------------
svc = svm.SVC()
svc.fit(X_train_resampled, y_train_resampled)
y_pred = svc.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_estimator(svc, X_test, y_test)
plt.title("SVM Confusion Matrix")
plt.show()


