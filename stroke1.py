import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
from collections import Counter

# Load dataset
file_path = "stroke.csv"
df = pd.read_csv(file_path)

print("Dataset Description\n", df.head())
print("\nDataset Statistics summary\n", df.describe())
print("\nSkewness\n", df["avg_glucose_level"].skew())
print("\nKurtosis\n", df["avg_glucose_level"].kurtosis())

# Data Visualization
plt.figure(figsize=(8,6))
sns.histplot(df["avg_glucose_level"], bins=20, kde=True)
plt.xlabel("avg_glucose_level")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(x=df["avg_glucose_level"])
plt.title("Boxplot")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x=df["avg_glucose_level"], y=df["bmi"], palette=["red", "blue"], data=df, alpha=0.7, edgecolor="black")
plt.xlabel("avg_glucose_level")
plt.ylabel("bmi")
plt.title("ScatterPlot")
plt.show()

# Handling Missing Values
print("\nMissing values\n", df.isnull().sum())
df["bmi"].fillna(df["bmi"].mean(), inplace=True)
print("\nData after Handling missing values\n", df.isnull().sum())

# Encoding Categorical Columns
cat_col = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
encoder = LabelEncoder()
for col in cat_col:
    df[col] = encoder.fit_transform(df[col])
print("\nAfter Handling Categorical column\n", df.head())

# Feature Scaling
num_col = ["avg_glucose_level", "bmi", "age"]
scaler = MinMaxScaler()
df[num_col] = scaler.fit_transform(df[num_col])
print("\nAfter Feature Scaling(Min Max)\n", df.head())

# Correlation Matrix
corr_mat = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_mat, cmap="coolwarm", fmt=".2f", annot=True)
plt.title("Correlations of variables")
plt.show()

# Class Distribution Before Oversampling
plt.figure(figsize=(8,6))
sns.countplot(x=df["stroke"])
plt.xlabel("Stroke")
plt.ylabel("Count")
plt.title("Class Distribution of stroke")
plt.show()

# Splitting Majority and Minority Classes
df_majority = df[df["stroke"] == 0]
df_minority = df[df["stroke"] == 1]

# Oversample Minority Class
df_minority_oversampled = resample(df_minority, 
                                   replace=True, 
                                   n_samples=len(df_majority), 
                                   random_state=42)

# Combine Oversampled Minority Class with Majority Class
df_resampled = pd.concat([df_majority, df_minority_oversampled])
df_resampled = df_resampled.sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution after oversampling:", Counter(df_resampled["stroke"]))

# Class Distribution After Oversampling
plt.figure(figsize=(8,6))
sns.countplot(x=df_resampled["stroke"])
plt.xlabel("Stroke")
plt.ylabel("Count")
plt.title("Class Distribution of stroke After Oversampling")
plt.show()

# Splitting Dataset into Features and Labels
X = df_resampled.drop(columns="stroke")
y = df_resampled["stroke"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

# Evaluation Metrics
print("\nClassification Report\n", classification_report(y_test, y_pred))
print("\nAccuracy Score\n", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), cmap="coolwarm", annot=True, fmt=".2f")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'AUC = {auc(fpr, tpr):.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()
