import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load and explore the dataset
file_path = "IRIS.csv"
df = pd.read_csv(file_path)

print("Dataset Description")
print(df.head())

print("Dataset summary statistics")
print(df.describe())

num_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

for col in num_cols:
    print(f"\nStatistics for {col}:")
    print("Median:", df[col].median())
    print("Mode:", df[col].mode().iloc[0])
    print("Standard Deviation (SD):", np.std(df[col], ddof=1))
    print("Interquartile Range (IQR):", df[col].quantile(0.75) - df[col].quantile(0.25))
    print("Quartile Deviation:", (df[col].quantile(0.75) - df[col].quantile(0.25)) / 2)
    print("Coefficient of Variation (CV):", (df[col].std() / df[col].mean()) * 100)
    print("Mean Absolute Deviation (MAD):", (df[col] - df[col].mean()).abs().mean())
    print("Median Absolute Deviation:", median_abs_deviation(df[col]))
    print("Skewness:", df[col].skew())
    print("Kurtosis:", df[col].kurtosis())

print("Dataset Visualization")

print("1.Histogram")
plt.figure(figsize=(8,6))
sns.histplot(df['sepal_length'], bins=20, kde=True)
plt.title("Histogram of Sepal Length")
plt.xlabel("sepal_length")
plt.ylabel("frequency")
plt.show()

print("2.BoxPlot")
plt.figure(figsize=(8,6))
sns.boxplot(x=df["sepal_length"])
plt.title("Boxplot of Sepal Length")
plt.show()

print("3.Scatter Plot")
plt.figure(figsize=(8,6))
sns.scatterplot(x="sepal_length", y="sepal_width", data=df, alpha=0.7, edgecolor="black")
plt.title("Scatter plot")
plt.xlabel("sepal_length")
plt.ylabel("sepal_width")
plt.show()

print("Missing values in dataset")
print(df.isnull().sum())

print("Handling missing values")
df["sepal_length"].fillna(df["sepal_length"].mean(), inplace=True)

print("Label Encoding Species class")
encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])
print("Encoded classes : ", df["species"].unique())

print("Feature Scaling - MinMaxScaler")
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("Data after Feature scaling ", df.head())

X = df.drop(columns=["species"])
y = df["species"]

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["species"] = y_resampled

print("Class distribution after SMOTE:", Counter(y_resampled))
print(df_resampled.head())

sns.countplot(x=df_resampled["species"])
plt.title("Class Distribution After SMOTE Oversampling")
plt.show()

df_resampled.to_csv("IRIS_SMOTE.csv", index=False)
print("Resampled dataset saved as IRIS_SMOTE.csv")

print("Correlation matrix")
print(df.corr())

print("Covariance matrix")
print(df.cov())

print("4.Correlation Heatmap")
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("5.Covariance Heatmap")
plt.figure(figsize=(8,6))
sns.heatmap(df.cov(), cmap="coolwarm", annot=True, fmt=".2f")
plt.title("Covariance Heatmap")
plt.show()

df = pd.read_csv("IRIS_SMOTE.csv")

# Split features and target
X = df.drop(columns=["species"])
y = df["species"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report and Accuracy
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
