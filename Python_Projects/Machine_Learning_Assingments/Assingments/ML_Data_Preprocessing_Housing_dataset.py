import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
h_data = pd.read_csv(r'C:\NEUROBYTE\Data Science\Practice\PYTHON\Machine_Learning\housing_dataset.csv')
print(h_data.head())
print(h_data.info())
print(h_data.describe())
print(h_data.isnull().sum())
data_cleaned = h_data.dropna()
categorical_cols = data_cleaned.select_dtypes(include=['object']).columns
numerical_cols = data_cleaned.select_dtypes(exclude=['object']).columns
print("Categorical:", categorical_cols.tolist())
print("Numericals:", numerical_cols.tolist())
data_encoded = pd.get_dummies(h_data, columns=categorical_cols, drop_first=True)
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    return outliers
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]
for col in numerical_cols:
    data_encoded = remove_outliers_iqr(data_cleaned, col)
scaler = StandardScaler()
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])
for col in numerical_cols:
    plt.figure(figsize=(6, 3))
    plt.hist(h_data[col], bins=30, edgecolor='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
X = data_encoded.drop(['Price'], axis=1)
y = data_encoded['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Shape:", X_train.shape)
print("Test Shape:", X_test.shape)