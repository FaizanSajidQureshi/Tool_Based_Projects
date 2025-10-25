import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% Load Dataset
data = pd.read_csv(r'C:\NEUROBYTE\Data Science\Practice\PYTHON\Machine_Learning\car.csv')

# ============================
# 🛠 Data Preprocessing
# ============================

# Missing values
print("Missing Values:\n", data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Check duplicates
duplicates = data.duplicated().sum()
print("Duplicate rows:", duplicates)
if duplicates > 0:
    data.drop_duplicates(inplace=True)

# Detect Outliers using Boxplot
plt.figure(figsize=(10,6))
sns.boxplot(data[['mpg','horsepower','weight','acceleration']])
plt.title("Boxplot to Detect Outliers")
plt.show()

# One-Hot Encoding
data = pd.get_dummies(data, columns=['origin'], drop_first=True)

# ============================
# 📊 Exploratory Data Analysis
# ============================

# Distribution of target variable
plt.figure(figsize=(8,6))
sns.histplot(data['mpg'], bins=20, kde=True)
plt.title("Distribution of MPG")
plt.xlabel("mpg")
plt.ylabel("Frequency")
plt.show()

# Countplot for cylinders
plt.figure(figsize=(8,6))
sns.countplot(x='cylinders', data=data)
plt.title("Count of Cars by Cylinders")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ============================
# 🔀 Train Test Split
# ============================
X = data[['cylinders', 'displacement', 'horsepower', 'weight',
          'acceleration', 'model_year', 'origin_japan', 'origin_usa']]
y = data['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# 🤖 Model Training
# ============================
model = LinearRegression()
model.fit(X_train, y_train)

# ============================
# 📈 Model Evaluation
# ============================
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-Squared (R²) Score: {r2:.2f}')

# ============================
# 📊 Visualization of Results
# ============================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='red', alpha=0.7)
plt.xlabel('Actual mpg')
plt.ylabel('Predicted mpg')
plt.title('Actual vs Predicted mpg of Cars')
plt.grid(True)
plt.show()
