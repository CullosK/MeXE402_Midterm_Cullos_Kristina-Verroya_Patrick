import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv('housing.csv')
dataset.head()  # Display 5 rows of dataset

# Split dataset into features (X) and target (y)
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit the linear regression model on the standardized data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict using the test set
y_pred = model.predict(X_test_scaled)

# Calculate R-squared and adjusted R-squared
r2 = r2_score(y_test, y_pred)
k = X_test.shape[1]
n = X_test.shape[0]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)

# Calculate variable importance using standardized coefficients
coefficients = model.coef_
importance = pd.Series(coefficients, index=dataset.columns[:-2]).sort_values(ascending=False)
print("Variable importance based on standardized coefficients:\n", importance)
