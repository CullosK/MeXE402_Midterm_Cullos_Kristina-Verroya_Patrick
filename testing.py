import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
dataset = pd.read_csv('housing.csv') 

# Prepare the data
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the R^2 score
r2 = r2_score(y_test, y_pred)

# Calculate the adjusted R^2 score
k = X_test.shape[1]
n = X_test.shape[0]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# Function to calculate R^2 without a specific feature
def calculate_r2_without_feature(X, y, feature_index):
    X_reduced = np.delete(X, feature_index, axis=1)  # Remove the feature
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.2, random_state=1)
    model_reduced = LinearRegression()
    model_reduced.fit(X_train_reduced, y_train_reduced)
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    return r2_score(y_test_reduced, y_pred_reduced)

# Calculate R^2 scores for each feature
feature_importance = {}
for i in range(X.shape[1]):
    r2_without_feature = calculate_r2_without_feature(X, y, i)
    importance = r2 - r2_without_feature
    feature_importance[f'Feature {i}'] = importance

# Display feature importances
for feature, importance in feature_importance.items():
    print(f'{feature}: {importance:.4f}')

# Optional: Display the overall model R^2 and adjusted R^2
print(f'Overall R^2: {r2:.4f}')
print(f'Adjusted R^2: {adj_r2:.4f}')
