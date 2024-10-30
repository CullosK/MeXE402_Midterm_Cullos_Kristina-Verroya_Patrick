import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv('housing.csv')

# Display the first 5 rows of the dataset
print(dataset.head())

# [rows, columns]
X = dataset.iloc[:, :-2].values
y = dataset.iloc[:, -2].values

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the test set results
y_pred = model.predict(X_test)

# Extract median_income from the dataset
median_income = dataset.iloc[:, -1].values  # Assuming median_income is the last column

# Create a DataFrame for y_pred and median_income
pred_income_df = pd.DataFrame({'y_pred': y_pred, 'median_income': median_income})

# Fit a new linear regression model between y_pred and median_income
income_model = LinearRegression()
income_model.fit(pred_income_df[['median_income']], pred_income_df['y_pred'])

# Predicting y_pred based on median_income
y_income_pred = income_model.predict(pred_income_df[['median_income']])

# Calculate R^2 score for the new model
income_r2 = r2_score(pred_income_df['y_pred'], y_income_pred)

# Print R^2 score for the new model
print(f'R^2 score for the linear regression model between y_pred and median_income: {income_r2}')

# Optional: Visualize the results
plt.scatter(pred_income_df['median_income'], pred_income_df['y_pred'], color='blue', label='Predicted Values')
plt.plot(pred_income_df['median_income'], y_income_pred, color='red', label='Regression Line')
plt.xlabel('Median Income')
plt.ylabel('Predicted Housing Prices')
plt.title('Linear Regression between Predicted Values and Median Income')
plt.legend()
plt.show()
