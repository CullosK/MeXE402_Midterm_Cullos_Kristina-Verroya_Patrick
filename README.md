# MeXE402_Midterm_Cullos_Kristina-Verroya_Patrick
## Introduction: Overview of Linear and Logistic Regression.
### Linear Regression
• Purpose: Linear regression is a supervised learning algorithm used for predicting continuous numerical outcomes. Its goal is to model the relationship between one or more independent variables (features) and a dependent variable (target) by fitting a linear equation to observed data.

• Types: Simple Linear Regression that uses one independent variable and Multiple Linear Regression that uses more than one independent variable.

• Goal: To minimize the sum of squared residuals (differences between observed and predicted values) to find the best-fitting line, typically using methods like Ordinary Least Squares (OLS).

• Applications: Linear regression is used to predict housing prices based on features like size, location, and amenities and also to estimate stock prices based on market variables.

• Assumptions: Linearity between independent and dependent variables, independence of errors, homoscedasticity (constant variance of errors), and normality of residuals.

### Logistic Regression
• Purpose: Logistic regression is a supervised learning algorithm used for binary classification (e.g., 0 or 1, yes or no, true or false). Unlike linear regression, it predicts the probability that a given input point belongs to a particular class.

• Decision Boundary: By default, probabilities greater than 0.5 are classified as 1, and those below 0.5 as 0.

• Types: Binary Logistic Regression that predicts between two classes (0 or 1), Multinomial Logistic Regression that extend for more than two classes, and Ordinal Logistic Regression that predicts ordered outcomes.

• Goal: It's goal is to find the best parameters to maximize the likelihood of correctly classifying observations, typically using methods like Maximum Likelihood Estimation (MLE).

• Applications: Predicting whether a customer will purchase a product (yes/no) or for medical diagnosis (e.g., disease present or not) is the application of logistic regression.

• Assumptions: Independence of observations, linearity of independent variables and log-odds, and large sample size (for reliable results).

## Dataset Description

## Project Objectives


  
# Linear Regression
## Part 1 - Data Preprocessing
### Importing libraries and the dataset



```python

import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#variable and this is a function for uploading the dataset
dataset = pd.read_csv('housing.csv') 

```


```python

dataset.head() #display 5 rows of dataset

#10,000 rows
#data points collected from a combined cycle power plant over six years
#5 columns: AT ambient temp,V exhaust vacuum, AP ambient pressure, RH relative humdity, PE net hourly  electrical energy output
# independent variables: AT, V, AP and RH
# dependent variable: PE

```