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


# Methodology

## Data Cleaning


## Linear Regression Data
### Process
* The original data consist of 20641 rows and 10 rows. The data is cleaned using Microsoft Excel and 
* The **rows that are incomplete are completely removed** for more accurate reading.
* Here are the step by step process on deleting unwanted rows

    1. Open Your Excel File: Find the rows using **Find and Select** then **Go to Special**.

    ![alt text](<Image Resources/image4.png>)


    2. After selecting **Go to Special**. Select the **Blanks** to search for empty cell.

    ![alt text](<Image Resources/image6.png>)

    3. Highlight the rows you want to delete.

    ![alt text](<Image Resources/image7.png>)

* Deleted rows are rows 292 and 342
  
# Linear Regression
* Linear regression models the relationship between a dependent variable and one or more independent variables.
* The model aims to fit a line (or hyperplane for multiple variables) that minimizes the error between predicted and actual values.



## Part 1 - Data Preprocessing


### Importing libraries and the datase

Libraries:
* **tkinter** - Python’s standard **GUI (Graphical User Interface) library**.
* **pandas**  - Provides DataFrame and Series objects for handling and **analyzing data in tabular** (spreadsheet-like) form.
* **numpy** - Numerical computing library used for handling arrays and performing **mathematical operations.**
* **sklearn.model_selection.train_test_split** - Utility function in scikit-learn for splitting datasets into **training and testing sets.**
* **sklearn.linear_model.LinearRegression** - A model from scikit-learn used for **performing linear regression.**
* **sklearn.metrics.r2_score** - A function from scikit-learn to **evaluate regression model** performance.
* **matplotlib.pyplot (imported as plt)** - Used to create various **types of charts and plots** to visualize data.
* **matplotlib.backends.backend_tkagg.FigureCanvasTkAgg** - A matplotlib component that **integrates matplotlib plots into a tkinter** 


```python

import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

```

* Loads the data from housing.csv into the dataset variable, allowing the data to be manipulated and analyzed in code.

```python
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
![alt text](<Image Resources/image8.png>)


```python
# [rows,columns]
X= dataset.iloc[:,:-9].values
X

```

```python

y = dataset.iloc[:,-2].values
y

```


### Creating the Training Set and the Test Set

```python

# scikitlearn is a library
# model_selection is a module
# train_test_split is a function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2,random_state=1)

```


Defining X and Y training and testing variables.

```python

X_train
X_test
y_train
y_test


```


## Part 2 - Building and training the model
### Building the model

* from sklearn.linear_model import LinearRegression imports the LinearRegression class.

```python

# linear_model is the module
# `LinearRegression is a class` is defining that `LinearRegression` is a class within the `linear_model` module. It indicates that `LinearRegression` is a blueprint or template for creating objects that represent linear regression models.
# Class is a pre-coded blueprint of something we want to build from which objects are created.
from sklearn.linear_model import LinearRegression
model = LinearRegression()

```

### Training the Model

```python

# fit is a method inside LinearRegression class - they are like functions.
model.fit(X_train, y_train)


```

![alt text](<Image Resources/image.png>)


### Inference

```python
y_pred = model.predict(X_test)
y_pred
```

### Making the prediction of a single data point with Longitude = -122.23, Latitude = 37.84, Housing Median Age = 50, Total rooms = 2515, Total Bedrooms = 399 , Populations = 970, Households = 373, Median Income = 5.8596

#### Test Sample (Row No. 120) = -122.23,37.84,50,2515,399,970,373,5.8596
![alt text](<Image Resources/image3.png>)


#### Actual Value = $327,600



### Prediction Model

```python
model.predict([[-122.23,37.84,50,2515,399,970,373,5.8596]])
```

#### Predicted Value = $328,762.40


## Logistic Regression Data
### Process

* The original data consist of 64,375 rows and 12 rows. The data is cleaned using Microsoft Excel and 
* The **rows that are incomplete are completely removed** for more accurate reading.
* Here are the step by step process on deleting unwanted rows:

   1. Open Your Excel File: Find the rows using **Find and Select** then **Go to Special**.
      ![Screenshot 2024-10-30 110744](https://github.com/user-attachments/assets/23f20c0d-d00f-4e2e-ac74-d30f9a324146)

   3. Select the **Blanks** to search for empty cell.
      ![Screenshot 2024-10-30 111851](https://github.com/user-attachments/assets/af9e99ec-0918-4d44-8cb7-a32e6935633d)

   5. Since the result found nothing blank in the data, we can proceed for the data processing.
      ![Screenshot 2024-10-30 112323](https://github.com/user-attachments/assets/3c4607d2-667d-4678-b90b-822556043dad)

# Logistic Regression
* Logistic Regression present the learning algorithm used for binary classification.
* This works with numerical inputs, so categorical data (e.g., gender, product type) needs to be encoded. Common techniques include.

## Part-1 Data Processing

### Importing libraries and the dataset

Libraries:
* **tkinter** - Python’s standard **GUI (Graphical User Interface) library**.
* **pandas**  - Provides DataFrame and Series objects for handling and **analyzing data in tabular** (spreadsheet-like) form.
* **numpy** - Numerical computing library used for handling arrays and performing **mathematical operations.**
* **sklearn.model_selection.train_test_split** - Utility function in scikit-learn for splitting datasets into **training and testing sets.**
* **sklearn.linear_model.LinearRegression** - A model from scikit-learn used for **performing linear regression.**
* **sklearn.metrics.r2_score** - A function from scikit-learn to **evaluate regression model** performance.
* **matplotlib.pyplot (imported as plt)** - Used to create various **types of charts and plots** to visualize data.
* **matplotlib.backends.backend_tkagg.FigureCanvasTkAgg** - A matplotlib component that **integrates matplotlib plots into a tkinter**


```python

import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

```
* Imports the data from logisticff.csv into the dataset variable, enabling data manipulation and analysis in code.

```python
#variable and this is a function for uploading the dataset
dataset = pd.read_csv('logisticff.csv')

```

```python
dataset.head(10)

```
![image](https://github.com/user-attachments/assets/8ec0d376-8ac5-4249-9afa-63dcb3710033)
