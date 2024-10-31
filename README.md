
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

### k 
## Project Objectives


# Methodology

## Data Cleaning


## Linear Regression Data
### Process
* The data sheets contains the house pricinh
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
X= dataset.iloc[:,:-2].values
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

```python
dataset.info

```
![image](https://github.com/user-attachments/assets/4ddc834e-5f96-4a39-8b83-d56b0415a67e)


### Getting the Inputs and the Outputs

```python
X1 = dataset.iloc[:,1:2].values
X2 = dataset.iloc[:,3:7].values
X3 = dataset.iloc[:,9:11].values
X = np.concatenate((X1,X2,X3),1)
y = dataset.iloc[:,-1].values

```

```python
X

```
![image](https://github.com/user-attachments/assets/75375dce-3d7b-43d7-996a-941223571b33)


```python
y

```
![image](https://github.com/user-attachments/assets/fe43aa65-8a95-4bbd-9bc3-2eedef023b0d)


### Creating the Training Set and the Test Set

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

```

```python
X_train
X_test
y_train
y_test

```

* Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

```

```python

X_train

```


## Part-2 Building and training the model

### Building the Model

* from sklearn.linear_model import LogisticRegression imports LogisticRegression class.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)

```

### Training the Model

```python
model.fit(X_train, y_train)

```

### Inference

* Making the predictons of the data points in the test set.
  
```python
y_pred = model.predict(sc.transform(X_test))
y_pred

```

## Part-3 Evaluating the model

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

```

### Accuracy

```python
(5503+4967)/(5503+4967+1154+1251)

```
![image](https://github.com/user-attachments/assets/38b89735-abe9-4a99-bd48-83862f74ead6)


```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

```

```python
numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

```

![image](https://github.com/user-attachments/assets/8fac340a-078a-46a1-9747-9156bdbdaf5e)





# Summary

# Linear Regression

## Variable Imporatance Analysis Based on R-squared

* Explanation 
![alt text](<Image Resources/Linear Regresion Image Resources/Variable Importance Base on R2.png>)

## Linear Regression Model Graph of Training Variables

### Longitude

![alt text](<Image Resources/Linear Regresion Image Resources/Longitude.png>)

### Latitude

![alt text](<Image Resources/Linear Regresion Image Resources/Latitude.png>)


### Housing Median Age

![alt text](<Image Resources/Linear Regresion Image Resources/Housing Median Age.png>)

### Total Rooms

![alt text](<Image Resources/Linear Regresion Image Resources/Total Rooms.png>)

### Total Bedrooms

![alt text](<Image Resources/Linear Regresion Image Resources/Total Bedrooms.png>)

### Populations

![alt text](<Image Resources/Linear Regresion Image Resources/Population.png>)

### Households

![alt text](<Image Resources/Linear Regresion Image Resources/Households.png>)

### Median Income

![alt text](<Image Resources/Linear Regresion Image Resources/Median Incom.png>)




## 

# Logistic Regression

## Variable Importance Analysis base on Accuracy

* "Variable Importance Analysis based on Accuracy" is a method used in machine learning and statistical modeling to evaluate the significance of each input variable in predicting an outcome. Here’s a breakdown of how it works:

   1. Variable Importance: This refers to the contribution of each predictor (or feature) in a model. It shows how much each variable contributes to the model’s accuracy, helping to identify the most influential variables.

   2. Based on Accuracy: This approach measures the importance by assessing how each variable affects the overall model accuracy. Generally, this is done by:

   3. Training the model with all features to get a baseline accuracy.
       * Then, each variable is "shuffled" (i.e., its values are permuted randomly or temporarily removed).
       * The model's accuracy is recalculated without that variable, and the drop in accuracy shows the variable's importance: a larger drop indicates a higher importance for that feature.
       * Interpretation: Variables with the greatest accuracy drop are the most influential, as the model depends heavily on them to make predictions. This method can be particularly useful in tree-based models (e.g., random forests), but can also apply to other machine learning models.

The process provides insights into which variables are driving the model’s performance, which is especially useful for feature selection, model simplification, and interpretation of results.


### Age

* An accuracy of 57.48% indicates that while age and tenure have some relevance, they are insufficient on their own for strong predictive performance.

![alt text](<Image Resources/Logistic Image Resources/Age Con. Matrix.png>)


### Tenure

*  The accuracy of 57.48% indicates that while age and tenure have some predictive value, they are insufficient on their own for strong predictive performance.
*  The model might benefit from exploring additional variables to improve accuracy.

![alt text](<Image Resources/Logistic Image Resources/Tenure Con.png>)


### Usage Frequency

* An accuracy of 54.71% shows that Usage Frequency contributes minimally and is insufficient on its own for strong predictive performance.
* Additional features would likely improve the model’s accuracy.

![alt text](<Image Resources/Logistic Image Resources/Usage Freq. Con.png>)


### Support Calls

* An accuracy of 64.47% indicates that Support Calls has moderate predictive value, suggesting some correlation with the target outcome.
* While useful as a baseline, including additional features would likely improve the model's accuracy and reliability.

![alt text](<Image Resources/Logistic Image Resources/Support Calls Con.png>)


### Payment Delays

* An accuracy of 76.93% indicates that Payment Delays is a strong predictor of the outcome, providing reliable insights on its own.
* However, adding other relevant features could help refine the model for even greater accuracy and nuanced predictions.

![alt text](<Image Resources/Logistic Image Resources/payment delays con.png>)


### Total Spend

* An accuracy of 54.27% suggests that while Total Spend has a slight correlation with the target outcome.
* It is insufficient on its own for strong predictive power.
* Additional features would likely be necessary for a more accurate and robust model.

![alt text](<Image Resources/Logistic Image Resources/total spent con.png>)


### Last Interaction

* An accuracy of 52.46% indicates that Last Interaction has very little predictive power by itself.
* And including additional features would likely be necessary to achieve a model with practical predictive accuracy.

![alt text](<Image Resources/Logistic Image Resources/Last  Interaction Con.png>)


# Conclusion

* In summary, each variable individually contributes to predicting the outcome, but with varying levels of accuracy and predictive power:

- **Age and Tenure**: Achieving an accuracy of 57.48%, these variables have some predictive relevance but are insufficient alone for robust predictions.
- **Usage Frequency** and **Total Spend**: With accuracies of 54.71% and 54.27% respectively, these variables provide only minimal predictive value, suggesting they capture limited patterns related to the outcome.
- **Support Calls**: At 64.47% accuracy, Support Calls is a moderately strong predictor, offering some baseline insight. However, further improvements are expected with additional variables.
- **Payment Delays**: With a high accuracy of 76.93%, Payment Delays stands out as the most influential variable, providing reliable predictive power on its own.
- **Last Interaction**: With an accuracy of 52.46%, Last Interaction has minimal predictive relevance, indicating it contributes little on its own.

**Conclusion**: While *Payment Delays* shows strong predictive power, most other variables have limited standalone value. To improve model accuracy and robustness, adding more relevant features would likely capture a fuller picture and better support predictions. This approach would refine the model, leveraging each variable’s unique contribution alongside others to achieve more reliable and nuanced outcomes.

![image](https://github.com/user-attachments/assets/f750a4b2-f7eb-42ab-ba57-b74077ded53b)

