
# MeXE402_Midterm_Cullos_Kristina-Verroya_Patrick
## Introduction: Overview of Linear and Logistic Regression.
### Linear Regression
• Purpose: Linear regression is a supervised learning algorithm used for predicting continuous numerical outcomes. Its goal is to model the relationship between one or more independent variables (features) and a dependent variable (target) by fitting a linear equation to observed data.

• Types: Simple Linear Regression that uses one independent variable and Multiple Linear Regression that uses more than one independent variable.

• Applications: Linear regression is used to predict housing prices based on features like size, location, and amenities and also to estimate stock prices based on market variables.

• Assumptions: Linearity between independent and dependent variables, independence of errors, homoscedasticity (constant variance of errors), and normality of residuals.

### Logistic Regression
• Purpose: Logistic regression is a supervised learning algorithm used for binary classification (e.g., 0 or 1, yes or no, true or false). Unlike linear regression, it predicts the probability that a given input point belongs to a particular class.

• Decision Boundary: By default, probabilities greater than 0.5 are classified as 1, and those below 0.5 as 0.

• Types: Binary Logistic Regression that predicts between two classes (0 or 1), Multinomial Logistic Regression that extend for more than two classes, and Ordinal Logistic Regression that predicts ordered outcomes.

• Applications: Predicting whether a customer will purchase a product (yes/no) or for medical diagnosis (e.g., disease present or not) is the application of logistic regression.

• Assumptions: Independence of observations, linearity of independent variables and log-odds, and large sample size (for reliable results).

# Dataset Description

## Dataset used in Linear Regression
* The data used in Linear Regression is named **"housing.csv"**.
* The original data consist of 20641 rows and 10 rows.


## Dataset used in Linear Regression
* The data used in Logistic Regression is named **"customer_churn_dataset-testing-master.csv"** and later renamed to **"logistic.csv"**.
* The original data consists of 64,735 rows and 12 rows.
* The Following Columns are removed from the dataset: Customer ID, Gender, Subscription Type, and Contract Length.
![alt text](<Image Resources/Logistic Image Resources/logistic_removed.png>)
* After the process, the data set used consist of **64,735 rows and 8 rows.**

# Project Objectives
* To create a machine learning model using a linear regression model with an R-squared between 0.50 to 0.99.
* To create a machine learning model using a logistic regression model with at least 75% Accuracy.
* To analyze the importance of each variable to the outcome of linear and logistic model prediction,

# Methodology
# Linear Regression
## Data Cleaning Process

* The data sheets contains the house pricinh
*  The data is cleaned using Microsoft Excel and 
* The **rows that are incomplete are completely removed** for more accurate reading.
* Here are the step by step process on deleting unwanted rows

    1. Open Your Excel File: Find the rows using **Find and Select** then **Go to Special**.

   ![alt text](<Image Resources/Go to Special.png>)

    2. After selecting **Go to Special**. Select the **Blanks** to search for an empty cell.

    ![alt text](<Image Resources/image6.png>)

    3. Highlight the rows you want to delete.

    ![alt text](<Image Resources/image7.png>)

* Deleted rows are rows 292 and 342

### Details about the used variables in Linear Regression

1. **Longitude**: A measure of how far west a house is; a higher value is farther west.
   
2. **Latitude**: A measure of how far north a house is; a higher value is farther north.
   
3. **Housing Median Age**: Median age of a house within a block; a lower number is a newer building.
   
4. **Total Rooms**: Total number of rooms within a block.
   
5. **Total Bedrooms**: Total number of bedrooms within a block.
    
6. **Population**: Total number of people residing within a block.
    
7. **Households**: Total number of households, a group of people residing within a home unit, for a block.
    
8. **Median Income**: Median income for households within a block of houses (measured in tens of thousands of US Dollars).
    
9. **Median HouseValue**: Median house value for households within a block (measured in US Dollars).
    
10. **Ocean Proximity**: Location of the house w.r.t ocean/sea.


## Part 1 - Data Preprocessing


## Importing libraries and the dataset

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


```python
#variable and this is a function for uploading the dataset
dataset = pd.read_csv('housing.csv') 

```
* Loads the data from housing.csv into the dataset variable, allowing the data to be manipulated and analyzed in code.


```python

dataset.head() #display 5 rows of dataset

# 20,434 rows
# Data points collected from a combined cycle power plant over six years
# 9 columns: longitude, latitude, housing median age, total rooms, total bedrooms, populations, households, median income, median house value
# independent variables: longitude, latitude, housing median age, total rooms, total bedrooms, populations, households, median income
# dependent variable:  median house value
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




```python

X_train
X_test
y_train
y_test

```
* Defining X and Y training and testing variables.

## Part 2 - Building and training the model

### Building the model

* from sklearn.linear_model import LinearRegression imports the LinearRegression class.

```python

# linear_model is the module
# `LinearRegression is a class` defines that `LinearRegression` is a class within the `linear_model` module. It indicates that `LinearRegression` is a blueprint or template for creating objects that represent linear regression models.
# Class is a pre-coded blueprint of something we want to build from which objects are created.
from sklearn.linear_model import LinearRegression
model = LinearRegression()

```

### Training the Model

```python

# fit is a method inside LinearRegression class - they are like functions.
model.fit(X_train, y_train)

```
* Fitting the value of x_train and y_train for linear refression model
![alt text](<Image Resources/image.png>)


### Inference
* Predicting

```python
y_pred = model.predict(X_test)
y_pred
```
* y_pred is used in the predicting with x_test as the x variables

### Making the prediction of a single data point with Longitude = -122.23, Latitude = 37.84, Housing Median Age = 50, Total rooms = 2515, Total Bedrooms = 399 , Populations = 970, Households = 373, Median Income = 5.8596

#### Test Sample (Row No. 120) = -122.23,37.84,50,2515,399,970,373,5.8596
![alt text](<Image Resources/image3.png>)
#### Actual Value = $327,600


### Prediction Model

```python
model.predict([[-122.23,37.84,50,2515,399,970,373,5.8596]])
```

#### Predicted Value = $328,762.40

## Part 3: Evaluating the Model
* Evaluating how effective the model is:

### R-Squared
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2
```

### Adjusted R-Squared
```python
k = X_test.shape[1]
n = X_test.shape[0]
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
adj_r2
```


## Part 4 Data Visualiziation

### Linear Regression Model Graph of Training Variables

* Relationship between x_train and y_train
* visual representation of the:
    * `python model = LinearRegression() model.fit(X_train, y_train)`
    * Linear regression model between each independent variable in relation to the dependent variable.
    
## Dependent Variable and Independent Variable Linear Regression Model 
```python
# Plotting the linear regression
feature_index = 0  # Change this index to visualize other features
X_feature_train = X_train[:, feature_index]

# Create the plot
plt.figure(figsize=(10, 6))
sns.regplot(x=X_feature_train, y=y_train, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
plt.xlabel('Longitude')  # Label for the chosen feature
plt.ylabel('Target Value')  # Label for the target variable
plt.title('Linear Regression: Training Data')  # Title for the plot
plt.grid()

# Set y-axis limit
plt.ylim(0, 500000)  # Set y-axis limit from 0 to 500,000
plt.show()
```

* This code shows or plots the linear regression between independent variables and median house value.
* "feature_index" values are set from 0 to 6 to obtain individual variable results.
   * feature_index = 0 - Longitude
   * feature_index = 1 - Latitude
   * feature_index = 2 - Housing Median Age
   * feature_index = 3 - Total Rooms
   * feature_index = 4 - Total Bedrooms
   * feature_index = 5 - Population
   * feature_index = 6 - Median Income

* plt.xlim() is also used to adjust the range of the x-axis for better data viewing




## Logistic Regression Data
### Data Cleaning Process


* The original data consists of 64,375 rows and 12 rows. The data is cleaned using Microsoft Excel and 
* The **rows that are incomplete are completely removed** for more accurate reading.
* Here is the step by step process for deleting unwanted rows:

   1. Open Your Excel File: Find the rows using **Find and Select** then **Go to Special**.
      ![Screenshot 2024-10-30 110744](https://github.com/user-attachments/assets/23f20c0d-d00f-4e2e-ac74-d30f9a324146)

   3. Select the **Blanks** to search for empty cell.
      ![Screenshot 2024-10-30 111851](https://github.com/user-attachments/assets/af9e99ec-0918-4d44-8cb7-a32e6935633d)

   5. Since the result found nothing blank in the data, we can proceed for the data processing.
      ![Screenshot 2024-10-30 112323](https://github.com/user-attachments/assets/3c4607d2-667d-4678-b90b-822556043dad)
      

### Details about the used variables in Logistic Regression

* **Age**: The age of a user can influence how often they use the product, how they interact with its features, and how old the person is using the product.

* **Gender**: Gender may impact product usage patterns, as different demographics often have unique preferences and needs.

* **Tenure**: Tenure, or how long a user has been using the product, can affect their familiarity and loyalty to the product.

* **Usage Frequency**: Usage frequency measures how often a user engages with the product, indicating their level of dependence or satisfaction.

* **Support Calls**: The number of support calls reflects how frequently a user needs assistance, which may suggest their ease of use or any issues with the product.

* **Payment Delay**: Payment delay records whether users consistently pay on time, providing insight into their financial commitment to the product.



# Logistic Regression
* Logistic Regression presents the learning algorithm used for binary classification.
* This works with numerical inputs, so categorical data (e.g., gender, product type) needs to be encoded. Common techniques include.

## Part-1

### Importing libraries and the dataset

### **Libraries:**
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

* Imports the data from logistic.csv into the dataset variable, enabling data manipulation and analysis in code.

```python
#variable and this is a function for uploading the dataset
dataset = pd.read_csv('logistic.csv')

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

* Making the predictons 
  
```python
y_pred = model.predict(sc.transform(X_test))
y_pred

```

### Predicting with the following variables:

#### Row 27241

1. Age = 42
2. Tenure = 46
3. Usage Frequency = 10
4. Support Calls = 0
5. Payment Delays = 26
6. Total Spend = 313
7. Last Interaction = 3

```python
#42,46,10,0,26,313,3
# Actual Churn Prediction = 1
model.predict(sc.transform([[42,46,10,0,26,313,3]]))
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

## Part 4 Data Visualiziation

### Overall Accuracy and Confusion Matrix
* This code is used for the representation of the overall accuracy and confusion matrix.

```python

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# Assuming y_test and y_pred are defined earlier in your code
# y_test = [...]  # Your true labels
# y_pred = [...]  # Your predicted labels

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy as a percentage
accuracy = accuracy_score(y_test, y_pred) * 100

# Set up the figure
plt.figure(figsize=(8, 6))

# Plot the confusion matrix with a more refined color map and style
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix', fontsize=20, weight='bold', pad=20)
plt.suptitle(f'Accuracy: {accuracy:.2f}%', fontsize=16, color='darkslategray', weight='bold', y=0.92)

# Define tick marks based on the number of unique classes
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test), fontsize=12, weight='bold')
plt.yticks(tick_marks, np.unique(y_test), fontsize=12, weight='bold')

# Set labels for the axes with bold fonts
plt.ylabel('True Label', fontsize=14, weight='bold', labelpad=15)
plt.xlabel('Predicted Label', fontsize=14, weight='bold', labelpad=15)

# Annotate each cell in the confusion matrix with the count
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
             color='white' if cm[i, j] > thresh else 'black',
             fontsize=14, fontweight='bold')

# Customize color bar for a subtle, integrated look
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

# Minor gridlines for a polished separation of cells
plt.gca().set_xticks(np.arange(-.5, len(tick_marks)), minor=True)
plt.gca().set_yticks(np.arange(-.5, len(tick_marks)), minor=True)
plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
plt.tick_params(which='minor', bottom=False, left=False)  # Hide minor ticks

# Adjust layout for a clean, professional look
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
```

![image](https://github.com/user-attachments/assets/8fac340a-078a-46a1-9747-9156bdbdaf5e)



### Individual Accuracy of Variables and Confusion Matrix
* This code is used to represent individual accuracy and confusion matrix.
* 
```python
X = dataset.iloc[:,0:1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
model.fit(X_train, y_train)
y_pred = model.predict(sc.transform(X_test))

cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy as a percentage
accuracy = accuracy_score(y_test, y_pred) * 100

# Set up the figure
plt.figure(figsize=(8, 6))

# Plot the confusion matrix with a more refined color map and style
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix', fontsize=20, weight='bold', pad=20)
plt.suptitle(f'Accuracy: {accuracy:.2f}%', fontsize=16, color='darkslategray', weight='bold', y=0.92)

# Define tick marks based on the number of unique classes
tick_marks = np.arange(len(np.unique(y_test)))
plt.xticks(tick_marks, np.unique(y_test), fontsize=12, weight='bold')
plt.yticks(tick_marks, np.unique(y_test), fontsize=12, weight='bold')

# Set labels for the axes with bold fonts
plt.ylabel('True Label', fontsize=14, weight='bold', labelpad=15)
plt.xlabel('Predicted Label', fontsize=14, weight='bold', labelpad=15)

# Annotate each cell in the confusion matrix with the count
thresh = cm.max() / 2
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]}', ha='center', va='center',
             color='white' if cm[i, j] > thresh else 'black',
             fontsize=14, fontweight='bold')

# Customize color bar for a subtle, integrated look
cbar = plt.colorbar(fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)

# Minor gridlines for a polished separation of cells
plt.gca().set_xticks(np.arange(-.5, len(tick_marks)), minor=True)
plt.gca().set_yticks(np.arange(-.5, len(tick_marks)), minor=True)
plt.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
plt.tick_params(which='minor', bottom=False, left=False)  # Hide minor ticks

# Adjust layout for a clean, professional look
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
```

* "X = dataset.iloc[:,0:1].values" values are set from [:,0:1] to [:,6:7] to obtain individual variable results.
   * X = dataset.iloc[:,0:1] - Age
   * X = dataset.iloc[:,1:2] - Tenure
   * X = dataset.iloc[:,2:3] - Usage Frequency
   * X = dataset.iloc[:,3:4] - Support Calls
   * X = dataset.iloc[:,4:5] - Payment Delays
   * X = dataset.iloc[:,5:6] - Total Spend
   * X = dataset.iloc[:,6:7] - Last Interaction



# Interference using GUI
## Linear Regression
* Based on the model the value of the following variables are set t0:
   1. Longitude = -122.23
   2. Latitude = 37.84
   3. Housing Median Age = 50
   4. Total rooms = 2515
   5. Total Bedrooms = 399
   6. Populations = 970
   7. Households = 373
   8. Median Income = 5.8596

* Actual
   * House Median Value = $327,600

* Model Prediction
   * House Median Value = $328,762.41
   * Accuracy = 81.32 %

![alt text](<Image Resources/Linear Regresion Image Resources/Linear Regression Interface.png>)


## Logistic Regression
* Based on the model the value of the following variables are set t0:
   1. Age = 42
   2. Tenure = 46
   3. Usage Frequency = 10
   4. Support Calls = 0
   5. Payment Delays = 26
   6. Total Spend = 313
   7. Last Interaction = 3

![alt text](<Image Resources/Logistic Image Resources/Logistic Regression Interference.png>)

* Actual
   *Churn = 1 
* Model Prediction
   * Churn = 1
   * Accuracy = 81.32 %

# Summary
# Linear Regression

## Variable Importance Analysis Based on R-squared

* Explanation 
![alt text](<Image Resources/Linear Regresion Image Resources/Variable Importance Base on R2.png>)



## Linear Regression Model Graph of Training Variables

### Longitude

* The regression line doesn’t capture much variability in the data points, showing a **poor fit.**
* This suggests that longitude alone is **not a strong predictor** of housing value in this dataset.
* There’s a **minor negative trend** between longitude and housing value.
* **Other factors** likely play a larger role in determining housing values.

![alt text](<Image Resources/Linear Regresion Image Resources/Longitude.png>)

### Latitude
* The regression line **does not capture much of the data variability**, with data points spread widely around it.
* This suggests that latitude alone is **not a strong predictor** of housing value in this dataset.
* A **minor negative trend** exists between latitude and housing value.
* High variability at each latitude suggests that **other factors likely play a more significant role** in determining housing values.

![alt text](<Image Resources/Linear Regresion Image Resources/Latitude.png>)


### Housing Median Age
* The regression line captures only a **minor trend**, with a **lot of spread** around it, indicating **weak predictive power**.
* This suggests that housing age alone **does not strongly predict housing value.**
* There is a **minor positive trend** between housing age and value.
* Significant data spread at each age suggests **other factors are likely more influential** in determining housing values.

![alt text](<Image Resources/Linear Regresion Image Resources/Housing Median Age.png>)

### Total Rooms

* There is a positive correlation between total rooms and housing value, though it is relatively weak.
![alt text](<Image Resources/Linear Regresion Image Resources/Total Rooms.png>)


![alt text](<Image Resources/Linear Regresion Image Resources/Total Rooms Highlight.png>)'


* The upward trend in the regression line indicates a **moderate positive correlation** between total rooms and housing value.
* A higher number of rooms generally corresponds to a higher housing value, but many exceptions are observed in the data distribution.
![alt text](<Image Resources/Linear Regresion Image Resources/Total Rooms Zoom.png>)



### Total Bedrooms

* The positive slope indicates a **minimal positive correlation** between total bedrooms and housing median value.

* The spread of data points around the line, especially at higher bedroom counts, shows that the **relationship is weak** and other factors likely influence housing values more.

![alt text](<Image Resources/Linear Regresion Image Resources/Total Bedrooms.png>)


![alt text](<Image Resources/Linear Regresion Image Resources/Totad Bedrooms Highlight.png>)

* There is a slight positive correlation between the number of bedrooms and housing median value, but the relationship is weak.
* Housing value generally increases with the number of bedrooms, but the data’s spread indicates that other variables may play a more significant role in determining housing value.

![alt text](<Image Resources/Linear Regresion Image Resources/Total Bedrooms zoom.png>)

### Populations
* The negative slope indicates a **minimal negative correlation** between population and housing median value.
* The downward slope of the line suggests a negative correlation. This means that as the population increases, the target value tends to decrease.

![alt text](<Image Resources/Linear Regresion Image Resources/Population.png>)

![alt text](<Image Resources/Linear Regresion Image Resources/Population Zoom.png>)

### Households

![alt text](<Image Resources/Linear Regresion Image Resources/Households.png>)

![alt text](<Image Resources/Linear Regresion Image Resources/Household Zoom.png>)

### Median Income

![alt text](<Image Resources/Linear Regresion Image Resources/Median Incom.png>)

![alt text](<Image Resources/Linear Regresion Image Resources/Median Income Zoom.png>)


## Training vs Testing
* Scatter Plot shows how the training and testing variable are plot wiith respect to each other.
   * Blue - Training Data
   * Orange - Testing Data
![alt text](<Image Resources/Linear Regresion Image Resources/Training vs Testing Predictions.png>)

# Logistic Regression

## Variable Importance Analysis based on Accuracy

* "Variable Importance Analysis based on Accuracy" is a method used in machine learning and statistical modeling to evaluate the significance of each input variable in predicting an outcome. Here’s a breakdown of how it works:

  * Variable Importance: This refers to the contribution of each predictor (or feature) in a model. It shows how much each variable contributes to the model’s accuracy, helping to identify the most influential variables.

  * Based on Accuracy: This approach measures the importance of assessing how each variable affects the overall model accuracy. Generally, this is done by:

  * Training the model with all features to get a baseline accuracy.
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

**Conclusion**: 
* While *Payment Delays* show strong predictive power, most other variables have limited standalone value.
* To improve model accuracy and robustness, adding more relevant features would likely capture a fuller picture and better support predictions.
* This approach would refine the model, leveraging each variable’s unique contribution alongside others to achieve more reliable and nuanced outcomes.

![image](https://github.com/user-attachments/assets/f750a4b2-f7eb-42ab-ba57-b74077ded53b)


### Details about the used variables in Logistic Regression

* **Age**: The age of a user can influence how often they use the product, how they interact with its features, and how old the person is using the product.

* **Gender**: Gender may impact product usage patterns, as different demographics often have unique preferences and needs.

* **Tenure**: Tenure, or how long a user has been using the product, can affect their familiarity and loyalty to the product.

* **Usage Frequency**: Usage frequency measures how often a user engages with the product, indicating their level of dependence or satisfaction.

* **Support Calls**: The number of support calls reflects how frequently a user needs assistance, which may suggest their ease of use or any issues with the product.

* **Payment Delay**: Payment delay records whether users consistently pay on time, providing insight into their financial commitment to the product.
