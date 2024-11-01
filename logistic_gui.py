import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the dataset
try:
    dataset = pd.read_csv('housing.csv')
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None  # Set dataset to None if loading fails

# Create the main window
root = tk.Tk()
root.title("Linear Regression: California Housing Price")
root.attributes('-fullscreen', True)  # Set the window to full screen
root.configure(bg="#e6f7ff")  # Light blue background

# Title label at the top left
title_label = tk.Label(root, text="Linear Regression: California Housing Price", bg="#e6f7ff", font=("Arial", 16, "bold"))
title_label.pack(anchor='nw', padx=20, pady=10)

# Label to display prediction results
prediction_label = tk.Label(root, text="", bg="#e6f7ff", font=("Arial", 12, "bold"))
prediction_label.pack(side=tk.LEFT, padx=20, pady=10)

# Label to display R-squared and Adjusted R-squared
r2_label = tk.Label(root, text="", bg="#e6f7ff", font=("Arial", 10))
r2_label.pack(side=tk.LEFT, padx=20, pady=10)

# Function to plot the regression line and predicted values
def plot_regression_line(X_test, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_test, y_pred, color='blue', label='Predicted vs. Actual', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Linear Regression: California Housing Price')
    ax.legend()
    ax.grid(True)

    global canvas
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Function to overlay new predictions on the plot
def plot_prediction_overlay(new_input=None, new_prediction=None):
    if new_input is not None and new_prediction is not None:
        canvas.figure.gca().plot(new_input[0], new_prediction, 'go', markersize=10, label='Input Prediction')
        canvas.draw()

# Function to create text boxes for selected checkboxes
def continue_action():
    if dataset is None:
        prediction_label.config(text="Error: Dataset not loaded.")
        return

    selected_options.clear()
    selected_columns = []

    for i, var in enumerate(checkbox_vars):
        if var.get() == 1:
            selected_options.append(checkbox_labels[i])
            selected_columns.append(dataset.iloc[:, i].values.reshape(-1, 1))

    if selected_columns:
        X = np.concatenate(selected_columns, axis=1)
    else:
        return

    global y_test, y_pred, model, X_test
    y = dataset.iloc[:, -2].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    k = X_test.shape[1]
    n = X_test.shape[0]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    r2_label.config(text=f"R-squared: {r2:.4f}\nAdjusted R-squared: {adj_r2:.4f}")

    predict_button.config(state=tk.NORMAL)

    if selected_options:
        for widget in text_frame.winfo_children():
            widget.destroy()

        user_input.clear()
        for option in selected_options:
            label = tk.Label(text_frame, text=f"Enter value for {option}:", bg="#e6f7ff", font=("Arial", 10))
            label.pack(anchor='w', padx=10, pady=2)
            entry = tk.Entry(text_frame, width=30)
            entry.pack(anchor='w', padx=10, pady=2)
            user_input.append(entry)

    plot_regression_line(X_test, y_test, y_pred)

def predict_value():
    input_values = [float(entry.get()) for entry in user_input]
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)

    # Display the predicted value in two rows
    prediction_label.config(text=f"Prediction Result:\n${prediction[0]:,.2f}")

    # Overlay the new prediction on the existing plot
    plot_prediction_overlay(new_input=[input_values], new_prediction=prediction)

# Function to minimize the window
def minimize_window():
    root.iconify()

# Function to exit the application
def exit_app():
    root.destroy()

# Title bar buttons
title_frame = tk.Frame(root, bg="#e6f7ff")
title_frame.pack(side=tk.TOP, fill=tk.X)

minimize_button = tk.Button(title_frame, text="_", command=minimize_window, bg="#ff9966", fg="white", font=("Arial", 12), relief='flat')
minimize_button.pack(side=tk.RIGHT, padx=5)

exit_button = tk.Button(title_frame, text="X", command=exit_app, bg="#ff6666", fg="white", font=("Arial", 12), relief='flat')
exit_button.pack(side=tk.RIGHT)

# Frames for checkboxes, text boxes, and plot
frame = tk.Frame(root, bg="#e6f7ff", padx=20, pady=10)
frame.pack(side=tk.LEFT, padx=20, pady=10)

text_frame = tk.Frame(root, bg="#e6f7ff")
text_frame.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=10)

plot_frame = tk.Frame(root, bg="#e6f7ff")
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Checkbox and button setup
checkbox_vars = []
checkbox_labels = ["Longitude", "Latitude", "Housing Median Age", "Total Rooms", "Total Bedrooms", "Population", "Households", "Median Income"]

selected_options = []
user_input = []

for i, label in enumerate(checkbox_labels):
    var = tk.IntVar()
    bg_color = "#cceeff" if i % 2 == 0 else "#b3e0ff"
    checkbox = tk.Checkbutton(frame, text=label, variable=var, bg=bg_color, font=("Arial", 10), anchor='w', selectcolor="#ffcccb")
    checkbox.pack(anchor='w', padx=10, pady=2)
    checkbox_vars.append(var)

button_frame = tk.Frame(frame, bg="#e6f7ff")
button_frame.pack(anchor='w', padx=10, pady=10)

continue_button = tk.Button(button_frame, text="Continue", command=continue_action, bg="#66b3ff", fg="white", font=("Arial", 12), relief='raised')
continue_button.pack(side=tk.LEFT, padx=5)

predict_button = tk.Button(button_frame, text="Predict", command=predict_value, bg="#66b3ff", fg="white", font=("Arial", 12), relief='raised')
predict_button.pack(side=tk.LEFT, padx=5)
predict_button.config(state=tk.DISABLED)

# Run the application
root.mainloop()
