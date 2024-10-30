import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sc = StandardScaler()

# Load the dataset
try:
    dataset = pd.read_csv('logisticff.csv')
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = None  # Set dataset to None if loading fails

# Create the main window
root = tk.Tk()
root.title("Checkbox Example")
root.attributes('-fullscreen', True)  # Set the window to full screen
root.configure(bg="#e6f7ff")  # Light blue background

# Label to display prediction results
prediction_label = tk.Label(root, text="", bg="#e6f7ff", font=("Arial", 12, "bold"))
prediction_label.pack(side=tk.LEFT, padx=20, pady=10)

# Label to display accuracy score
accuracy_label = tk.Label(root, text="", bg="#e6f7ff", font=("Arial", 10))
accuracy_label.pack(side=tk.LEFT, padx=20, pady=10)

# Placeholder for the confusion matrix canvas
canvas = None

# Function to create text boxes for selected checkboxes
def continue_action():
    global canvas  # Refer to the global canvas

    if dataset is None:
        prediction_label.config(text="Error: Dataset not loaded.")
        return

    selected_options.clear()  # Clear previous selections

    # Initialize an empty list to hold selected dataset columns
    selected_columns = []

    # Check the selected options and assign values to the variables
    for i, var in enumerate(checkbox_vars):
        if var.get() == 1:
            selected_options.append(checkbox_labels[i])
            selected_columns.append(dataset.iloc[:, i].values.reshape(-1, 1))  # Reshape to 2D for horizontal stacking

    # Concatenate selected columns into a single variable X
    if selected_columns:
        X = np.concatenate(selected_columns, axis=1)  # Use concatenate with axis=1 for horizontal stacking
    else:
        return  # Exit if no selections are made

    # Prepare the target variable y
    global y_test, y_pred, model, X_test
    y = dataset.iloc[:, -1].values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Scale the data
    X_train = sc.fit_transform(X_train)

    # Initialize and fit the model
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    # Make predictions using the test set
    y_pred = model.predict(sc.transform(X_test))

    # Calculate and display the Accuracy score
    accuracy_dec = accuracy_score(y_test, y_pred)
    accuracy = accuracy_dec * 100
    accuracy_label.config(text=f"Accuracy Score: {accuracy:.2f} %")

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Remove previous confusion matrix plot, if any
    if canvas is not None:
        canvas.get_tk_widget().destroy()

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.6)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Embed the confusion matrix plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, padx=20, pady=10)

    # Prepare for user input to predict a new value
    if selected_options:  # Only if there are selected options
        # Clear previous input fields
        for widget in text_frame.winfo_children():
            widget.destroy()

        # Create text boxes for user input for selected options
        user_input = []
        for option in selected_options:
            label = tk.Label(text_frame, text=f"Enter value for {option}:", bg="#e6f7ff", font=("Arial", 10))
            label.pack(anchor='w', padx=10, pady=2)
            entry = tk.Entry(text_frame, width=30)
            entry.pack(anchor='w', padx=10, pady=2)
            user_input.append(entry)

        # Create a button to predict based on user input
        predict_button = tk.Button(text_frame, text="Predict", command=lambda: predict_value(user_input), bg="#66b3ff", fg="white", font=("Arial", 12))
        predict_button.pack(anchor='w', padx=10, pady=10)

def predict_value(user_input):
    # Collect values from user input fields
    input_values = [float(entry.get()) for entry in user_input]

    # Convert the list to a NumPy array and reshape it
    input_array = np.array(input_values).reshape(1, -1)

    # Scale the input before making a prediction
    input_array_scaled = sc.transform(input_array)

    # Make the prediction
    prediction = model.predict(input_array_scaled)

    # Display the prediction result
    prediction_label.config(text=f"Churn Prediction: {prediction[0]:.2f}")

# Function to minimize the window
def minimize_window():
    root.iconify()

# Function to exit the application
def exit_app():
    root.destroy()

# Create a frame for the title bar with buttons
title_frame = tk.Frame(root, bg="#e6f7ff")
title_frame.pack(side=tk.TOP, fill=tk.X)

# Create minimize and exit buttons
minimize_button = tk.Button(title_frame, text="_", command=minimize_window, bg="#ff9966", fg="white", font=("Arial", 12), relief='flat')
minimize_button.pack(side=tk.RIGHT, padx=5)

exit_button = tk.Button(title_frame, text="X", command=exit_app, bg="#ff6666", fg="white", font=("Arial", 12), relief='flat')
exit_button.pack(side=tk.RIGHT)

# Create a frame for checkboxes
frame = tk.Frame(root, bg="#e6f7ff", padx=20, pady=10)
frame.pack(side=tk.LEFT, padx=20)

# Create a frame for text boxes
text_frame = tk.Frame(root, bg="#e6f7ff")
text_frame.pack(side=tk.LEFT, padx=20)

# List to store the checkbox variables
checkbox_vars = []
# Labels for the checkboxes
checkbox_labels = [
    "Age", 
    "Tenure", 
    "Usage Frequency", 
    "Support Calls", 
    "Payment Delay", 
    "Total Spend", 
    "Last Interaction",
]

# List to store selected options
selected_options = []

# Create checkboxes with the given labels
for i, label in enumerate(checkbox_labels):
    var = tk.IntVar()
    bg_color = "#cceeff" if i % 2 == 0 else "#b3e0ff"
    checkbox = tk.Checkbutton(frame, text=label, variable=var, bg=bg_color, font=("Arial", 10), anchor='w', selectcolor="#ffcccb")
    checkbox.pack(anchor='w', padx=10, pady=2)
    checkbox_vars.append(var)

# Create a continue button
continue_button = tk.Button(root, text="Continue", command=continue_action, bg="#66b3ff", fg="white", font=("Arial", 12), relief='raised')
continue_button.pack(side=tk.LEFT, padx=20, pady=10)

# Add a decorative label at the top and anchor it to the left
title_label = tk.Label(root, text="Logistic Regression: Customer Churn Prediction", bg="#e6f7ff", font=("Arial", 14, "bold"))
title_label.pack(anchor='w', padx=20, pady=10)

# Start the Tkinter event loop
root.mainloop()
