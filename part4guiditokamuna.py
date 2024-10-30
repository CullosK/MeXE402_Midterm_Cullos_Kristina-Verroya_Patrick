import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Checkbox Example")
root.attributes('-fullscreen', True)  # Set the window to full screen
root.configure(bg="#e6f7ff")  # Light blue background

# Function to create text boxes for selected checkboxes
def continue_action():
    # Clear previous text boxes
    for widget in text_frame.winfo_children():
        widget.destroy()
    
    selected_options.clear()  # Clear previous selections
    for i, var in enumerate(checkbox_vars):
        if var.get() == 1:
            selected_options.append(checkbox_labels[i])
    
    # Create text boxes for selected options
    for option in selected_options:
        label = tk.Label(text_frame, text=f"Enter value for {option}:", bg="#e6f7ff", font=("Arial", 10))
        label.pack(anchor='w', padx=10, pady=2)  # Reduced padding for a compact layout
        entry = tk.Entry(text_frame, width=30)
        entry.pack(anchor='w', padx=10, pady=2)  # Reduced padding for a compact layout

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
frame = tk.Frame(root, bg="#e6f7ff", padx=20, pady=10)  # Adjusted padding
frame.pack(side=tk.LEFT, padx=20)  # Padded to the left of the screen

# Create a frame for text boxes
text_frame = tk.Frame(root, bg="#e6f7ff")
text_frame.pack(side=tk.LEFT, padx=20)  # Padded to the left of the screen

# List to store the checkbox variables
checkbox_vars = []
# Labels for the checkboxes
checkbox_labels = [
    "Longitude", 
    "Latitude", 
    "Housing Median Age", 
    "Total Rooms", 
    "Total Bedrooms", 
    "Population", 
    "Households",
    "Median Income"
]

# List to store selected options
selected_options = []

# Create checkboxes with the given labels
for i, label in enumerate(checkbox_labels):
    var = tk.IntVar()
    # Alternate background colors for each checkbox
    bg_color = "#cceeff" if i % 2 == 0 else "#b3e0ff"
    checkbox = tk.Checkbutton(frame, text=label, variable=var, bg=bg_color, font=("Arial", 10), anchor='w', selectcolor="#ffcccb")
    checkbox.pack(anchor='w', padx=10, pady=2)  # Reduced padding for a compact layout
    checkbox_vars.append(var)

# Create a continue button
continue_button = tk.Button(root, text="Continue", command=continue_action, bg="#66b3ff", fg="white", font=("Arial", 12), relief='raised')
continue_button.pack(side=tk.LEFT, padx=20, pady=10)  # Padded to the left of the screen

# Add a decorative label at the top and anchor it to the left
title_label = tk.Label(root, text="Linear Regression: California Housing Prices", bg="#e6f7ff", font=("Arial", 14, "bold"))
title_label.pack(anchor='w', padx=20, pady=10)  # Fixed to the left of the screen

# Start the Tkinter event loop
root.mainloop()
