import tkinter as tk
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Sample wine data (features)
wine_data = [
    [7.0, 0.4, 0.3, 2.0, 0.075, 15.0, 50.0, 0.995, 3.2, 0.6, 10.5, 5],
    [6.5, 0.5, 0.2, 2.5, 0.076, 30.0, 50.0, 0.995, 3.3, 0.7, 11.2, 6],
    # Add more samples as needed
]

def predict_quality():
    # Load the wine data
    dataset = pd.DataFrame(wine_data, columns=[
        'Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
        'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density',
        'pH', 'Sulphates', 'Alcohol', 'Quality'  # Include 'Quality' as the target column
    ])

    # Extract features (e.g., various chemical properties)
    features = dataset.drop('Quality', axis=1)

    # Create a random forest classifier model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Sample target variable (wine quality)
    target = dataset['Quality'].astype(int).values  # Convert to integer and get values

    # Fit the model to the data
    model.fit(features, target)

    # Get input values from the GUI
    input_data = [float(entry.get()) for entry in entry_fields]

    # Predict the wine quality using manual data
    predicted_quality = model.predict([input_data])

    # Display the predicted quality
    result_label.config(text=f'Predicted Quality: {predicted_quality[0]}')

root = tk.Tk()
root.title('Wine Quality Prediction')

label_names = ['Fixed Acidity:', 'Volatile Acidity:', 'Citric Acid:', 'Residual Sugar:',
               'Chlorides:', 'Free Sulfur Dioxide:', 'Total Sulfur Dioxide:', 'Density:',
               'pH:', 'Sulphates:', 'Alcohol:']

entry_fields = []

for i, label_name in enumerate(label_names):
    label = ttk.Label(root, text=label_name)
    label.grid(row=i, column=0, padx=10, pady=5)
    
    entry = ttk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    
    entry_fields.append(entry)

predict_button = ttk.Button(root, text='Predict Quality', command=predict_quality)
predict_button.grid(row=len(label_names), columnspan=2, padx=10, pady=10)

result_label = ttk.Label(root, text='', font=('Helvetica', 14))
result_label.grid(row=len(label_names) + 1, columnspan=2, padx=10, pady=10)

root.mainloop()
