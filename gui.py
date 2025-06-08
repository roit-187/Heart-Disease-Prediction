import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import numpy as np
import os

from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and assets
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")

def predict(file_path):
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        ids = df['id']
        original_sex = df['sex']

        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))

        # Preprocess
        X = df.drop(columns=['id', 'dataset'], errors='ignore')
        X = pd.get_dummies(X, drop_first=True)
        X = X.reindex(columns=model_columns, fill_value=0)
        X_scaled = scaler.transform(X)

        # Predict
        predictions = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Result dataframe
        df_results = pd.DataFrame({
            'id': ids,
            'age': df['age'],
            'sex': original_sex,
            'predicted': ['Disease' if p == 1 else 'No Disease' for p in predictions],
            'probability': probs
        })

        # Save results
        result_path = "gui_predicted_results.csv"
        df_results.to_csv(result_path, index=False)

        # Confusion Matrix and ROC
        if 'target' in df.columns:
            y_true = df['target']
            conf = confusion_matrix(y_true, predictions)
            sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.show()

            fpr, tpr, _ = roc_curve(y_true, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='darkorange')
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            plt.grid()
            plt.show()

        messagebox.showinfo("Prediction Complete", f"Predictions saved to {result_path}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        predict(file_path)

# GUI window
window = tk.Tk()
window.title("Heart Disease Prediction")
window.geometry("400x200")

label = tk.Label(window, text="Upload a CSV file to predict heart disease", font=("Arial", 12))
label.pack(pady=20)

upload_btn = tk.Button(window, text="Upload CSV", command=browse_file, font=("Arial", 12))
upload_btn.pack()

window.mainloop()
