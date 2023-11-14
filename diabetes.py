import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb

def predict_diabetes(event=None):
    # Get the selected values from the dropdown menus
    gender = gender_var.get()
    age = age_var.get()
    hypertension = hypertension_var.get()
    smoking_history = smoking_var.get()
    heart_disease = heart_disease_var.get()
    bmi = bmi_var.get()
    hba1c_level = hba1c_var.get()
    blood_glucose_level = blood_glucose_var.get()
    
    # Perform any necessary data preprocessing
    gender = 0 if gender=='Female' else (1 if gender=='Male' else 2)
    hypertension = 0 if hypertension=='No' else 1
    smoking_history = 0  if smoking_history=="No Info" else( 1 if smoking_history=="current" else(2 if smoking_history=="ever" else (3 if smoking_history=="former" else (4 if smoking_history=="never" else 5))))
    heart_disease = 0 if heart_disease=='No' else 1
    x = np.asarray([[gender, age, hypertension, smoking_history, heart_disease, bmi, hba1c_level, blood_glucose_level]])
    x = np.asarray(x).astype(np.float32)
    x = normalized_model(x)
    x = x.numpy()
    x = xgb.DMatrix(x)
    # Make the prediction using the loaded model
    prediction = model.predict(x)
    #prediction = tf.nn.sigmoid(prediction)
    # Display the prediction and accuracy
    #accuracy = 100*prediction[0]
    #if prediction[0]<0.5:
    #    accuracy = 100 - accuracy
    
    prediction_label.config(text=f"Diabetes: {'Yes' if prediction[0]>0.5 else 'No'}")

if __name__=="__main__":
    normalized_model = load_model('Normalized_model')
    model = xgb.Booster({'nthread': 2})
    model.load_model('diabetes_model')
    # Create the main window
    window = tk.Tk()
    window.title("Diabetes Prediction")

    # Create the labels and dropdown menus
    gender_label = tk.Label(window, text="Gender:")
    gender_label.grid(row=0, column=0, padx=10, pady=5)
    gender_var = tk.StringVar(window)
    gender_var.set("Male")  # Set the default value
    gender_dropdown = tk.OptionMenu(window, gender_var, "Male", "Female", "Other")
    gender_dropdown.grid(row=0, column=1, padx=10, pady=5)

    age_label = tk.Label(window, text="Age:")
    age_label.grid(row=1, column=0, padx=10, pady=5)
    age_var = tk.StringVar(window)
    age_entry = tk.Entry(window, textvariable=age_var)
    age_entry.grid(row=1, column=1, padx=10, pady=5)

    hypertension_label = tk.Label(window, text="Hypertension:")
    hypertension_label.grid(row=2, column=0, padx=10, pady=5)
    hypertension_var = tk.StringVar(window)
    hypertension_var.set("No")  # Set the default value
    hypertension_dropdown = tk.OptionMenu(window, hypertension_var, "No", "Yes")
    hypertension_dropdown.grid(row=2, column=1, padx=10, pady=5)

    smoking_label = tk.Label(window, text="Smoking History:")
    smoking_label.grid(row=3, column=0, padx=10, pady=5)
    smoking_var = tk.StringVar(window)
    smoking_var.set("No Info")  # Set the default value
    smoking_dropdown = tk.OptionMenu(window, smoking_var, "No Info", "current", "ever", "former", "never", "not current")
    smoking_dropdown.grid(row=3, column=1, padx=10, pady=5)

    heart_disease_label = tk.Label(window, text="Heart Disease:")
    heart_disease_label.grid(row=4, column=0, padx=10, pady=5)
    heart_disease_var = tk.StringVar(window)
    heart_disease_var.set("No")  # Set the default value
    heart_disease_dropdown = tk.OptionMenu(window, heart_disease_var, "No", "Yes")
    heart_disease_dropdown.grid(row=4, column=1, padx=10, pady=5)

    bmi_label = tk.Label(window, text="BMI:")
    bmi_label.grid(row=5, column=0, padx=10, pady=5)
    bmi_var = tk.StringVar(window)
    bmi_entry = tk.Entry(window, textvariable=bmi_var)
    bmi_entry.grid(row=5, column=1, padx=10, pady=5)

    hba1c_label = tk.Label(window, text="HbA1c Level:")
    hba1c_label.grid(row=6, column=0, padx=10, pady=5)
    hba1c_var = tk.StringVar(window)
    hba1c_entry = tk.Entry(window, textvariable=hba1c_var)
    hba1c_entry.grid(row=6, column=1, padx=10, pady=5)

    blood_glucose_label = tk.Label(window, text="Blood Glucose Level:")
    blood_glucose_label.grid(row=7, column=0, padx=10, pady=5)
    blood_glucose_var = tk.StringVar(window)
    blood_glucose_entry = tk.Entry(window, textvariable=blood_glucose_var)
    blood_glucose_entry.grid(row=7, column=1, padx=10, pady=5)

    # Create the predict button
    predict_button = tk.Button(window, text="Predict", command=predict_diabetes)
    predict_button.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

    prediction_label = tk.Label(window, text="")
    prediction_label.grid(row=9,column=0,columnspan=2, pady=10, padx=10, sticky="ns")

    # Bind the Enter key to the submit_name function
    window.bind("<Return>", predict_diabetes)
    window.mainloop()



