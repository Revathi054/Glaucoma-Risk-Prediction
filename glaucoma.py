# glaucoma_app.py
# Streamlit App for Glaucoma Prediction

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# ------------------ Load Model ------------------
st.title("👁️ Glaucoma Prediction App")

try:
    with open("logreg_model.pkl", "rb") as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Model file 'logreg_model.pkl' not found. Please train and save the model first.")
    st.stop()

# ------------------ Label Encoder (Target Classes) ------------------
# Make sure this matches your training label encoder classes_
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Primary Open Angle", "Angle Closure", "Normal"])  # adjust to your dataset

# ------------------ Sidebar for User Input ------------------
st.sidebar.header("Enter Patient Details")

# Numeric features
age = st.sidebar.slider("Age", min_value=20, max_value=80, value=50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", min_value=10.0, max_value=40.0, value=20.0, step=0.5)
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
pachymetry = st.sidebar.slider("Pachymetry (µm)", min_value=450, max_value=650, value=540)

# Categorical features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
cataract = st.sidebar.selectbox("Cataract Status", ["Present", "Absent"])
angle_closure = st.sidebar.selectbox("Angle Closure Status", ["Yes", "No"])
diagnosis = st.sidebar.selectbox("Diagnosis", ["Glaucoma Suspect", "Confirmed", "None"])
visual_acuity = st.sidebar.selectbox("Visual Acuity Measurements", ["Normal", "Reduced"])

# ------------------ Preprocessing Function ------------------
def preprocess_input(age, iop, cdr, pachymetry, gender, family_history, cataract, angle_closure, diagnosis, visual_acuity):
    data = {
        "Age": age,
        "Intraocular Pressure (IOP)": iop,
        "Cup-to-Disc Ratio (CDR)": cdr,
        "Pachymetry": pachymetry,
    }
    df = pd.DataFrame([data])

    # One-hot encode categorical features
    df[f"Gender_{gender}"] = 1
    df[f"Family History_{family_history}"] = 1
    df[f"Cataract Status_{cataract}"] = 1
    df[f"Angle Closure Status_{angle_closure}"] = 1
    df[f"Diagnosis_{diagnosis}"] = 1
    df[f"Visual Acuity Measurements_{visual_acuity}"] = 1

    # Expected columns (must match training)
    expected_columns = [
        "Age", "Intraocular Pressure (IOP)", "Cup-to-Disc Ratio (CDR)", "Pachymetry",
        "Gender_Female", "Gender_Male",
        "Family History_Yes", "Family History_No",
        "Cataract Status_Present", "Cataract Status_Absent",
        "Angle Closure Status_Yes", "Angle Closure Status_No",
        "Diagnosis_Glaucoma Suspect", "Diagnosis_Confirmed", "Diagnosis_None",
        "Visual Acuity Measurements_Normal", "Visual Acuity Measurements_Reduced"
    ]

    # Add missing columns with 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder
    df = df[expected_columns]
    return df

# ------------------ Prediction ------------------
if st.sidebar.button("Predict Glaucoma Type"):
    input_df = preprocess_input(age, iop, cdr, pachymetry, gender, family_history, cataract, angle_closure, diagnosis, visual_acuity)
    
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.subheader("🔮 Prediction Result")
        st.success(f"The predicted **Glaucoma Type** is: {predicted_label}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
# Load feature names
with open("features.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# After building your input_df
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
# After preprocessing input data (encoding, scaling, etc.)
# Make sure feature alignment is done before prediction



# Now safe to predict
prediction = model.predict(input_df)



# ------------------ Instructions ------------------
st.write("""
### 📌 Instructions
1. Enter patient details in the sidebar.
2. Adjust sliders for **Age, IOP, CDR, Pachymetry**.
3. Select categorical features like Gender, Family History, etc.
4. Click **Predict Glaucoma Type** to see the result.
""")


