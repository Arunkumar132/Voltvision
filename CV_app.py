import streamlit as st
import joblib

# Load the trained models
model1_filename = r"C:\Users\Thamo\Downloads\random_forest_model.pkl"  # Replace with the actual path
model2_filename = r"C:\Users\Thamo\Downloads\DPV.pkl"        # Replace with the path to the second model
loaded_model1 = joblib.load(model1_filename)
loaded_model2 = joblib.load(model2_filename)

# Streamlit app
st.title("VOLT-VISION: An Automated Voltammetry Tool")

st.write("""
### Enter the values for prediction:
""")

# Section for Random Forest Model
st.header("Cyclic Voltammetry")
rf_concentration = st.number_input("CV - Concentration (e.g., 0.02):", 
                                    min_value=0.0, step=0.001, format="%.3f", key="rf_concentration")
rf_potential = st.number_input("CV - Potential (e.g., -0.8):", 
                                step=0.01, format="%.3f", key="rf_potential")
if st.button("Predict Current CV", key="rf_button"):
    try:
        rf_prediction = loaded_model1.predict([[rf_concentration, rf_potential]])[0]
        # Display the full precision value
        st.success(f"Predicted Current CV: {rf_prediction}")
    except Exception as e:
        st.error(f"Error in prediction : {e}")

# Section for Another Model
st.header("Differential Pulse Voltammetry")
another_concentration = st.number_input("DPV - Concentration (e.g., 0.05):", 
                                        min_value=0.0, step=0.001, format="%.3f", key="another_concentration")
another_potential = st.number_input("DPV - Potential (e.g., -0.7):", 
                                    step=0.01, format="%.2f", key="another_potential")
if st.button("Predict Current DPV ", key="another_button"):
    try:
        another_prediction = loaded_model2.predict([[another_concentration, another_potential]])[0]
        # Display the full precision value
        st.success(f"Another Model Predicted Current: {another_prediction}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
