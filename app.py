import streamlit as st
import joblib

# Load the trained models
model1_filename = r"C:\Users\Thamo\Downloads\random_forest_model.pkl"  # Replace with the actual path
model2_filename = r"C:\Users\Thamo\Downloads\DPV.pkl"        # Replace with the path to the second model
loaded_model1 = joblib.load(model1_filename)
loaded_model2 = joblib.load(model2_filename)

# Streamlit app with enhanced visuals
st.set_page_config(page_title="VOLT-VISION", page_icon="⚡", layout="wide")

# Title and description with styled text
st.markdown("""
<style>
.title {
    font-size: 50px;
    color: #2C3E50;
    text-align: center;
    font-weight: bold;
}
.description {
    font-size: 20px;
    text-align: center;
    color: #555;
    margin-bottom: 30px;
    padding: 15px;
    border-radius: 10px;
    background: linear-gradient(to right, #16A085, #1ABC9C);
}
</style>
<div class="title">⚡ VOLT-VISION ⚡</div>
<div class="description">An Advanced Voltammetry Tool to Predict Electrical Currents Accurately</div>
""", unsafe_allow_html=True)

# Side-by-side sections with enhanced visuals
col1, col2 = st.columns(2)

# Section for Random Forest Model
with col1:
    st.header("🔋 Cyclic Voltammetry")
    st.write("### Input Values")
    rf_concentration = st.number_input("Concentration (e.g., 0.02):", min_value=0.0, step=0.001, format="%.3f", key="rf_concentration")
    rf_potential = st.number_input("Potential (e.g., -0.8):", step=0.01, format="%.3f", key="rf_potential")
    if st.button("🌟 Predict Current (CV)", key="rf_button"):
        try:
            rf_prediction = loaded_model1.predict([[rf_concentration, rf_potential]])[0]
            st.success(f"Predicted Current (CV): {rf_prediction} A")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Section for Another Model
with col2:
    st.header("🔬 Differential Pulse Voltammetry")
    st.write("### Input Values")
    another_concentration = st.number_input("Concentration (e.g., 0.05):", min_value=0.0, step=0.001, format="%.3f", key="another_concentration")
    another_potential = st.number_input("Potential (e.g., -0.7):", step=0.01, format="%.2f", key="another_potential")
    if st.button("🌟 Predict Current (DPV)", key="another_button"):
        try:
            another_prediction = loaded_model2.predict([[another_concentration, another_potential]])[0]
            st.success(f"Predicted Current (DPV): {another_prediction} A")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Add a gradient footer with additional style
st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to right, #16A085, #1ABC9C);
    color: white;
    text-align: center;
    padding: 10px 0;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1);
}
</style>
<div class="footer">Developed by Team Vikings | Powered by Streamlit</div>
""", unsafe_allow_html=True)
