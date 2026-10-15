import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="HydroPredict - Aqua Epoch", page_icon="💧", layout="wide")

st.title("💧 HydroPredict: Real-Time Water Diagnostic System")
st.markdown("### techFEST'26 | The Aqua-Epoch Competition")

st.sidebar.header("Water Parameters Input")

hardness = st.sidebar.number_input("Hardness (mg/L)", 0.0, 500.0, 204.89)
solids = st.sidebar.number_input("Total Dissolved Solids (ppm)", 0.0, 60000.0, 20791.32)
chloramines = st.sidebar.number_input("Chloramines (ppm)", 0.0, 15.0, 7.30)
sulfate = st.sidebar.number_input("Sulfate (mg/L)", 0.0, 600.0, 368.51)
conductivity = st.sidebar.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 421.88)
organic_carbon = st.sidebar.number_input("Organic Carbon (ppm)", 0.0, 35.0, 10.37)
trihalomethanes = st.sidebar.number_input("Trihalomethanes (μg/L)", 0.0, 130.0, 66.42)
turbidity = st.sidebar.number_input("Turbidity (NTU)", 0.0, 10.0, 3.96)

if st.button("Run Model Inference", type="primary"):
    try:
        imputer = joblib.load('imputer.joblib')
        scaler = joblib.load('scaler.joblib')
        clf = joblib.load('clf_model.joblib')
        reg = joblib.load('reg_model.joblib')
        
        features = np.array([[hardness, solids, chloramines, sulfate, conductivity, organic_carbon, trihalomethanes, turbidity]])
        feat_scaled = scaler.transform(imputer.transform(features))
        
        potability = clf.predict(feat_scaled)[0]
        prob = clf.predict_proba(feat_scaled)[0][1]
        ph_val = reg.predict(feat_scaled)[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Task A: Water Safety Classification")
            if potability == 1:
                st.success(f"**Status: SAFE / POTABLE**\nConfidence: {prob*100:.1f}%")
            else:
                st.error(f"**Status: UNSAFE / NON-POTABLE**\nConfidence: {(1-prob)*100:.1f}%")
                
        with col2:
            st.subheader("Task B: pH Value Regression")
            st.info(f"**Predicted pH:** {ph_val:.2f}")
            if 6.5 <= ph_val <= 8.5:
                st.caption("✅ Complies with WHO drinking water standards (6.5 - 8.5)")
            else:
                st.caption("⚠️ Non-standard pH level")
    except Exception as e:
        st.error(f"Error loading models. Please ensure train_model.py has been run: {e}")