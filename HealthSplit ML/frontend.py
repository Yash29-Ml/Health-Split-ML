import streamlit as st
import pickle
import numpy as np

def main():
    # Load models
    charges_model = pickle.load(open("charges_model.pkl", "rb"))
    risk_model = pickle.load(open("risk_model.pkl", "rb"))
    scaler_charges = pickle.load(open("scaler_charges.pkl", "rb"))
    scaler_risk = pickle.load(open("scaler_risk.pkl", "rb"))

    st.title("🏥 Health Charges & Risk Prediction App")

    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.5)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    sex = st.selectbox("Sex", ["male", "female"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    sex_map = {"male": 0, "female": 1}
    smoker_map = {"no": 0, "yes": 1}
    region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}

    input_data = np.array([[
        age,
        sex_map[sex],
        bmi,
        children,
        smoker_map[smoker],
        region_map[region]
    ]])

    if st.button("Predict"):
        # Predict charges
        charges_scaled = scaler_charges.transform(input_data)
        charges_pred = charges_model.predict(charges_scaled)[0]

        # Predict risk
        risk_scaled = scaler_risk.transform(input_data)
        risk_pred = risk_model.predict(risk_scaled)[0]
        risk_prob = risk_model.predict_proba(risk_scaled)[0]

        st.success(f"💰 Estimated Medical Charges: **${charges_pred:.2f}**")
        st.success(f"🩺 Predicted Health Risk Level: **{risk_pred}**")
        st.info(f"Risk Probabilities → Low: {risk_prob[0]*100:.1f}%, Medium: {risk_prob[1]*100:.1f}%, High: {risk_prob[2]*100:.1f}%")

if __name__ == "__main__":
    main()
