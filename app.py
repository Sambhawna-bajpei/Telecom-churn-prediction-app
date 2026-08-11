"""Streamlit interface for telecom customer churn prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_decision_tree.joblib"
YES_NO = ["No", "Yes"]


@st.cache_resource
def load_model():
    """Load the fitted pipeline once per app session."""
    return joblib.load(MODEL_PATH)


def build_features(inputs: dict[str, object]) -> pd.DataFrame:
    """Recreate the engineered features used when training the model."""
    features = pd.DataFrame([inputs])
    tenure = float(features.at[0, "tenure"])
    total_charges = float(features.at[0, "TotalCharges"])
    features["tenure_group"] = pd.cut(
        features["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    )
    features["average_monthly_spend"] = total_charges / tenure if tenure else float(features.at[0, "MonthlyCharges"])
    add_on_columns = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    features["services_count"] = features[add_on_columns].eq("Yes").sum(axis=1)
    return features


st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="wide")
st.title("Telecom Customer Churn Predictor")
st.caption("Estimate churn risk and select a retention action for an individual customer.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `python src/train_deployment_model.py` first.")
    st.stop()

with st.form("customer_details"):
    profile_column, service_column, billing_column = st.columns(3)

    with profile_column:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = int(st.checkbox("Senior citizen"))
        partner = st.selectbox("Partner", YES_NO)
        dependents = st.selectbox("Dependents", YES_NO)
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with service_column:
        phone_service = st.selectbox("Phone service", YES_NO)
        multiple_lines = st.selectbox("Multiple lines", YES_NO)
        internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security", YES_NO)
        online_backup = st.selectbox("Online backup", YES_NO)
        device_protection = st.selectbox("Device protection", YES_NO)
        tech_support = st.selectbox("Tech support", YES_NO)
        streaming_tv = st.selectbox("Streaming TV", YES_NO)
        streaming_movies = st.selectbox("Streaming movies", YES_NO)

    with billing_column:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless billing", YES_NO)
        payment_method = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=float(monthly_charges * max(tenure, 1)),
            step=10.0,
        )

    submitted = st.form_submit_button("Predict churn risk", type="primary")

if submitted:
    model = load_model()
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    customer_features = build_features(input_data)
    churn_probability = float(model.predict_proba(customer_features)[0, 1])
    churn_prediction = churn_probability >= 0.5

    metric_column, action_column = st.columns([1, 2])
    with metric_column:
        st.metric("Churn probability", f"{churn_probability:.1%}")
        if churn_prediction:
            st.error("High churn risk")
        else:
            st.success("Lower churn risk")
    with action_column:
        st.subheader("Suggested retention action")
        if churn_prediction and contract == "Month-to-month":
            st.write("Offer a contract-conversion incentive or a targeted bundle discount.")
        elif churn_prediction:
            st.write("Schedule proactive outreach and review support or security add-ons.")
        else:
            st.write("Continue normal customer-success engagement and monitor future changes.")

    with st.expander("Processed model input"):
        st.dataframe(customer_features, use_container_width=True)
