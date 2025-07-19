import streamlit as st
import pandas as pd
import pickle
from datetime import date

# Load models and features
rf_model = pickle.load(open("rf_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

def main():
    st.set_page_config(page_title="Loan Predictor", layout="wide")
    # Title and header
    st.markdown("# Loan Status Predictor")
    st.write("Use the form below to input applicant details and predict loan approval status.")

    # Load data for dropdown options
    df_full = pd.read_csv("df.csv")
    age_bands = df_full['AgeBand'].unique().tolist()
    genders = df_full['Gender'].unique().tolist()
    cities = df_full['City'].unique().tolist()
    account_types = df_full['Account Type'].unique().tolist()
    loan_types = df_full['Loan Type'].unique().tolist()
    card_types = df_full['Card Type'].unique().tolist()

    # Sidebar for inputs
    with st.sidebar:
        st.header("Applicant Information")
        data = {}
        for feature in features:
            if feature == "AgeBand":
                data[feature] = st.selectbox("Age Band", age_bands)
            elif feature == "Gender":
                data[feature] = st.selectbox("Gender", genders)
            elif feature == "City":
                data[feature] = st.selectbox("City", cities)
            elif feature == "Account Type":
                data[feature] = st.selectbox("Account Type", account_types)
            elif feature == "Loan Type":
                data[feature] = st.selectbox("Loan Type", loan_types)
            elif feature == "Card Type":
                data[feature] = st.selectbox("Card Type", card_types)
            elif "Balance" in feature:
                data[feature] = st.number_input(feature, min_value=0, format="%d")
            elif "Loan Amount" in feature:
                data[feature] = st.number_input("Loan Amount", min_value=0, step=500, format="%d")
            elif "Loan Term" in feature:
                data[feature] = st.slider("Loan Term (months)", 12, 60, 12)
            elif "Date Of Account Opening" in feature:
                data[feature] = st.date_input("Date of Account Opening", date.today())
            elif "Last Transaction Date" in feature:
                data[feature] = st.date_input("Last Transaction Date", date(2020, 1, 1), min_value=date(2015, 1, 1), max_value=date.today())
            else:
                data[feature] = st.number_input(feature, format="%f")

        st.markdown("---")
        if st.button("Predict"):
            predict_loan(data)

    # Main content area for results
    if 'result' in st.session_state:
        display_results()


def predict_loan(data):
    # Prepare input DataFrame
    input_df = pd.DataFrame([data])
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col] = pd.factorize(input_df[col])[0]

    try:
        pred_rf = rf_model.predict(input_df)[0]
        # pred_xgb = xgb_model.predict(input_df)[0]
        label_map = {0: "Approved", 1: "Rejected", 2: "Closed"}

        st.session_state['result'] = label_map[pred_rf]
    except Exception as e:
        st.error(f"Prediction failed: {e}")


def display_results():
    st.markdown("## Prediction Result")
    st.success(f"**{st.session_state['result']}**")
    # Clear button
    if st.button("Reset"):  
        del st.session_state['result']
        st.experimental_rerun()


if __name__ == "__main__":
    main()
