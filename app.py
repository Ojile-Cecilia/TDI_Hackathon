
import streamlit as st
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier  # or whichever classes are used
import pickle
from datetime import date

rf_model = pickle.load(open("rf_model.pkl", "rb"))
xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
features = pickle.load(open("model_features.pkl", "rb"))

st.set_page_config(page_title="Loan Predictor")
st.title("Loan Status Predictor")

# Load full dataframe to extract dropdown options

df_full = pd.read_csv("df.csv")

# Extract unique values for dropdowns
age_bands = df_full['AgeBand'].unique().tolist()
genders = df_full['Gender'].unique().tolist()
cities = df_full['City'].unique().tolist()
account_types = df_full['Account Type'].unique().tolist()
loan_types = df_full['Loan Type'].unique().tolist()
card_types = df_full['Card Type'].unique().tolist()

# UI Input collection
data = {}
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
        data[feature] = st.number_input(feature, min_value=0)
    elif "Loan Amount" in feature:
        data[feature] = st.number_input("Loan Amount", min_value=0, step=500)
    elif "Loan Term" in feature:
        data[feature] = st.slider("Loan Term", 12, 60, 12)
    elif "Date Of Account Opening" in feature:
        data[feature] = st.date_input("Date of Account Opening", date.today())
    elif "Last Transaction Date" in feature:
        data[feature] = st.date_input("Last Transaction Date", date(2020, 1, 1), min_value=date(2015, 1, 1), max_value=date.today())
    else:
        data[feature] = st.number_input(feature)

# Convert categorical to numerical as needed
input_df = pd.DataFrame([data])
for col in input_df.select_dtypes(include='object').columns:
    input_df[col] = pd.factorize(input_df[col])[0]

if st.button("Predict"):
    try:
        pred_rf = rf_model.predict(input_df)[0]
        pred_xgb = xgb_model.predict(input_df)[0]
        label_map = {0: "Approved", 1: "Rejected", 2: "Closed"}

        st.success(f"Predicted Loan Status: {label_map[pred_rf]}")
        # st.success(f"XGBoost: {label_map[pred_xgb]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")




# import streamlit as st
# import pandas as pd
# import pickle
# from datetime import date

# rf_model = pickle.load(open("rf_model.pkl", "rb"))
# xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
# features = pickle.load(open("model_features.pkl", "rb"))

# st.set_page_config(page_title="Loan Predictor")
# st.title("Loan Status Predictor")

# # Load full dataframe to extract dropdown options
# df_full = pd.read_csv("df.csv")

# # Extract unique values for dropdowns
# age_bands = df_full['AgeBand'].unique().tolist()
# genders = df_full['Gender'].unique().tolist()
# cities = df_full['City'].unique().tolist()
# account_types = df_full['Account Type'].unique().tolist()
# loan_types = df_full['Loan Type'].unique().tolist()
# card_types = df_full['Card Type'].unique().tolist()

# # UI Input collection
# data = {}
# data = {}
# for feature in features:
#     if feature == "AgeBand":
#         data[feature] = st.selectbox("Age Band", age_bands)
#     elif feature == "Gender":
#         data[feature] = st.selectbox("Gender", genders)
#     elif feature == "City":
#         data[feature] = st.selectbox("City", cities)
#     elif feature == "Account Type":
#         data[feature] = st.selectbox("Account Type", account_types)
#     elif feature == "Loan Type":
#         data[feature] = st.selectbox("Loan Type", loan_types)
#     elif feature == "Card Type":
#         data[feature] = st.selectbox("Card Type", card_types)
#     elif "Balance" in feature:
#         data[feature] = st.number_input(feature, min_value=0)
#     elif "Loan Amount" in feature:
#         data[feature] = st.number_input("Loan Amount", min_value=0, step=1000)
#     elif "Loan Term" in feature:
#         data[feature] = st.slider("Loan Term", 12, 60, 12)
#     elif "Date Of Account Opening" in feature:
#         data[feature] = st.date_input("Date of Account Opening", date(2020, 1, 1), min_value=date(2000, 1, 1), max_value=date.today())
#     elif "LastTransactionDate" in feature:
#         data[feature] = st.date_input("Last Transaction Date", date(2020, 1, 1), min_value=date(2000, 1, 1), max_value=date.today())
#     else:
#         data[feature] = st.number_input(feature)

# Convert categorical to numerical as needed
# input_df = pd.DataFrame([data])
# for col in input_df.select_dtypes(include='object').columns:
#     input_df[col] = pd.factorize(input_df[col])[0]

# if st.button("Predict"):
#     try:
#         pred_rf = rf_model.predict(input_df)[0]
#         pred_xgb = xgb_model.predict(input_df)[0]
#         label_map = {0: "Approved", 1: "Rejected", 2: "Closed"}

#         st.success(f"Random Forest: {label_map[pred_rf]}")
#         st.success(f"XGBoost: {label_map[pred_xgb]}")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")




# import streamlit as st
# import pandas as pd
# import pickle

# rf_model = pickle.load(open("rf_model.pkl", "rb"))
# xgb_model = pickle.load(open("xgb_model.pkl", "rb"))
# features = pickle.load(open("model_features.pkl", "rb"))

# st.set_page_config(page_title="Loan Predictor")
# st.title("Loan Status Predictor")

# # Load full dataframe to extract dropdown options
# df_full = pd.read_csv("df.csv")

# # Extract unique values for dropdowns
# age_bands = df_full['AgeBand'].unique().tolist()
# genders = df_full['Gender'].unique().tolist()
# cities = df_full['City'].unique().tolist()
# account_types = df_full['Account Type'].unique().tolist()
# loan_types = df_full['Loan Type'].unique().tolist()
# card_types = df_full['Card Type'].unique().tolist()

# # UI Input collection
# data = {}
# data = {}
# for feature in features:
#     if feature == "AgeBand":
#         data[feature] = st.selectbox("AgeBand", age_bands)
#     elif feature == "Gender":
#         data[feature] = st.selectbox("Gender", genders)
#     elif feature == "City":
#         data[feature] = st.selectbox("City", cities)
#     elif feature == "Account Type":
#         data[feature] = st.selectbox("Account Type", account_types)
#     elif feature == "Loan Type":
#         data[feature] = st.selectbox("Loan Type", loan_types)
#     elif feature == "Card Type":
#         data[feature] = st.selectbox("Card Type", card_types)
#     elif "Balance" in feature:
#         data[feature] = st.number_input(feature, min_value=0)
#     elif "LoanAmount" in feature:
#         data[feature] = st.number_input("Loan Amount", min_value=0, step=1000)
#     elif "TermYears" in feature:
#         data[feature] = st.slider("Loan Term (Years)", 12, 60, 12)    
           
#     else:
#         data[feature] = st.number_input(feature)

# # Convert categorical to numerical as needed
# input_df = pd.DataFrame([data])
# for col in input_df.select_dtypes(include='object').columns:
#     input_df[col] = pd.factorize(input_df[col])[0]

# if st.button("Predict"):
#     try:
#         pred_rf = rf_model.predict(input_df)[0]
#         pred_xgb = xgb_model.predict(input_df)[0]
#         label_map = {0: "Approved", 1: "Rejected", 2: "Closed"}

#         st.success(f"Random Forest: {label_map[pred_rf]}")
#         st.success(f"XGBoost: {label_map[pred_xgb]}")
#     except Exception as e:
#         st.error(f"Prediction failed: {e}")