import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load the cleaned CSV file
df = pd.read_csv("df.csv")

# Convert target column to numbers
df['Loan Status'] = df['Loan Status'].map({'Approved': 0, 'Rejected': 1, 'Closed': 2})

# Convert text columns to numbers
for col in df.select_dtypes(include='object').columns:
    df[col] = pd.factorize(df[col])[0]

X = df.drop("Loan Status", axis=1)
y = df["Loan Status"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model = XGBClassifier(eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(xgb_model, open("xgb_model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("model_features.pkl", "wb"))