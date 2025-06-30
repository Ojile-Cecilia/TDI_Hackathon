# Loan Status Prediction App
![](loans.jpg)

An interactive Streamlit app that predicts loan outcomes—**Rejected**, **Approved**, or **Closed**—based on user-input features like credit score, loan amount, loan type, date of account opening, last transaction date.

## Why It Matters

- **Faster Lending Decisions**: Replaces manual underwriting, providing instant results.
- **Consistent Outcomes**: Reduces subjective bias with uniform, data-driven logic.
- **Risk Mitigation**: Identifies high-risk applicants early to cut down defaults.
- **Operational Efficiency**: Allows underwriters to focus on edge cases, not routine submissions.

## Technical Pipeline

1.	Data Ingestion & Preprocessing
Load historic loan records → clean, impute, normalize/encode.
2.	Feature Engineering
Generate ratios (e.g., debt-to-income), select key predictors.
3.	Train-Test Split
Use an 80/20 split to prevent overfitting.
4.	Model Training
Fit a classifier (Random Forest) on training data. Tune hyperparameters.
5.	Evaluation
Evaluate using accuracy, precision, recall, F1 scores, and confusion matrix.
6.	Deployment
Serialize the model with pickle and integrate into a Streamlit app (sidebar for inputs, Predict button, result display).


1. **Data Ingestion & Cleaning**  
   Load historical loan data, handle missing values, normalize numerical and encode categorical features.

2. **Feature Engineering**  
   Create strong predictors like debt-to-income ratio; remove low-information fields.

3. **Train–Test Split**  
   Use an 80/20 split to train and evaluate the model.

4. **Model Training**  
   Use Random Forest or XGBoost classifiers with hyperparameter tuning and class weighting.

5. **Evaluation**  
   Measure accuracy, precision, recall, F1-score, and inspect the confusion matrix.

6. **Deployment**  
   Serialize the trained model using `pickle` and integrate it into the Streamlit app.

7. **Inference & UI**  
   Sidebar inputs → click **Predict** → view predicted status + probabilities in real time.

---
![](loan_status_prediction.jpg)
## Challenges & Solutions

- **Poor Data Quality**: Missing and inconsistent entries required careful imputation and cleaning.
- **Class Imbalance**: Dominance of one class led to skewed performance; mitigated using SMOTE, class weighting, and resampling.
- **Feature Selection**: Identified low-variance features and engineered stronger signals to avoid overfitting.
- **Data Drift**: Recognized changing economic trends and implemented drift monitoring and retraining plans.
