# Telecom Customer Churn Prediction & Retention Strategy

This project predicts whether a telecom customer will churn and turns the results into practical retention actions. It uses the IBM Telco Customer Churn dataset originally published through [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## What is included

- Cleaning for `TotalCharges`, duplicate records, and inconsistent no-service labels.
- Feature engineering for `tenure_group`, `average_monthly_spend`, and `services_count`.
- Exploratory charts covering contract, tenure, charges, correlations, internet service, and payment method.
- Logistic Regression, capped Decision Tree, baseline Random Forest, and tuned Random Forest.
- A 5-fold `GridSearchCV` search of Random Forest `n_estimators`, `max_depth`, and `min_samples_leaf`, scored with F1.
- A shared 80/20 stratified test-set comparison using accuracy, precision, recall, F1, and ROC-AUC.

## Run

```powershell
python -m pip install -r requirements.txt
python src/run_analysis.py
```

## Streamlit deployment

Train and save the selected deployment model, then start the local app:

```powershell
python src/train_deployment_model.py
streamlit run app.py
```

For Streamlit Community Cloud, upload this project to GitHub, set the app entry point to `app.py`, and deploy. The `models/churn_decision_tree.joblib` file must be included in the repository; generate it with the training command before pushing.

The runnable notebook is [notebooks/telecom_churn_analysis.ipynb](notebooks/telecom_churn_analysis.ipynb). Run the script first (or execute its code cell) to refresh all artifacts.

## Outputs

The script writes charts, a model comparison table, top feature drivers, and a JSON business summary to `outputs/`.

## Deployment recommendation

Deploy the capped Decision Tree for the initial retention workflow: it achieved the strongest test F1 (0.5879) and recall (0.5856), catching more true churners than the alternatives. Logistic Regression has the best ROC-AUC (0.8351), so it remains a strong option when probability ranking matters more than a default classification threshold. Retention teams should rank customers by churn probability and concentrate first on month-to-month customers, then present conversion or bundled-support offers. The final evidence and GridSearchCV parameters are saved in `outputs/analysis_summary.json` and `outputs/model_comparison.csv`.

## Project structure

```text
data/        Source Telco CSV
notebooks/   Notebook launcher and results viewer
src/         Reproducible analysis code
outputs/     Generated charts and evaluation tables
```
