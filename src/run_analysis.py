"""Train and evaluate telecom churn models and save project artefacts."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"


def save_plot(name: str) -> None:
    """Save the current figure consistently and release its memory."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and create model features without leaking the target."""
    data = raw.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"]).drop_duplicates()
    data = data.replace({"No internet service": "No", "No phone service": "No"})

    data["tenure_group"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    )
    data["average_monthly_spend"] = data["TotalCharges"] / data["tenure"].replace(0, np.nan)
    data["average_monthly_spend"] = data["average_monthly_spend"].fillna(data["MonthlyCharges"])
    add_on_columns = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies",
    ]
    data["services_count"] = data[add_on_columns].eq("Yes").sum(axis=1)

    target = data.pop("Churn").eq("Yes").astype(int)
    features = data.drop(columns="customerID")
    return features, target


def create_eda(raw: pd.DataFrame, cleaned: pd.DataFrame, target: pd.Series) -> None:
    """Create the requested business-focused exploratory charts."""
    plot_data = cleaned.copy()
    plot_data["Churn"] = target.map({1: "Churned", 0: "Retained"})

    sns.set_theme(style="whitegrid", palette="deep")
    churn_by_contract = pd.crosstab(raw["Contract"], raw["Churn"], normalize="index")["Yes"].sort_values(ascending=False)
    churn_by_contract.mul(100).plot(kind="bar", color="#d95f02")
    plt.title("Churn Rate by Contract Type")
    plt.ylabel("Churn rate (%)")
    plt.xlabel("")
    save_plot("01_churn_by_contract.png")

    sns.histplot(data=plot_data, x="tenure", hue="Churn", bins=30, multiple="layer", stat="density", common_norm=False)
    plt.title("Tenure Distribution by Churn Status")
    save_plot("02_tenure_by_churn.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=plot_data, x="Churn", y="MonthlyCharges", ax=axes[0])
    axes[0].set_title("Monthly Charges by Churn")
    sns.boxplot(data=plot_data, x="Churn", y="TotalCharges", ax=axes[1])
    axes[1].set_title("Total Charges by Churn")
    save_plot("03_charge_boxplots.png")

    numeric = plot_data[["tenure", "MonthlyCharges", "TotalCharges", "average_monthly_spend", "services_count"]].copy()
    numeric["Churn"] = target.values
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Numeric Feature Correlations")
    save_plot("04_numeric_correlation_heatmap.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, column in zip(axes, ["InternetService", "PaymentMethod"]):
        rates = pd.crosstab(raw[column], raw["Churn"], normalize="index")["Yes"].sort_values(ascending=False)
        rates.mul(100).plot(kind="bar", ax=axis, color="#7570b3")
        axis.set_title(f"Churn Rate by {column}")
        axis.set_ylabel("Churn rate (%)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=35)
    save_plot("05_churn_by_service_and_payment.png")


def metric_row(name: str, model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float | str]:
    """Return all classification metrics on the shared untouched test set."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "Model": name,
        "Accuracy": accuracy_score(target, predictions),
        "Precision": precision_score(target, predictions),
        "Recall": recall_score(target, predictions),
        "F1": f1_score(target, predictions),
        "ROC-AUC": roc_auc_score(target, probabilities),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    features, target = prepare_data(raw)
    create_eda(raw, features, target)

    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )
    numeric_columns = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = train_features.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_columns),
    ])

    models = {
        "Logistic Regression": Pipeline([("prep", preprocessor), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Decision Tree": Pipeline([("prep", preprocessor), ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE))]),
        "Random Forest (baseline)": Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1))]),
    }
    for model in models.values():
        model.fit(train_features, train_target)

    search = GridSearchCV(
        Pipeline([("prep", preprocessor), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1))]),
        param_grid={
            "model__n_estimators": [100, 200],
            "model__max_depth": [8, None],
            "model__min_samples_leaf": [1, 3],
        },
        scoring="f1", cv=5, n_jobs=1, refit=True,
    )
    search.fit(train_features, train_target)
    models["Random Forest (tuned)"] = search.best_estimator_

    comparison = pd.DataFrame([metric_row(name, model, test_features, test_target) for name, model in models.items()])
    comparison = comparison.sort_values(["F1", "ROC-AUC"], ascending=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, test_features, test_target, name=name)
    plt.title("ROC Curves: Churn Models")
    save_plot("06_roc_curves.png")

    best_two = comparison["Model"].head(2).tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, name in zip(axes, best_two):
        ConfusionMatrixDisplay.from_estimator(models[name], test_features, test_target, ax=axis, cmap="Blues")
        axis.set_title(name)
    save_plot("07_confusion_matrices.png")

    feature_names = models["Random Forest (tuned)"].named_steps["prep"].get_feature_names_out()
    forest_importance = pd.Series(models["Random Forest (tuned)"].named_steps["model"].feature_importances_, index=feature_names)
    logistic_coefficients = pd.Series(models["Logistic Regression"].named_steps["model"].coef_[0], index=feature_names)
    drivers = pd.DataFrame({
        "Random Forest importance": forest_importance,
        "Logistic coefficient": logistic_coefficients,
    }).sort_values("Random Forest importance", ascending=False).head(15)
    drivers.to_csv(OUTPUT_DIR / "top_feature_drivers.csv")
    drivers.sort_values("Random Forest importance").plot(kind="barh", figsize=(9, 6))
    plt.title("Top Churn Drivers: Random Forest Importance")
    plt.xlabel("Importance / coefficient")
    save_plot("08_feature_importance.png")

    best = comparison.iloc[0]
    report = {
        "raw_rows": len(raw),
        "modeling_rows": len(features),
        "dropped_blank_total_charges_rows": int(raw["TotalCharges"].astype(str).str.strip().eq("").sum()),
        "best_grid_parameters": search.best_params_,
        "best_cv_f1": round(search.best_score_, 4),
        "recommended_model": best["Model"],
        "test_metrics": {key: round(float(best[key]), 4) for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]},
        "retention_actions": [
            "Prioritize month-to-month customers with high predicted churn probability for proactive outreach.",
            "Offer a contract conversion incentive or bundle discount to at-risk fibre and electronic-check customers.",
            "Proactively promote technical support and online-security bundles where service gaps signal churn risk.",
        ],
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(comparison.round(4).to_string(index=False))
    print(f"\nBest GridSearchCV parameters: {search.best_params_}")
    print(f"Best cross-validated F1: {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
