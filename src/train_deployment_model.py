"""Train the recommended churn model and save it for the Streamlit app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from run_analysis import DATA_PATH, RANDOM_STATE, prepare_data


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_decision_tree.joblib"


def main() -> None:
    raw_data = pd.read_csv(DATA_PATH)
    features, target = prepare_data(raw_data)
    train_features, _, train_target, _ = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
    )

    numeric_columns = train_features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = train_features.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_columns),
    ])
    model = Pipeline([
        ("prep", preprocessor),
        ("model", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
    ])
    model.fit(train_features, train_target)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved deployment model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
