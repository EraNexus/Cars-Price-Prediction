"""
train_models.py — Train and compare Linear Regression, Decision Tree and
Random Forest on the CLEANED used car price data.

How to run:
    python train_models.py --data used_cars_sample_clean.csv
    python train_models.py --data used_cars_clean.csv --tune     # also tune the forest. Requires cleaned csv.

Cleaned data must have:
    - a numeric "price" column (The target. Can be changed with --target)
    - number columns (mileage, model_year, horsepower, 0/1 flags...) are
          filled if missing and scaled
    - text columns (brand, model, fuel_type, colors...) are "one-hot" encoded. Thank you google.
    So the cleaning can change column names or add features without
    touching this file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42


# 1. Load the cleaned data and split features by type
def load_data(path: str, target: str) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"No '{target}' column in {path}. Columns: {list(df.columns)}")

    df = df.dropna(subset=[target])
    y = df[target].astype(float)
    X = df.drop(columns=[target]).dropna(axis=1, how="all")  # ignore empty columns

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return X, y, numeric, categorical


# 2. Preprocessing shared by every model
def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # fill gaps + "was missing" flags
        ("scale", StandardScaler()),                                       # Required for Linear Regression
    ])
    categorical_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",   # categories only in the test set won't crash it
            min_frequency=5,                        # rare categories grouped together
            sparse_output=False,
        )),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, numeric),
        ("cat", categorical_steps, categorical),
    ])


# 3. The three models. Each learns log(price) and converts back to dollars.
def make_models(numeric, categorical) -> dict[str, TransformedTargetRegressor]:
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    return {
        name: TransformedTargetRegressor(
            regressor=Pipeline([
                ("prep", make_preprocessor(numeric, categorical)),
                ("model", reg),
            ]),
            func=np.log1p,
            inverse_func=np.expm1,
        )
        for name, reg in regressors.items()
    }


# 4. Metrics, tuning, feature importance
def score(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def tune_random_forest(model, X_train, y_train):
    """Small randomized search over Random Forest settings (--tune)."""
    grid = {
        "regressor__model__n_estimators": [200, 300, 500],
        "regressor__model__max_depth": [None, 15, 25],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__max_features": [1.0, "sqrt", 0.5],
    }
    search = RandomizedSearchCV(
        model, grid, n_iter=15, cv=3, scoring="neg_mean_absolute_error",
        n_jobs=-1, random_state=RANDOM_STATE, verbose=1,
    )
    search.fit(X_train, y_train)
    print("  Best RF params:", search.best_params_)
    return search.best_estimator_


def feature_importance_table(rf_model) -> pd.Series:
    pipe = rf_model.regressor_
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    imp.index = imp.index.str.replace(r"^(num|cat)__", "", regex=True)
    return imp.sort_values(ascending=False).rename("importance")


# Main
def main():
    parser = argparse.ArgumentParser(description="Train used car price models.")
    parser.add_argument("--data", required=True, help="Path to the cleaned CSV")
    parser.add_argument("--target", default="price", help="Name of the price column")
    parser.add_argument("--out", default="outputs", help="Folder for results")
    parser.add_argument("--tune", action="store_true", help="Tune the Random Forest (slower)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    X, y, numeric, categorical = load_data(args.data, args.target)
    print(f"Loaded {len(X):,} rows")
    print(f"  Numeric features ({len(numeric)}): {numeric}")
    print(f"  Text features    ({len(categorical)}): {categorical}")

    # 80/20 split, same for every model so the comparison is fair
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}\n")

    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted, preds = {}, {}, {}

    for name, model in make_models(numeric, categorical).items():
        print(f"Training {name}...")

        # 5-fold cross-validation on the training set for a more stable estimate
        cv = cross_validate(
            model, X_train, y_train, cv=kfold,
            scoring=["neg_mean_absolute_error", "r2"], n_jobs=-1,
        )

        if name == "Random Forest" and args.tune:
            model = tune_random_forest(model, X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        m = score(y_test, y_pred)
        m["Train_R2"] = r2_score(y_train, model.predict(X_train))  # big gap vs test = overfitting
        m["CV_MAE"] = -cv["test_neg_mean_absolute_error"].mean()
        m["CV_R2"] = cv["test_r2"].mean()
        m["CV_R2_std"] = cv["test_r2"].std()

        rows[name], fitted[name], preds[name] = m, model, y_pred
        print(f"  Test  MAE ${m['MAE']:,.0f} | RMSE ${m['RMSE']:,.0f} | R² {m['R2']:.3f}"
              f"  (train R² {m['Train_R2']:.3f}, CV R² {m['CV_R2']:.3f} ± {m['CV_R2_std']:.3f})")

    results = pd.DataFrame(rows).T
    results.to_csv(out / "results.csv", float_format="%.4f", index_label="model")

    best = results["MAE"].idxmin()
    print("\n=== Results (test set) ===")
    print(results[["MAE", "RMSE", "R2", "Train_R2", "CV_R2"]].round(3).to_string())
    print(f"\nBest model by MAE: {best}")

    # Each test car with its actual price and every model's prediction
    pred_df = X_test.copy()
    pred_df.insert(0, "actual_price", y_test)
    for name, p in preds.items():
        pred_df[f"pred_{name.lower().replace(' ', '_')}"] = p.round(0)
    pred_df.to_csv(out / "test_predictions.csv", index_label="row_id")

    importance = feature_importance_table(fitted["Random Forest"])
    importance.to_csv(out / "feature_importance.csv", index_label="feature")
    print("\nTop Random Forest features:")
    print(importance.head(10).round(4).to_string())

    joblib.dump(fitted[best], out / "best_model.joblib")
    print(f"\nSaved results and best_model.joblib to {out.resolve()}")


if __name__ == "__main__":
    main()