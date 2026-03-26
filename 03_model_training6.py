# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Training

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn shap mlflow -q

# COMMAND ----------

CATALOG           = "mlops_aidev_poc"
SILVER_TABLE      = f"{CATALOG}.mlops_silver.used_cars_features"
EXPERIMENT_NAME   = "/Users/me@company.com/used-car-price-prediction"
UC_MODEL_NAME     = f"{CATALOG}.mlops_silver.used_car_price_model"
TARGET_COL        = "price"
RANDOM_STATE      = 42
TEST_SIZE         = 0.2

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md ## 1. Load & cast in Spark

# COMMAND ----------

df_spark = spark.table(SILVER_TABLE)

for c, t in [("year", IntegerType()), ("car_age", IntegerType()),
             ("odometer", DoubleType()), ("price", DoubleType())]:
    df_spark = df_spark.withColumn(c, F.col(c).cast(t))

for c in ["manufacturer","model","condition","fuel",
          "transmission","drive","type","paint_color","state"]:
    df_spark = df_spark.withColumn(
        c, F.lower(F.trim(F.coalesce(F.col(c).cast(StringType()), F.lit("unknown"))))
    )

df = df_spark.toPandas()
print(f"Rows: {len(df):,}")

# COMMAND ----------

# MAGIC %md ## 2. Build pure float32 numpy matrix

# COMMAND ----------

CATEGORICAL_COLS = ["manufacturer","model","condition","fuel",
                    "transmission","drive","type","paint_color","state"]
NUMERIC_COLS     = ["year","car_age","odometer"]

for col in CATEGORICAL_COLS:
    df[col] = (df[col].astype(str).str.lower().str.strip()
               .replace({"nan":"unknown","none":"unknown","":"unknown"}))

for col in NUMERIC_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
df = df.dropna(subset=[TARGET_COL])

encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                         unknown_value=-1, dtype=np.float32)
encoder.fit(df[CATEGORICAL_COLS])

def to_matrix(df_in):
    cats = encoder.transform(df_in[CATEGORICAL_COLS])
    nums = df_in[NUMERIC_COLS].to_numpy(dtype=np.float32)
    return np.ascontiguousarray(np.hstack([cats, nums]), dtype=np.float32)

X = to_matrix(df)
y = df[TARGET_COL].to_numpy(dtype=np.float32)
feat_names = CATEGORICAL_COLS + NUMERIC_COLS

print(f"X: dtype={X.dtype}  shape={X.shape}")

# COMMAND ----------

# MAGIC %md ## 3. Train/test split

# COMMAND ----------

idx = np.arange(len(X))
idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

df_train = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[idx_train].reset_index(drop=True)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")

# COMMAND ----------

# MAGIC %md ## 4. Build DMatrix

# COMMAND ----------

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names,
                     feature_types=["float"] * len(feat_names))
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feat_names,
                     feature_types=["float"] * len(feat_names))

# COMMAND ----------

# MAGIC %md ## 5. Train

# COMMAND ----------

PARAMS = {
    "objective":        "reg:squarederror",
    "max_depth":        7,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "tree_method":      "hist",
    "seed":             RANDOM_STATE,
    "nthread":          -1,
}
N_ROUNDS = 500

with mlflow.start_run(run_name="xgboost-baseline") as run:
    RUN_ID = run.info.run_id
    print(f"Run ID: {RUN_ID}")

    mlflow.log_params({**PARAMS, "n_estimators": N_ROUNDS,
                       "train_size": len(X_train), "test_size": len(X_test)})

    booster = xgb.train(
        params          = PARAMS,
        dtrain          = dtrain,
        num_boost_round = N_ROUNDS,
        evals           = [(dtest, "validation")],
        verbose_eval    = 50,
    )

    y_pred = booster.predict(dtest)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mape   = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    mlflow.log_metrics({"mae": round(mae,2), "rmse": round(rmse,2),
                        "r2":  round(r2,4),  "mape": round(mape,2)})
    print(f"\nMAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.1f}%")

    # ── Feature importance ───────────────────────────────────────────────
    importances = booster.get_score(importance_type="gain")
    fi_df = (pd.DataFrame({"feature": list(importances.keys()),
                            "importance": list(importances.values())})
               .sort_values("importance", ascending=True))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi_df["feature"], fi_df["importance"], color="#4F8EF7")
    ax.set_title("Feature Importances (gain)")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close()

    # ── SHAP ─────────────────────────────────────────────────────────────
    n_shap   = min(2000, len(X_test))
    rng      = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_test), size=n_shap, replace=False)
    X_shap   = X_test[shap_idx]

    dshap = xgb.DMatrix(X_shap, feature_names=feat_names,
                        feature_types=["float"] * len(feat_names))
    shap_contribs = booster.predict(dshap, pred_contribs=True)
    shap_values   = shap_contribs[:, :-1]

    fig2, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feat_names, show=False)
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.close()

    # ── Build explicit MLflow signature ──────────────────────────────────
    # Unity Catalog REQUIRES a signature. We build it manually from the
    # feature list so MLflow never tries to infer it from a raw DataFrame
    # (which would fail because categorical columns are object dtype).
    input_schema = Schema([
        ColSpec("string",  "manufacturer"),
        ColSpec("string",  "model"),
        ColSpec("string",  "condition"),
        ColSpec("string",  "fuel"),
        ColSpec("string",  "transmission"),
        ColSpec("string",  "drive"),
        ColSpec("string",  "type"),
        ColSpec("string",  "paint_color"),
        ColSpec("string",  "state"),
        ColSpec("long",    "year"),
        ColSpec("long",    "car_age"),
        ColSpec("double",  "odometer"),
    ])
    output_schema = Schema([ColSpec("double", "predicted_price")])
    signature     = ModelSignature(inputs=input_schema, outputs=output_schema)

    # ── Input example: a small float32 numpy array (not a DataFrame) ─────
    # Passing the numpy array avoids XGBoost's pandas object-dtype check
    # that was causing the earlier signature-inference failure.
    input_example_np = X_train[:5]   # float32 ndarray, safe for DMatrix

    mlflow.xgboost.log_model(
        xgb_model             = booster,
        artifact_path         = "model",
        signature             = signature,          # explicit — no inference
        input_example         = input_example_np,   # float32 array — no object cols
        registered_model_name = UC_MODEL_NAME,
    )
    print(f"\n✅ Registered: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## 6. Residual Analysis

# COMMAND ----------

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.3, s=5, color="#4F8EF7")
axes[0].axhline(0, color="red", linewidth=1)
axes[0].set(xlabel="Predicted Price ($)", ylabel="Residual ($)", title="Residuals vs Predicted")
axes[1].scatter(y_test, y_pred, alpha=0.3, s=5, color="#4F8EF7")
lims = [max(0, min(float(y_test.min()), float(y_pred.min()))),
        max(float(y_test.max()), float(y_pred.max()))]
axes[1].plot(lims, lims, "r-", linewidth=1)
axes[1].set(xlabel="Actual Price ($)", ylabel="Predicted Price ($)", title="Actual vs Predicted")
plt.tight_layout()
display(fig)

# COMMAND ----------

dbutils.jobs.taskValues.set(key="run_id",    value=RUN_ID)
dbutils.jobs.taskValues.set(key="model_name", value=UC_MODEL_NAME)
print(f"\n✅ Training complete. Run ID: {RUN_ID}")
