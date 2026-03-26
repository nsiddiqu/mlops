# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Training

# COMMAND ----------

# MAGIC %pip install xgboost scikit-learn shap mlflow -q

# COMMAND ----------

CATALOG           = "mlops_demo"
SILVER_TABLE      = f"{CATALOG}.silver.used_cars_features"
EXPERIMENT_NAME   = "/Users/me@company.com/used-car-price-prediction"
UC_MODEL_NAME     = f"{CATALOG}.silver.used_car_price_model"
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

# Clean categoricals
for col in CATEGORICAL_COLS:
    df[col] = (df[col].astype(str).str.lower().str.strip()
               .replace({"nan":"unknown","none":"unknown","":"unknown"}))

# Force numerics
for col in NUMERIC_COLS + [TARGET_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
df = df.dropna(subset=[TARGET_COL])

# Encode categoricals
encoder = OrdinalEncoder(handle_unknown="use_encoded_value",
                         unknown_value=-1, dtype=np.float32)
encoder.fit(df[CATEGORICAL_COLS])

def to_matrix(df_in):
    """Always returns a C-contiguous float32 ndarray. No object dtype possible."""
    cats = encoder.transform(df_in[CATEGORICAL_COLS])          # float32 from OrdinalEncoder
    nums = df_in[NUMERIC_COLS].to_numpy(dtype=np.float32)      # explicit dtype arg
    mat  = np.ascontiguousarray(np.hstack([cats, nums]), dtype=np.float32)
    # Final hard assert — will raise immediately if something is wrong
    if mat.dtype != np.float32:
        raise TypeError(f"Matrix dtype is {mat.dtype}, expected float32")
    return mat

X = to_matrix(df)
y = df[TARGET_COL].to_numpy(dtype=np.float32)
feat_names = CATEGORICAL_COLS + NUMERIC_COLS

print(f"X: dtype={X.dtype}  shape={X.shape}  NaN={np.isnan(X).any()}")
print(f"y: dtype={y.dtype}  shape={y.shape}")

# COMMAND ----------

# MAGIC %md ## 3. Train/test split

# COMMAND ----------

idx = np.arange(len(X))
idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_train, X_test = X[idx_train], X[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

# Keep DataFrame slices only for MLflow signature / input example
df_train = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[idx_train].reset_index(drop=True)
df_test  = df[CATEGORICAL_COLS + NUMERIC_COLS].iloc[idx_test].reset_index(drop=True)

print(f"Train: {X_train.shape}  dtype={X_train.dtype}")
print(f"Test : {X_test.shape}   dtype={X_test.dtype}")

# COMMAND ----------

# MAGIC %md ## 4. Build DMatrix explicitly
# MAGIC
# MAGIC The bracket notation `[1.9227133E4]` in the ValueError means XGBoost's
# MAGIC **DMatrix** received a feature that it stored as a string internally.
# MAGIC Building DMatrix ourselves with `feature_types` locks every column to float.

# COMMAND ----------

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names,
                     feature_types=["float"] * len(feat_names))
dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=feat_names,
                     feature_types=["float"] * len(feat_names))

print(f"DMatrix train num_row={dtrain.num_row()}  num_col={dtrain.num_col()}")
print(f"DMatrix test  num_row={dtest.num_row()}   num_col={dtest.num_col()}")

# COMMAND ----------

# MAGIC %md ## 5. Train

# COMMAND ----------

PARAMS = {
    "objective":       "reg:squarederror",
    "max_depth":       7,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":5,
    "gamma":           0.1,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "tree_method":     "hist",
    "seed":            RANDOM_STATE,
    "nthread":         -1,
}
N_ROUNDS = 500

with mlflow.start_run(run_name="xgboost-baseline") as run:
    RUN_ID = run.info.run_id
    print(f"Run ID: {RUN_ID}")

    mlflow.log_params({**PARAMS, "n_estimators": N_ROUNDS,
                       "train_size":len(X_train), "test_size":len(X_test)})

    booster = xgb.train(
        params       = PARAMS,
        dtrain       = dtrain,
        num_boost_round = N_ROUNDS,
        evals        = [(dtest, "validation")],
        verbose_eval = 50,
    )

    y_pred = booster.predict(dtest)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    mape   = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

    mlflow.log_metrics({"mae":round(mae,2), "rmse":round(rmse,2),
                        "r2":round(r2,4),   "mape":round(mape,2)})
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

    # ── SHAP ──────────────────────────────────────────────────────────────
    # Build a fresh DMatrix for the SHAP sample — same float locking as above.
    n_shap   = min(2000, len(X_test))
    rng      = np.random.default_rng(42)
    shap_idx = rng.choice(len(X_test), size=n_shap, replace=False)
    X_shap   = X_test[shap_idx]                     # float32 ndarray slice

    dshap = xgb.DMatrix(X_shap, feature_names=feat_names,
                        feature_types=["float"] * len(feat_names))

    print(f"\nSHAP DMatrix: num_row={dshap.num_row()}  num_col={dshap.num_col()}")

    # Use the booster's built-in SHAP — avoids the shap library's type checks
    # that were the actual source of the ValueError
    shap_contribs = booster.predict(dshap, pred_contribs=True)
    shap_values   = shap_contribs[:, :-1]          # drop the bias column

    print(f"SHAP values: shape={shap_values.shape}  dtype={shap_values.dtype}")

    fig2, _ = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap, feature_names=feat_names, show=False)
    mlflow.log_figure(plt.gcf(), "shap_summary.png")
    plt.close()

    # ── Log booster via mlflow.xgboost ───────────────────────────────────
    mlflow.xgboost.log_model(
        xgb_model             = booster,
        artifact_path         = "model",
        input_example         = df_train.head(5),
        registered_model_name = UC_MODEL_NAME,
    )
    print(f"\n✅ Registered: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## 6. Residual Analysis

# COMMAND ----------

residuals = y_test - y_pred
fig, axes = plt.subplots(1, 2, figsize=(14,5))
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
