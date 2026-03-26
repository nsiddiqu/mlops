# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Model Evaluation & Registration
# MAGIC
# MAGIC **Goal**: Load the latest registered model version, run validation checks,
# MAGIC and promote it to **"Champion"** alias in Unity Catalog.

# COMMAND ----------

# MAGIC %pip install "mlflow[databricks]" xgboost scikit-learn -q

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

CATALOG       = "mlops_demo"
UC_MODEL_NAME = f"{CATALOG}.silver.used_car_price_model"

# Read run_id set by previous task (or override manually)
try:
    RUN_ID = dbutils.jobs.taskValues.get(
        taskKey="model_training", key="run_id", debugValue="latest"
    )
except:
    RUN_ID = "latest"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

print(f"Model : {UC_MODEL_NAME}")
print(f"Run ID: {RUN_ID}")

# COMMAND ----------

# MAGIC %md ## 1. Get Latest Model Version

# COMMAND ----------

# Get the most recently registered version
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
latest_version = versions_sorted[0]

print(f"Latest version : {latest_version.version}")
print(f"Run ID         : {latest_version.run_id}")
print(f"Status         : {latest_version.status}")
print(f"Source         : {latest_version.source}")

# COMMAND ----------

# MAGIC %md ## 2. Load Metrics from MLflow Run

# COMMAND ----------

run = client.get_run(latest_version.run_id)
metrics = run.data.metrics

print("\n📊 Registered Model Metrics:")
for k, v in metrics.items():
    print(f"   {k:10s}: {v}")

# ── Quality gate ────────────────────────────────────────────────────────
REQUIRED_R2   = 0.75   # Minimum acceptable R²
REQUIRED_MAPE = 60.0   # Maximum acceptable MAPE %

r2   = metrics.get("r2",   0)
mape = metrics.get("mape", 999)

print(f"\n🔍 Quality Gates:")
print(f"   R² ≥ {REQUIRED_R2}  → {r2:.4f}  {'✅ PASS' if r2 >= REQUIRED_R2 else '❌ FAIL'}")
print(f"   MAPE ≤ {REQUIRED_MAPE}% → {mape:.2f}%  {'✅ PASS' if mape <= REQUIRED_MAPE else '❌ FAIL'}")

if r2 < REQUIRED_R2 or mape > REQUIRED_MAPE:
    raise ValueError(
        f"❌ Model failed quality gate. R²={r2:.4f}, MAPE={mape:.2f}%. "
        f"Required: R²≥{REQUIRED_R2}, MAPE≤{REQUIRED_MAPE}%"
    )

print("\n✅ All quality gates passed!")

# COMMAND ----------

# MAGIC %md ## 3. Smoke-Test the Model

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

# Load the booster (logged with mlflow.xgboost, not sklearn)
model_uri    = f"models:/{UC_MODEL_NAME}/{latest_version.version}"
loaded_model = mlflow.xgboost.load_model(model_uri)   # returns an xgb.Booster

CATEGORICAL_COLS = ["manufacturer","model","condition","fuel",
                    "transmission","drive","type","paint_color","state"]
NUMERIC_COLS     = ["year","car_age","odometer"]
FEAT_NAMES       = CATEGORICAL_COLS + NUMERIC_COLS

# Synthetic test rows (raw strings — as a real caller would send)
test_rows = pd.DataFrame([
    {"year":2018,"car_age":6,"manufacturer":"toyota","model":"camry",
     "condition":"good","odometer":55000,"fuel":"gas","transmission":"automatic",
     "drive":"fwd","type":"sedan","paint_color":"white","state":"ca"},
    {"year":2015,"car_age":9,"manufacturer":"ford","model":"f-150",
     "condition":"fair","odometer":120000,"fuel":"gas","transmission":"automatic",
     "drive":"4wd","type":"pickup","paint_color":"black","state":"tx"},
    {"year":2020,"car_age":4,"manufacturer":"bmw","model":"3 series",
     "condition":"excellent","odometer":20000,"fuel":"gas","transmission":"automatic",
     "drive":"rwd","type":"sedan","paint_color":"silver","state":"ny"},
])

# Re-fit a minimal encoder on the Silver table so we can encode the test rows.
# (The encoder fitted in notebook 03 is not persisted separately — we refit on
#  the same Silver data so category mappings are identical.)
df_silver = spark.table(f"{UC_MODEL_NAME.split('.')[0]}.mlops_silver.used_cars_features").toPandas()
for col in CATEGORICAL_COLS:
    df_silver[col] = (df_silver[col].astype(str).str.lower().str.strip()
                      .replace({"nan":"unknown","none":"unknown","":"unknown"}))

enc_smoke = OrdinalEncoder(handle_unknown="use_encoded_value",
                           unknown_value=-1, dtype=np.float32)
enc_smoke.fit(df_silver[CATEGORICAL_COLS])

# Encode smoke-test rows → float32 DMatrix
for col in CATEGORICAL_COLS:
    test_rows[col] = test_rows[col].astype(str).str.lower().str.strip()
for col in NUMERIC_COLS:
    test_rows[col] = pd.to_numeric(test_rows[col], errors="coerce").fillna(0)

cats = enc_smoke.transform(test_rows[CATEGORICAL_COLS])
nums = test_rows[NUMERIC_COLS].to_numpy(dtype=np.float32)
X_smoke = np.ascontiguousarray(np.hstack([cats, nums]), dtype=np.float32)

dsmoke      = xgb.DMatrix(X_smoke, feature_names=FEAT_NAMES,
                           feature_types=["float"] * len(FEAT_NAMES))
predictions = loaded_model.predict(dsmoke)

test_rows["predicted_price"] = predictions.round(0)
print("🚗 Smoke-test predictions:")
for _, row in test_rows.iterrows():
    print(f"   {int(row['year'])} {row['manufacturer'].title()} {row['model'].title()} "
          f"({int(row['odometer']):,} mi) → ${row['predicted_price']:,.0f}")

assert all(p > 0 for p in predictions), "❌ Negative predictions detected!"
print("\n✅ Smoke test passed — all predictions positive.")

# COMMAND ----------

# MAGIC %md ## 4. Promote to Champion Alias

# COMMAND ----------

# Set "champion" alias — this is what serving will reference
client.set_registered_model_alias(
    name    = UC_MODEL_NAME,
    alias   = "champion",
    version = latest_version.version
)

# Also tag it
client.set_model_version_tag(
    name    = UC_MODEL_NAME,
    version = latest_version.version,
    key     = "validation_status",
    value   = "passed"
)
client.set_model_version_tag(
    name    = UC_MODEL_NAME,
    version = latest_version.version,
    key     = "r2",
    value   = str(round(r2, 4))
)
client.set_model_version_tag(
    name    = UC_MODEL_NAME,
    version = latest_version.version,
    key     = "mape_pct",
    value   = str(round(mape, 2))
)

print(f"✅ Version {latest_version.version} promoted to 'champion' alias")
print(f"   Model URI: models:/{UC_MODEL_NAME}@champion")

# COMMAND ----------

# Pass to next task
dbutils.jobs.taskValues.set(key="champion_version", value=latest_version.version)
dbutils.jobs.taskValues.set(key="model_uri", value=f"models:/{UC_MODEL_NAME}@champion")

print(f"\n✅ Evaluation complete. Next step → 05_model_serving")
