"""
==============================================================
  HEALTHCARE FRAUD DETECTION - Model Training Script
==============================================================
  Step 1: Run this file FIRST to train and save your ML models.
  Command: python train_model.py
==============================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  Healthcare Fraud Detection - Training Pipeline")
print("=" * 60)

# ---------------------------------------------------------------
# STEP 1: Generate Synthetic Medicare-style Fraud Dataset
# ---------------------------------------------------------------
# We simulate the public Medicare fraud dataset structure.
# Real datasets: https://www.cms.gov/Research-Statistics-Data-and-Systems
# (LEIE - List of Excluded Individuals/Entities is publicly available)

print("\n[1/6] Generating synthetic Medicare claims dataset...")

np.random.seed(42)
n_samples = 5000  # 5000 claim records

# --- Provider features ---
provider_ids = [f"PRV{str(i).zfill(5)}" for i in range(1, 501)]  # 500 unique providers

# --- Generate raw claim data ---
data = {
    "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, n_samples + 1)],
    "provider_id": np.random.choice(provider_ids, n_samples),

    # How many procedures were billed per claim
    "num_procedures": np.random.randint(1, 15, n_samples),

    # Total amount billed to insurance
    "claim_amount": np.round(np.random.lognormal(mean=7.5, sigma=1.2, size=n_samples), 2),

    # How many days patient was in hospital (0 = outpatient)
    "inpatient_days": np.random.randint(0, 30, n_samples),

    # Number of unique diagnosis codes on claim
    "num_diagnosis_codes": np.random.randint(1, 10, n_samples),

    # Patient age
    "patient_age": np.random.randint(18, 95, n_samples),

    # Number of claims this provider submitted this month
    "provider_monthly_claims": np.random.randint(10, 500, n_samples),

    # Average claim amount for this provider
    "provider_avg_claim": np.round(np.random.lognormal(mean=7.0, sigma=0.8, size=n_samples), 2),

    # Is this provider on the Medicare exclusion list?
    "provider_excluded": np.random.choice([0, 1], n_samples, p=[0.92, 0.08]),

    # Number of duplicate/similar claims submitted
    "duplicate_claim_count": np.random.randint(0, 5, n_samples),

    # Was the claim submitted during off-hours (nights/weekends)?
    "off_hours_submission": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),

    # Specialty mismatch: did provider bill outside their specialty?
    "specialty_mismatch": np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),

    # How much this claim deviates from the provider's normal billing
    "billing_deviation_score": np.round(np.random.uniform(0, 10, n_samples), 2),
}

df = pd.DataFrame(data)

# ---------------------------------------------------------------
# STEP 2: Create Fraud Labels (Target Variable)
# ---------------------------------------------------------------
# In real life these labels come from confirmed fraud investigations.
# Here we use domain rules to simulate realistic fraud patterns.

print("[2/6] Creating fraud labels using domain rules...")

# Base fraud probability for each claim (starts low)
fraud_prob = np.zeros(n_samples)

# Rule 1: Excluded providers are very high risk
fraud_prob += df["provider_excluded"] * 0.5

# Rule 2: Very high claim amounts are suspicious
fraud_prob += (df["claim_amount"] > df["claim_amount"].quantile(0.90)).astype(int) * 0.2

# Rule 3: Many procedures in one claim = upcoding risk
fraud_prob += (df["num_procedures"] > 10).astype(int) * 0.15

# Rule 4: Billing outside specialty is a red flag
fraud_prob += df["specialty_mismatch"] * 0.2

# Rule 5: High billing deviation from provider's norm
fraud_prob += (df["billing_deviation_score"] > 7).astype(int) * 0.15

# Rule 6: Duplicate claims
fraud_prob += (df["duplicate_claim_count"] > 2).astype(int) * 0.1

# Add some randomness to make it realistic
fraud_prob += np.random.uniform(0, 0.1, n_samples)

# Clip to [0, 1] range and assign binary labels
fraud_prob = np.clip(fraud_prob, 0, 1)
df["is_fraud"] = (fraud_prob > 0.55).astype(int)

fraud_rate = df["is_fraud"].mean() * 100
print(f"   → Dataset created: {n_samples} claims, {fraud_rate:.1f}% fraud rate")
print(f"   → Fraud cases: {df['is_fraud'].sum()} | Legitimate: {(df['is_fraud']==0).sum()}")

# ---------------------------------------------------------------
# STEP 3: Feature Engineering
# ---------------------------------------------------------------
print("\n[3/6] Engineering features...")

# Ratio of claim amount vs provider's average (anomaly indicator)
df["claim_to_avg_ratio"] = df["claim_amount"] / (df["provider_avg_claim"] + 1)

# High procedure density (many procedures for short stay)
df["procedure_intensity"] = df["num_procedures"] / (df["inpatient_days"] + 1)

# Provider risk score (combines multiple risk signals)
df["provider_risk_score"] = (
    df["provider_excluded"] * 3 +
    df["specialty_mismatch"] * 2 +
    (df["provider_monthly_claims"] > 300).astype(int)
)

print("   → New features created: claim_to_avg_ratio, procedure_intensity, provider_risk_score")

# ---------------------------------------------------------------
# STEP 4: Prepare Data for ML
# ---------------------------------------------------------------
print("\n[4/6] Preparing data for machine learning...")

# These are the features our ML models will learn from
FEATURE_COLUMNS = [
    "num_procedures",
    "claim_amount",
    "inpatient_days",
    "num_diagnosis_codes",
    "patient_age",
    "provider_monthly_claims",
    "provider_avg_claim",
    "provider_excluded",
    "duplicate_claim_count",
    "off_hours_submission",
    "specialty_mismatch",
    "billing_deviation_score",
    "claim_to_avg_ratio",
    "procedure_intensity",
    "provider_risk_score",
]

X = df[FEATURE_COLUMNS]  # Features (inputs)
y = df["is_fraud"]        # Target (output: 0=legit, 1=fraud)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize/scale the features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"   → Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ---------------------------------------------------------------
# STEP 5: Train Multiple ML Models
# ---------------------------------------------------------------
print("\n[5/6] Training ML models (this may take a minute)...")

models = {
    # Random Forest: An ensemble of decision trees — great for fraud detection
    "Random Forest": RandomForestClassifier(
        n_estimators=150,    # Number of trees
        max_depth=10,        # Max depth of each tree
        class_weight="balanced",  # Handle class imbalance
        random_state=42,
        n_jobs=-1            # Use all CPU cores
    ),

    # Gradient Boosting: Builds trees sequentially to fix mistakes
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ),

    # Logistic Regression: Simple but interpretable baseline
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")

    # Use scaled data for Logistic Regression, raw for tree models
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    results[name] = {
        "model": model,
        "auc": auc,
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }

    print(f"   ✓ AUC: {auc:.4f} | Accuracy: {report['accuracy']:.4f} | "
          f"Fraud Precision: {report['1']['precision']:.4f} | Recall: {report['1']['recall']:.4f}")

# Pick the best model by AUC score
best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_model_name]["model"]
print(f"\n   🏆 Best model: {best_model_name} (AUC = {results[best_model_name]['auc']:.4f})")

# Feature importance from Random Forest (most interpretable)
rf_model = results["Random Forest"]["model"]
feature_importance = dict(zip(FEATURE_COLUMNS, rf_model.feature_importances_))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# ---------------------------------------------------------------
# STEP 6: Save Everything to Disk
# ---------------------------------------------------------------
print("\n[6/6] Saving models and data...")

os.makedirs("models", exist_ok=True)

# Save the best ML model
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save the scaler (needed for Logistic Regression predictions)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save the feature list (must match what app.py uses)
with open("models/features.pkl", "wb") as f:
    pickle.dump(FEATURE_COLUMNS, f)

# Save model performance metrics for the dashboard
with open("models/metrics.pkl", "wb") as f:
    metrics_to_save = {
        name: {k: v for k, v in r.items() if k != "model"}
        for name, r in results.items()
    }
    pickle.dump(metrics_to_save, f)

# Save feature importances
with open("models/feature_importance.pkl", "wb") as f:
    pickle.dump(feature_importance, f)

# Save a sample of the dataset for the dashboard's data explorer
df.drop(columns=["claim_id", "provider_id"]).sample(200, random_state=42).to_csv(
    "models/sample_claims.csv", index=False
)

# Save the best model name
with open("models/best_model_name.pkl", "wb") as f:
    pickle.dump(best_model_name, f)

print("\n" + "=" * 60)
print("  ✅ Training Complete! Files saved to /models folder:")
print("     - best_model.pkl        (trained ML model)")
print("     - scaler.pkl            (data normalizer)")
print("     - features.pkl          (feature names)")
print("     - metrics.pkl           (model performance stats)")
print("     - feature_importance.pkl (which features matter most)")
print("     - sample_claims.csv     (sample data for dashboard)")
print("=" * 60)
print("\n  ➡️  NEXT STEP: Run  python app.py  to start the web app!")
print("=" * 60)
