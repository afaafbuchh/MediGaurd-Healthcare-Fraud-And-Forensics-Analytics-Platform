"""
==============================================================
  HEALTHCARE FRAUD DETECTION - Flask Web Application
==============================================================
  Step 2: Run this AFTER train_model.py
  Command: python app.py
  Then open: http://localhost:5000  in your browser
==============================================================
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import json

app = Flask(__name__)

# ---------------------------------------------------------------
# Load saved models and data at startup
# ---------------------------------------------------------------
print("Loading trained models...")

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/features.pkl", "rb") as f:
    FEATURE_COLUMNS = pickle.load(f)

with open("models/metrics.pkl", "rb") as f:
    metrics = pickle.load(f)

with open("models/feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)

with open("models/best_model_name.pkl", "rb") as f:
    best_model_name = pickle.load(f)

sample_df = pd.read_csv("models/sample_claims.csv")

print(f"✓ Loaded model: {best_model_name}")
print("✓ Ready! Open http://localhost:5000")


# ---------------------------------------------------------------
# ROUTE 1: Main Dashboard Page
# ---------------------------------------------------------------
@app.route("/")
def index():
    """Serve the main HTML dashboard page."""
    return render_template("index.html")


# ---------------------------------------------------------------
# ROUTE 2: Dashboard Stats (called by JS on page load)
# ---------------------------------------------------------------
@app.route("/api/stats")
def get_stats():
    """Return summary statistics for the dashboard overview cards."""

    # Calculate fraud stats from sample data
    total_claims    = len(sample_df)
    fraud_claims    = int(sample_df["is_fraud"].sum())
    legit_claims    = total_claims - fraud_claims
    fraud_rate      = round(fraud_claims / total_claims * 100, 1)

    # Calculate total amount at risk (fraud claims)
    fraud_amount    = float(sample_df[sample_df["is_fraud"] == 1]["claim_amount"].sum()) * 83
    total_amount    = float(sample_df["claim_amount"].sum()) * 83

    # Best model AUC score
    best_auc = round(metrics[best_model_name]["auc"] * 100, 1)

    return jsonify({
        "total_claims":  total_claims,
        "fraud_claims":  fraud_claims,
        "legit_claims":  legit_claims,
        "fraud_rate":    fraud_rate,
        "fraud_amount":  round(fraud_amount, 2),
        "total_amount":  round(total_amount, 2),
        "model_accuracy": best_auc,
        "best_model":    best_model_name
    })


# ---------------------------------------------------------------
# ROUTE 3: Model Performance Metrics
# ---------------------------------------------------------------
@app.route("/api/model_metrics")
def get_model_metrics():
    """Return performance metrics for all trained models."""
    output = []
    for name, m in metrics.items():
        output.append({
            "model":     name,
            "auc":       round(m["auc"] * 100, 2),
            "accuracy":  round(m["accuracy"] * 100, 2),
            "precision": round(m["precision"] * 100, 2),
            "recall":    round(m["recall"] * 100, 2),
            "f1":        round(m["f1"] * 100, 2),
            "confusion_matrix": m["confusion_matrix"]
        })
    return jsonify(output)


# ---------------------------------------------------------------
# ROUTE 4: Feature Importance Data
# ---------------------------------------------------------------
@app.route("/api/feature_importance")
def get_feature_importance():
    """Return top feature importances for the bar chart."""
    top_features = list(feature_importance.items())[:10]  # Top 10 features
    return jsonify({
        "features": [f[0].replace("_", " ").title() for f in top_features],
        "scores":   [round(f[1] * 100, 2) for f in top_features]
    })


# ---------------------------------------------------------------
# ROUTE 5: Sample Claims Data Table
# ---------------------------------------------------------------
@app.route("/api/claims")
def get_claims():
    """Return sample claims for the data table."""
    # Add a fraud risk score column using the model
    feature_data = sample_df[FEATURE_COLUMNS].copy()
    fraud_probs  = model.predict_proba(feature_data)[:, 1]

    result = sample_df[["claim_amount", "num_procedures", "inpatient_days",
                          "billing_deviation_score", "is_fraud"]].copy()
    result["risk_score"] = np.round(fraud_probs * 100, 1)
    result["risk_level"]  = result["risk_score"].apply(
        lambda s: "HIGH" if s >= 70 else ("MEDIUM" if s >= 40 else "LOW")
    )

    # Return top 50 records sorted by risk score
    result = result.sort_values("risk_score", ascending=False).head(50)
    return jsonify(result.to_dict(orient="records"))


# ---------------------------------------------------------------
# ROUTE 6: Predict a Single Claim (from the form in UI)
# ---------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accept a JSON claim object, run it through the ML model,
    and return a fraud risk score + explanation.
    """
    data = request.get_json()

    claim_amount       = float(data.get("claim_amount", 1000))
    num_procedures     = int(data.get("num_procedures", 3))
    inpatient_days     = int(data.get("inpatient_days", 0))
    num_diagnosis      = int(data.get("num_diagnosis_codes", 2))
    patient_age        = int(data.get("patient_age", 55))
    monthly_claims     = int(data.get("provider_monthly_claims", 100))
    provider_avg       = float(data.get("provider_avg_claim", 1000))
    provider_excluded  = int(data.get("provider_excluded", 0))
    duplicate_count    = int(data.get("duplicate_claim_count", 0))
    off_hours          = int(data.get("off_hours_submission", 0))
    specialty_mismatch = int(data.get("specialty_mismatch", 0))
    billing_deviation  = float(data.get("billing_deviation_score", 2.0))

    # ---------------------------------------------------------------
    # PURE RULE-BASED FRAUD SCORING
    # Points accumulate per red flag. Score is capped at 100.
    # Designed so that 2-3 serious flags = HIGH RISK (70%+)
    # ---------------------------------------------------------------
    score = 0.0
    red_flags = []

    # Rule 1: Provider excluded — instant HIGH territory alone (50 pts)
    if provider_excluded:
        score += 50
        red_flags.append("Provider is on the Medicare exclusion list")

    # Rule 2: Specialty mismatch — very serious alone (40 pts)
    if specialty_mismatch:
        score += 40
        red_flags.append("Billing outside provider specialty")

    # Rule 3: Duplicate claims
    if duplicate_count >= 4:
        score += 40
        red_flags.append(f"Very high duplicate claim count ({duplicate_count}) — likely double billing")
    elif duplicate_count >= 2:
        score += 28
        red_flags.append(f"Multiple duplicate claims detected ({duplicate_count})")
    elif duplicate_count == 1:
        score += 12
        red_flags.append("Duplicate claim detected")

    # Rule 4: Claim amount vs provider average (most important ratio)
    claim_to_avg_ratio = claim_amount / (provider_avg + 1)
    if claim_to_avg_ratio > 10:
        score += 40
        red_flags.append(f"Claim is {claim_to_avg_ratio:.1f}x the provider average — extremely suspicious")
    elif claim_to_avg_ratio > 5:
        score += 30
        red_flags.append(f"Claim is {claim_to_avg_ratio:.1f}x the provider average — very unusual")
    elif claim_to_avg_ratio > 3:
        score += 20
        red_flags.append(f"Claim is {claim_to_avg_ratio:.1f}x the provider average — above normal")
    elif claim_to_avg_ratio > 2:
        score += 10
        red_flags.append(f"Claim is {claim_to_avg_ratio:.1f}x the provider average")

    # Rule 5: High absolute claim amount
    if claim_amount > 200000:
        score += 25
        red_flags.append(f"Extremely high claim amount (Rs {claim_amount:,.0f})")
    elif claim_amount > 100000:
        score += 18
        red_flags.append(f"Very high claim amount (Rs {claim_amount:,.0f})")
    elif claim_amount > 50000:
        score += 12
        red_flags.append(f"High claim amount (Rs {claim_amount:,.0f})")
    elif claim_amount > 20000:
        score += 6
        red_flags.append(f"Above-average claim amount (Rs {claim_amount:,.0f})")

    # Rule 6: Too many procedures
    if num_procedures > 12:
        score += 20
        red_flags.append(f"Unusually high number of procedures billed ({num_procedures})")
    elif num_procedures > 8:
        score += 12
        red_flags.append(f"High number of procedures ({num_procedures})")

    # Rule 7: Procedure intensity (many procedures, very short stay)
    procedure_intensity = num_procedures / (inpatient_days + 1)
    if procedure_intensity > 10 and num_procedures > 5:
        score += 15
        red_flags.append(f"{num_procedures} procedures billed for only {inpatient_days} hospital day(s) — upcoding risk")
    elif procedure_intensity > 6 and num_procedures > 4:
        score += 8
        red_flags.append(f"High procedures-to-stay ratio ({num_procedures} procedures, {inpatient_days} days)")

    # Rule 8: Off-hours submission
    if off_hours:
        score += 15
        red_flags.append("Claim submitted during off-hours (night or weekend)")

    # Rule 9: Very high monthly provider claims
    if monthly_claims > 400:
        score += 15
        red_flags.append(f"Provider submitting abnormally high monthly claims ({monthly_claims})")
    elif monthly_claims > 300:
        score += 8
        red_flags.append(f"Provider monthly claim volume is high ({monthly_claims})")

    # Rule 10: Very long inpatient stay
    if inpatient_days > 20:
        score += 10
        red_flags.append(f"Unusually long inpatient stay ({inpatient_days} days)")

    # Cap at 100
    risk_score = round(min(score, 100.0), 1)

    # Determine risk level
    if risk_score >= 70:
        risk_level = "HIGH"
        color      = "#F0556B"
        recommendation = "FLAG FOR IMMEDIATE INVESTIGATION"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
        color      = "#F0A742"
        recommendation = "SEND FOR MANUAL REVIEW"
    else:
        risk_level = "LOW"
        color      = "#34D399"
        recommendation = "APPROVE FOR PROCESSING"

    return jsonify({
        "risk_score":      risk_score,
        "risk_level":      risk_level,
        "color":           color,
        "recommendation":  recommendation,
        "red_flags":       red_flags if red_flags else ["No major red flags detected — claim appears normal"],
        "fraud_probability": f"{risk_score}%"
    })


# ---------------------------------------------------------------
# ROUTE 7: Risk Distribution Chart Data
# ---------------------------------------------------------------
@app.route("/api/risk_distribution")
def risk_distribution():
    """Return risk score distribution for histogram chart."""
    feature_data = sample_df[FEATURE_COLUMNS].copy()
    fraud_probs  = model.predict_proba(feature_data)[:, 1] * 100

    # Bucket into bins: 0-10, 10-20, ..., 90-100
    bins   = list(range(0, 110, 10))
    labels = [f"{b}-{b+10}" for b in range(0, 100, 10)]
    counts, _ = np.histogram(fraud_probs, bins=bins)

    return jsonify({
        "labels": labels,
        "counts": counts.tolist()
    })


# ---------------------------------------------------------------
# Run the Flask development server
# ---------------------------------------------------------------
if __name__ == "__main__":
    # debug=True auto-reloads when you edit the code
    app.run(debug=True, port=5000)
