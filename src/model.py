"""
UPI Fraud Detector – ML Model Training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings, pickle

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import LabelEncoder, StandardScaler
from sklearn.linear_model       import LogisticRegression
from sklearn.ensemble           import RandomForestClassifier, IsolationForest
from sklearn.metrics            import (classification_report, confusion_matrix,
                                        roc_auc_score, roc_curve,
                                        precision_score, recall_score,
                                        f1_score, accuracy_score)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "upi_transactions.csv"
SRC_DIR    = BASE_DIR / "src"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor"  : "#444466", "axes.labelcolor": "#ccccee",
    "xtick.color"     : "#aaaacc", "ytick.color"    : "#aaaacc",
    "text.color"      : "#eeeeff", "grid.color"     : "#2a2a4a",
    "grid.linestyle"  : "--",      "grid.alpha"     : 0.5,
})
FRAUD_COLOR = "#ff4d6d"; NORMAL_COLOR = "#00c9a7"; ACCENT = "#7b5ea7"

# ── Load & encode ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print(f"✅ Loaded {len(df)} transactions")

cat_cols = ["sender_city","receiver_city","sender_bank","receiver_bank",
            "merchant_category","transaction_type","device_type","day_of_week"]
le = LabelEncoder()
for c in cat_cols:
    df[c] = le.fit_transform(df[c].astype(str))

FEATURES = ["amount","hour","month","is_new_recipient","failed_attempts",
            "is_odd_hour","cross_city","sender_city","receiver_city",
            "sender_bank","receiver_bank","merchant_category",
            "transaction_type","device_type"]
TARGET = "is_fraud"

X = df[FEATURES]; y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ─────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr.fit(X_train_sc, y_train)
lr_pred  = lr.predict(X_test_sc)
lr_proba = lr.predict_proba(X_test_sc)[:, 1]

rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                             max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso.fit(X_train)
iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)

print("\n" + "="*55)
print("  LOGISTIC REGRESSION")
print("="*55)
print(classification_report(y_test, lr_pred, target_names=["Normal","Fraud"]))
print(f"  ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

print("\n" + "="*55)
print("  RANDOM FOREST  ← Best Model")
print("="*55)
print(classification_report(y_test, rf_pred, target_names=["Normal","Fraud"]))
print(f"  ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

print("\n" + "="*55)
print("  ISOLATION FOREST (Unsupervised)")
print("="*55)
print(classification_report(y_test, iso_pred, target_names=["Normal","Fraud"]))

# ── Plot 4 – Model Performance ─────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Model Performance Evaluation", fontsize=16, fontweight="bold", y=1.02)

for ax, pred, title in zip(axes[:2], [lr_pred, rf_pred],
                            ["Logistic Regression", "Random Forest"]):
    cm = confusion_matrix(y_test, pred)
    im = ax.imshow(cm, cmap="RdYlGn", aspect="auto")
    ax.set_title(f"Confusion Matrix\n{title}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Fraud"]); ax.set_yticklabels(["Normal","Fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i,j] < cm.max()/2 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

ax = axes[2]
for proba, label, color in [(lr_proba,"Logistic Regression",ACCENT),(rf_proba,"Random Forest",NORMAL_COLOR)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    ax.plot(fpr, tpr, color=color, linewidth=2.5, label=f"{label} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], color="gray", linestyle="--", alpha=0.5)
ax.set_title("ROC Curve Comparison", fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(facecolor="#1a1a2e", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_model_performance.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("\n✅ Plot 4 saved")

# ── Plot 5 – Feature Importance ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0f0f1a"); ax.set_facecolor("#1a1a2e")
importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
colors = [FRAUD_COLOR if v > importances.median() else ACCENT for v in importances.values]
importances.plot(kind="barh", ax=ax, color=colors, edgecolor="#0f0f1a")
ax.set_title("Feature Importance – Random Forest", fontsize=15, fontweight="bold")
ax.set_xlabel("Importance Score")
for i, val in enumerate(importances.values):
    ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
ax.legend(handles=[mpatches.Patch(color=FRAUD_COLOR, label="High"),
                   mpatches.Patch(color=ACCENT,      label="Lower")], facecolor="#1a1a2e")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_feature_importance.png", dpi=150, bbox_inches="tight", facecolor="#0f0f1a")
plt.close()
print("✅ Plot 5 saved")

# ── Save model ─────────────────────────────────────────────
with open(SRC_DIR / "rf_model.pkl",  "wb") as f: pickle.dump(rf,       f)
with open(SRC_DIR / "scaler.pkl",    "wb") as f: pickle.dump(scaler,   f)
with open(SRC_DIR / "features.pkl",  "wb") as f: pickle.dump(FEATURES, f)
print("\n✅ Model saved to src/")

# ── Save metrics ───────────────────────────────────────────
metrics = {
    "Model"    : ["Logistic Regression","Random Forest","Isolation Forest"],
    "Accuracy" : [accuracy_score(y_test,lr_pred), accuracy_score(y_test,rf_pred), accuracy_score(y_test,iso_pred)],
    "Precision": [precision_score(y_test,lr_pred),precision_score(y_test,rf_pred),precision_score(y_test,iso_pred)],
    "Recall"   : [recall_score(y_test,lr_pred),   recall_score(y_test,rf_pred),   recall_score(y_test,iso_pred)],
    "F1 Score" : [f1_score(y_test,lr_pred),       f1_score(y_test,rf_pred),       f1_score(y_test,iso_pred)],
    "ROC-AUC"  : [roc_auc_score(y_test,lr_proba), roc_auc_score(y_test,rf_proba), "N/A"],
}
pd.DataFrame(metrics).round(4).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
print("✅ Metrics saved to outputs/model_metrics.csv")
print("\n🎉 All done!")
