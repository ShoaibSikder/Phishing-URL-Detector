import os
import json
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
from train_model import URLFeatureExtractor
from pathlib import Path

# ------------------------------------------
# Setup paths
# ------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT.parent / "dataset" / "phishing_data.csv"
METRIC_DIR = ROOT / "metrics"
os.makedirs(METRIC_DIR, exist_ok=True)

# ------------------------------------------
# Load model and pipelines
# ------------------------------------------
saved = joblib.load(ROOT / "model.pkl")
model = saved["model"]
num_pipe = saved["num_pipe"]
bool_pipe = saved["bool_pipe"]
tfidf = saved["tfidf"]
numeric_features = saved["numeric_features"]
bool_features = saved["bool_features"]

print("[INFO] Model components loaded successfully.")

# ------------------------------------------
# Load dataset
# ------------------------------------------
df = pd.read_csv(DATASET_PATH).dropna(subset=["url", "label"])
df["url"] = df["url"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

print("[INFO] Dataset loaded successfully:", df.shape)

# ------------------------------------------
# Feature extraction
# ------------------------------------------
url_features = URLFeatureExtractor().fit_transform(df["url"])

if isinstance(url_features, pd.DataFrame):
    if "url" in url_features.columns:
        url_features = url_features.drop(columns=["url"])
    df = pd.concat([df.reset_index(drop=True), url_features.reset_index(drop=True)], axis=1)
elif isinstance(url_features, pd.Series):
    df = pd.concat([df.reset_index(drop=True), url_features.rename("url_feat").reset_index(drop=True)], axis=1)
else:
    raise TypeError(f"Unexpected type from URLFeatureExtractor: {type(url_features)}")

df["url_text"] = df["url"].astype(str)

# ------------------------------------------
# Prepare features for model
# ------------------------------------------
X_num = df[numeric_features]
X_bool = df[bool_features]
X_text = df["url_text"]
y = df["label"]

X_num_scaled = num_pipe.transform(X_num)
X_bool_enc = bool_pipe.transform(X_bool)
X_text_tfidf = tfidf.transform(X_text)

X_full = sparse.hstack([
    sparse.csr_matrix(X_num_scaled),
    sparse.csr_matrix(X_bool_enc),
    X_text_tfidf
]).tocsr()

print("[INFO] Feature matrix built:", X_full.shape)

# ------------------------------------------
# Predict
# ------------------------------------------
y_pred = model.predict(X_full)
y_prob = model.predict_proba(X_full)[:, 1]

print("[INFO] Predictions completed.")

# ------------------------------------------
# 1. Confusion Matrix
# ------------------------------------------
cm = confusion_matrix(y, y_pred).tolist()
with open(METRIC_DIR / "confusion_matrix.json", "w") as f:
    json.dump({"confusion_matrix": cm}, f)

# ------------------------------------------
# 2. ROC Curve
# ------------------------------------------
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
with open(METRIC_DIR / "roc.json", "w") as f:
    json.dump(roc_data, f)

# ------------------------------------------
# 3. Precision–Recall Curve
# ------------------------------------------
precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
pr_data = {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": pr_thresholds.tolist()}
with open(METRIC_DIR / "pr.json", "w") as f:
    json.dump(pr_data, f)

# ------------------------------------------
# 4. Accuracy
# ------------------------------------------
accuracy = float(accuracy_score(y, y_pred))
with open(METRIC_DIR / "accuracy.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

# ------------------------------------------
# 5. Feature Importance (Top Features)
# ------------------------------------------
feature_names = numeric_features + bool_features
importances = model.feature_importances_[:len(feature_names)]  # only numeric/bool
top_idx = np.argsort(importances)[-len(feature_names):]
top_features = [{"feature": feature_names[i], "importance": float(importances[i])} for i in top_idx]

with open(METRIC_DIR / "feature_importance.json", "w") as f:
    json.dump({"top_features": top_features}, f)

print("\n[✓] All metrics generated successfully!")
print(f"[✓] Files saved inside: {METRIC_DIR}/")
