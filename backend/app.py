from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib, pandas as pd, numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import math

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
DATASET_PATH = ROOT.parent / "dataset" / "phishing_data.csv"

# -------------------------------
# Request model
# -------------------------------
class URLRequest(BaseModel):
    url: str

# -------------------------------
# Load model
# -------------------------------
saved = joblib.load(MODEL_PATH)
model = saved['model']
num_pipe = saved['num_pipe']
bool_pipe = saved['bool_pipe']
tfidf = saved['tfidf']
numeric_features = saved['numeric_features']
bool_features = saved['bool_features']

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="Phishing URL Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Helpers
# -------------------------------
def sanitize_list(lst):
    return [0.0 if (math.isnan(x) or math.isinf(x)) else float(x) for x in lst]

def extract_features(url: str):
    numeric_df = pd.DataFrame([{
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_digits': sum(c.isdigit() for c in url)
    }])[numeric_features]

    bool_df = pd.DataFrame([{
        'has_https': 1 if url.startswith('https') else 0,
        'local_keyword_flag': 0
    }])[bool_features]

    return numeric_df, bool_df, [url.lower()]

# -------------------------------
# Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"status": "ok", "msg": "Phishing Detector API running"}

# -------------------------------
# Model accuracy
# -------------------------------
@app.get("/metrics")
def metrics():
    return {"test_accuracy": round(saved.get('test_accuracy', 0) * 100, 2)}

# -------------------------------
# URL prediction
# -------------------------------
@app.post("/predict")
def predict(req: URLRequest):
    try:
        num_df, bool_df, text = extract_features(req.url)
        X_num = num_pipe.transform(num_df)
        X_bool = bool_pipe.transform(bool_df)
        X_text = tfidf.transform(text)
        X_full = sparse.hstack([sparse.csr_matrix(X_num),
                                sparse.csr_matrix(X_bool),
                                X_text]).tocsr()

        pred = int(model.predict(X_full)[0])
        prob = model.predict_proba(X_full)[0].tolist()

        return {
            'url': req.url,
            'prediction': pred,
            'probabilities': {'legitimate': prob[0], 'phishing': prob[1]}
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# Compute metrics dynamically
# -------------------------------
def compute_metrics():
    try:
        df = pd.read_csv(DATASET_PATH).dropna(subset=['url','label'])
        df['url'] = df['url'].astype(str).str.strip()
        df['label'] = df['label'].astype(int)

        X_num = df[numeric_features]
        X_bool = df[bool_features]
        X_text = df['url']

        X_num_scaled = num_pipe.transform(X_num)
        X_bool_enc = bool_pipe.transform(X_bool)
        X_text_tfidf = tfidf.transform(X_text)
        X_full = sparse.hstack([sparse.csr_matrix(X_num_scaled),
                                sparse.csr_matrix(X_bool_enc),
                                X_text_tfidf]).tocsr()

        y = df['label']
        y_pred = model.predict(X_full)
        y_prob = model.predict_proba(X_full)[:, 1]

        return y, y_pred, y_prob
    except Exception as e:
        print("[ERROR] Could not compute metrics:", e)
        return None, None, None

# -------------------------------
# Confusion matrix
# -------------------------------
@app.get("/confusion_matrix")
def get_confusion_matrix():
    y, y_pred, _ = compute_metrics()
    if y is None:
        return {"error": "Metrics not available"}
    cm = confusion_matrix(y, y_pred).tolist()
    return {"confusion_matrix": cm}

# -------------------------------
# ROC curve
# -------------------------------
@app.get("/roc_curve")
def get_roc_curve():
    y, _, y_prob = compute_metrics()
    if y is None:
        return {"error": "Metrics not available"}
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    return {
        "fpr": sanitize_list(fpr.tolist()),
        "tpr": sanitize_list(tpr.tolist()),
        "thresholds": sanitize_list(thresholds.tolist())
    }

# -------------------------------
# Precision–Recall curve
# -------------------------------
@app.get("/precision_recall")
def get_precision_recall():
    y, _, y_prob = compute_metrics()
    if y is None:
        return {"error": "Metrics not available"}
    precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
    return {
        "precision": sanitize_list(precision.tolist()),
        "recall": sanitize_list(recall.tolist()),
        "thresholds": sanitize_list(pr_thresholds.tolist())
    }

# -------------------------------
# Feature importance (top features)
# -------------------------------
@app.get("/feature_importance")
def get_feature_importance():
    try:
        importances = model.feature_importances_[:len(numeric_features + bool_features)]
        feature_names = numeric_features + bool_features
        top_idx = np.argsort(importances)[-len(feature_names):]
        top_features = [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in top_idx
        ]
        return {"top_features": top_features}
    except Exception as e:
        return {"error": str(e)}
