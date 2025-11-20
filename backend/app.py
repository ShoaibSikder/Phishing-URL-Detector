from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib, pandas as pd

from scipy import sparse

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

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
# Feature extraction for new URL
# -------------------------------
def extract_features(url: str):
    # Dummy values for numeric/bool features (model only expects correct column names)
    numeric_df = pd.DataFrame([{'url_length': len(url),
                                'num_dots': url.count('.'),
                                'num_digits': sum(c.isdigit() for c in url)}])
    numeric_df = numeric_df[numeric_features]

    bool_df = pd.DataFrame([{'has_https': 1 if url.startswith('https') else 0,
                             'local_keyword_flag': 0}])
    bool_df = bool_df[bool_features]

    return numeric_df, bool_df

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

@app.get("/")
def root():
    return {"status": "ok", "msg": "Phishing Detector API running"}

@app.get("/metrics")
def metrics():
    return {"test_accuracy": round(saved.get('test_accuracy', 0) * 100, 2)}

@app.post("/predict")
def predict(req: URLRequest):
    try:
        num_df, bool_df = extract_features(req.url)
        text = [req.url.lower().replace('https://','').replace('http://','')]

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
