# backend/train_model.py
import os, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, precision_recall_curve
from scipy import sparse

ROOT = Path(__file__).resolve().parent

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        # Keep as pass-through because numeric/bool features already exist
        return X

    def fit(self, X, y=None):
        return self

def main():
    data_path = ROOT.parent / "dataset" / "phishing_data.csv"
    if not data_path.exists():
        print("Place your CSV at dataset/phishing_data.csv")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['url','label']).reset_index(drop=True)
    df['url'] = df['url'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)

    # Basic features (match your dataset)
    numeric_features = ['url_length','num_dots','num_digits']
    bool_features = ['has_https','local_keyword_flag']

    # If dataset doesn't have numeric/bool columns already, create them
    if not set(numeric_features).issubset(df.columns):
        df['url_length'] = df['url'].apply(len)
        df['num_dots'] = df['url'].apply(lambda u: u.count('.'))
        df['num_digits'] = df['url'].apply(lambda u: sum(c.isdigit() for c in u))

    if not set(bool_features).issubset(df.columns):
        df['has_https'] = df['url'].apply(lambda u: 1 if u.lower().startswith('https') else 0)
        df['local_keyword_flag'] = 0  # default 0; you can update this logic as needed

    X_num = df[numeric_features]
    X_bool = df[bool_features]
    X_text = df['url']
    y = df['label']

    # Split (note: keep aligned splits)
    X_train_num, X_test_num, X_train_bool, X_test_bool, X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_num, X_bool, X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipelines
    num_pipe = Pipeline([('scaler', StandardScaler())])
    bool_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)

    # Fit/transform
    X_train_num_scaled = num_pipe.fit_transform(X_train_num)
    X_test_num_scaled = num_pipe.transform(X_test_num)
    X_train_bool_enc = bool_pipe.fit_transform(X_train_bool)
    X_test_bool_enc = bool_pipe.transform(X_test_bool)
    X_train_text_tfidf = tfidf.fit_transform(X_train_text)
    X_test_text_tfidf = tfidf.transform(X_test_text)

    X_train_full = sparse.hstack([sparse.csr_matrix(X_train_num_scaled),
                                  sparse.csr_matrix(X_train_bool_enc),
                                  X_train_text_tfidf]).tocsr()
    X_test_full = sparse.hstack([sparse.csr_matrix(X_test_num_scaled),
                                 sparse.csr_matrix(X_test_bool_enc),
                                 X_test_text_tfidf]).tocsr()

    # Train model with GridSearch (can be slower)
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {'n_estimators':[100,200],'max_depth':[None,20],'min_samples_split':[2,5]}
    gs = GridSearchCV(clf,param_grid,cv=3,n_jobs=-1,scoring='accuracy',verbose=1)
    print("Starting GridSearchCV...")
    gs.fit(X_train_full, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test_full)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Compute metrics for frontend
    cm = confusion_matrix(y_test, y_pred).tolist()
    # For ROC/PR we need probability for positive class (assume label '1' is phishing)
    if hasattr(best, "predict_proba"):
        y_score = best.predict_proba(X_test_full)[:, 1]  # phishing probability
    else:
        # fallback: decision_function if available
        try:
            y_score = best.decision_function(X_test_full)
        except:
            y_score = np.zeros_like(y_test)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_score)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)

    # Feature importance: combine numeric + bool (one-hot expanded) + tfidf names
    # numeric and bool positions
    tfidf_feature_count = X_train_text_tfidf.shape[1]
    # Create feature name list
    tfidf_names = [f"tfidf_{i}" for i in range(tfidf_feature_count)]
    # bool pipeline OneHotEncoder categories -> get names
    bool_ohe_feature_names = []
    try:
        # sklearn >=1.0: get_feature_names_out
        bool_ohe_feature_names = list(bool_pipe.named_steps['ohe'].get_feature_names_out(bool_features))
    except:
        # fallback simple names
        bool_ohe_feature_names = []
        for b in bool_features:
            bool_ohe_feature_names.append(b)

    feature_names = numeric_features + bool_ohe_feature_names + tfidf_names
    importances = best.feature_importances_
    # Safeguard length mismatch (if any)
    if len(importances) != len(feature_names):
        # Trim or pad names to match importance vector length
        if len(importances) < len(feature_names):
            feature_names = feature_names[:len(importances)]
        else:
            # pad names
            feature_names += [f"f_{i}" for i in range(len(importances) - len(feature_names))]

    # Save model + pipelines + metrics
    save_obj = {
        'model': best,
        'num_pipe': num_pipe,
        'bool_pipe': bool_pipe,
        'tfidf': tfidf,
        'numeric_features': numeric_features,
        'bool_features': bool_features,
        'test_accuracy': float(acc),
        'confusion_matrix': cm,
        'roc': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': roc_thresholds.tolist()
        },
        'precision_recall': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        },
        'feature_importance': {
            'features': feature_names,
            'importance': importances.tolist()
        }
    }
    joblib.dump(save_obj, ROOT / "model.pkl")
    print(f"Model + metrics saved at {ROOT / 'model.pkl'}")

if __name__ == "__main__":
    main()
