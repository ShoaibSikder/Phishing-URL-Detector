import os, re, math, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent

# -------------------------------
# Feature extractor (simple for your dataset)
# -------------------------------
class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def transform(self, X, y=None):
        # Only numeric and bool features already in dataset
        return X

    def fit(self, X, y=None):
        return self

# -------------------------------
# Main training
# -------------------------------
def main():
    data_path = ROOT.parent / "dataset" / "phishing_data.csv"
    if not data_path.exists():
        print("Place your CSV at dataset/phishing_data.csv")
        return

    df = pd.read_csv(data_path)
    df = df.dropna(subset=['url','label']).reset_index(drop=True)
    df['url'] = df['url'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)

    # Features based on your CSV columns
    numeric_features = ['url_length','num_dots','num_digits']
    bool_features = ['has_https','local_keyword_flag']

    X_num = df[numeric_features]
    X_bool = df[bool_features]
    X_text = df['url']
    y = df['label']

    # Split dataset
    X_train_num, X_test_num, X_train_bool, X_test_bool, X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_num, X_bool, X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipelines
    num_pipe = Pipeline([('scaler', StandardScaler())])
    bool_pipe = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000)

    X_train_num_scaled = num_pipe.fit_transform(X_train_num)
    X_test_num_scaled = num_pipe.transform(X_test_num)
    X_train_bool_enc = bool_pipe.fit_transform(X_train_bool)
    X_test_bool_enc = bool_pipe.transform(X_test_bool)
    X_train_text_tfidf = tfidf.fit_transform(X_train_text)
    X_test_text_tfidf = tfidf.transform(X_test_text)

    from scipy import sparse
    X_train_full = sparse.hstack([sparse.csr_matrix(X_train_num_scaled),
                                  sparse.csr_matrix(X_train_bool_enc),
                                  X_train_text_tfidf]).tocsr()
    X_test_full = sparse.hstack([sparse.csr_matrix(X_test_num_scaled),
                                 sparse.csr_matrix(X_test_bool_enc),
                                 X_test_text_tfidf]).tocsr()

    # Train model
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

    # Save model + pipelines
    save_obj = {
        'model': best,
        'num_pipe': num_pipe,
        'bool_pipe': bool_pipe,
        'tfidf': tfidf,
        'numeric_features': numeric_features,
        'bool_features': bool_features,
        'test_accuracy': acc
    }
    joblib.dump(save_obj, ROOT / "model.pkl")
    print(f"Model saved at {ROOT / 'model.pkl'}")

if __name__ == "__main__":
    main()
