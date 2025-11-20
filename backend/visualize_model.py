import joblib, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix
from train_model import URLFeatureExtractor
from scipy import sparse
import numpy as np

# Load model
saved = joblib.load("backend/model.pkl")
model = saved['model']
num_pipe = saved['num_pipe']
bool_pipe = saved['bool_pipe']
tfidf = saved['tfidf']
numeric_features = saved['numeric_features']
bool_features = saved['bool_features']

# Load dataset
df = pd.read_csv("dataset/phishing_data.csv").dropna(subset=['url','label'])
df['url'] = df['url'].astype(str).str.strip()
df['label'] = df['label'].astype(int)

url_feat = URLFeatureExtractor().fit_transform(df['url'])
df = pd.concat([df.reset_index(drop=True), url_feat.reset_index(drop=True)], axis=1)
df['url_text'] = df['url']

X_num = df[numeric_features]
X_bool = df[bool_features]
X_text = df['url_text']
y = df['label']

X_num_scaled = num_pipe.transform(X_num)
X_bool_enc = bool_pipe.transform(X_bool)
X_text_tfidf = tfidf.transform(X_text)

X_full = sparse.hstack([sparse.csr_matrix(X_num_scaled),sparse.csr_matrix(X_bool_enc),X_text_tfidf]).tocsr()

y_pred = model.predict(X_full)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate','Phishing'], yticklabels=['Legitimate','Phishing'])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# Feature importance
feature_names = numeric_features + bool_features + [f"tfidf_{i}" for i in range(X_text_tfidf.shape[1])]
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]

plt.figure(figsize=(8,6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 20 Feature Importances")
plt.show()
