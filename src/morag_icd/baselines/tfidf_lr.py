from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
import pickle
from pathlib import Path
import numpy as np

class TFIDFLRBaseline:
    def __init__(self, top_k: int = 10):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
        ])
        self.mlb = MultiLabelBinarizer()
        self.top_k = top_k
        self._is_fitted = False
        
    def fit(self, train_samples_or_texts, Y_train: list[list[int]] | None = None):
        if Y_train is not None:
            X_train = train_samples_or_texts
            y_bin = Y_train
            if getattr(self.mlb, "classes_", None) is None and len(y_bin) > 0:
                self.mlb.fit([[]])
        else:
            samples = train_samples_or_texts
            X_train = [s.get("text", "") for s in samples]
            y_codes = [s.get("gold_codes", []) for s in samples]
            y_bin = self.mlb.fit_transform(y_codes)
        self.model.fit(X_train, y_bin)
        self._is_fitted = True
        
    def predict(self, X_test: list[str]) -> list[list[int]]:
        if not self._is_fitted:
            return [[] for _ in X_test]
        y_pred = self.model.predict(X_test)
        if hasattr(self.mlb, "classes_") and len(self.mlb.classes_) > 0:
            return self.mlb.inverse_transform(y_pred)
        return [[] for _ in X_test]

    def process_note(self, sample_or_text):
        text = sample_or_text.get("text", "") if isinstance(sample_or_text, dict) else str(sample_or_text)
        if not self._is_fitted or not hasattr(self.mlb, "classes_") or len(self.mlb.classes_) == 0:
            return []

        prob = self.model.predict_proba([text])[0]
        top_indices = np.argsort(prob)[::-1][: self.top_k]
        preds = []
        for idx in top_indices:
            preds.append(
                {
                    "code": str(self.mlb.classes_[idx]),
                    "confidence": float(prob[idx]),
                    "supported": None,
                    "evidence_score": 0.0,
                    "icd_description": "",
                    "evidence_preview": "",
                    "rationale": "",
                    "risk_flag": "baseline",
                }
            )
        return preds
        
    def save(self, filepath: str | Path):
        with open(filepath, 'wb') as f:
            pickle.dump({"model": self.model, "mlb": self.mlb, "top_k": self.top_k}, f)
