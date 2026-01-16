import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

LABEL_MAP = {
    "chat": 0,
    "meta": 1,
    "rag": 2,
}

def load_data(data_dir):
    texts = []
    labels = []
    for name, label in LABEL_MAP.items():
        path = os.path.join(data_dir, f"{name}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(label)
    return texts, labels

def train_and_save(data_dir, out_dir="model"):
    os.makedirs(out_dir, exist_ok=True)
    texts, labels = load_data(data_dir)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        lowercase=True,
        strip_accents="unicode",
    )
    X = vectorizer.fit_transform(texts)
    
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs'
    )
    clf.fit(X, labels)
    
    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "classifier.joblib"))
    
    print("Saved:")
    print("- tfidf.joblib")
    print("- classifier.joblib")

if __name__ == "__main__":
    train_and_save("dataset")