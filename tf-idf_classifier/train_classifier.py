import os
import joblib
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np

LABEL_MAP = {
    "chat": 0,
    "meta": 1,
    "rag": 2,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

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

def log_metrics(y_true, y_pred, split_name, step=None):
    """Log metrics to W&B for a given split"""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics = {
        f"{split_name}/accuracy": acc,
        f"{split_name}/precision": precision,
        f"{split_name}/recall": recall,
        f"{split_name}/f1": f1,
    }
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    for i, label_name in LABEL_NAMES.items():
        if i < len(precision_per_class):
            metrics[f"{split_name}/{label_name}_precision"] = precision_per_class[i]
            metrics[f"{split_name}/{label_name}_recall"] = recall_per_class[i]
            metrics[f"{split_name}/{label_name}_f1"] = f1_per_class[i]
    
    wandb.log(metrics, step=step)
    
    # Log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    wandb.log({
        f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
        )
    }, step=step)
    
    return acc, f1

def train_and_save(data_dir, out_dir="model", test_size=0.2, val_size=0.1, random_seed=42):
    """
    Train classifier with train/val/test split and log to W&B
    
    Args:
        data_dir: Directory containing {chat,meta,rag}.txt files
        out_dir: Directory to save model artifacts
        test_size: Proportion of data for test set (default 0.2 = 20%)
        val_size: Proportion of remaining data for validation set (default 0.1 = 10%)
        random_seed: Random seed for reproducibility
    """
    # Initialize W&B
    wandb.init(
        project="text-classifier",
        config={
            "model": "LogisticRegression",
            "vectorizer": "TfidfVectorizer",
            "ngram_range": (1, 2),
            "max_features": 20000,
            "max_iter": 1000,
            "test_size": test_size,
            "val_size": val_size,
            "random_seed": random_seed,
        }
    )
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Load data
    texts, labels = load_data(data_dir)
    print(f"Total samples: {len(texts)}")
    wandb.log({"total_samples": len(texts)})
    
    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = {LABEL_NAMES[label]: count for label, count in zip(unique, counts)}
    print(f"Class distribution: {class_dist}")
    wandb.log({"class_distribution": class_dist})
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_seed,
        stratify=labels
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size relative to temp set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_seed,
        stratify=y_temp
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(texts)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(texts)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(texts)*100:.1f}%)")
    
    wandb.log({
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    })
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        lowercase=True,
        strip_accents="unicode",
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"\nFeature matrix shape: {X_train_vec.shape}")
    wandb.log({"n_features": X_train_vec.shape[1]})
    
    # Train
    clf = LogisticRegression(
        max_iter=1000,
        solver='lbfgs',
        random_state=random_seed
    )
    
    print("\nTraining...")
    clf.fit(X_train_vec, y_train)
    
    # Evaluate on all splits
    print("\nEvaluating...")
    
    # Train metrics
    y_train_pred = clf.predict(X_train_vec)
    train_acc, train_f1 = log_metrics(y_train, y_train_pred, "train")
    print(f"Train - Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    
    # Validation metrics
    y_val_pred = clf.predict(X_val_vec)
    val_acc, val_f1 = log_metrics(y_val, y_val_pred, "val")
    print(f"Val   - Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Test metrics
    y_test_pred = clf.predict(X_test_vec)
    test_acc, test_f1 = log_metrics(y_test, y_test_pred, "test")
    print(f"Test  - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Detailed classification report for test set
    print(f"\nTest Set Classification Report:")
    print(classification_report(
        y_test, y_test_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES.keys())]
    ))
    
    # Save models
    vectorizer_path = os.path.join(out_dir, "tfidf.joblib")
    classifier_path = os.path.join(out_dir, "classifier.joblib")
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, classifier_path)
    
    # Log model artifacts to W&B
    artifact = wandb.Artifact('text-classifier-model', type='model')
    artifact.add_file(vectorizer_path)
    artifact.add_file(classifier_path)
    wandb.log_artifact(artifact)
    
    print(f"\nSaved:")
    print(f"- {vectorizer_path}")
    print(f"- {classifier_path}")
    
    wandb.finish()

if __name__ == "__main__":
    train_and_save("tf-idf_classifier/dataset")