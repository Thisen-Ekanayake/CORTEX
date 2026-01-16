# Text Classification System Documentation

## Overview

This code implements a **text classification pipeline** that categorizes user queries into three intent types: `chat`, `meta`, and `rag`. It uses **TF-IDF vectorization** for feature extraction and **Logistic Regression** for classification.

## System Architecture

### Components

1. **Data Loading**: Reads labeled text data from separate files
2. **Feature Extraction**: Converts text to numerical features using TF-IDF
3. **Classification**: Trains a logistic regression model to predict intent labels
4. **Model Persistence**: Saves trained models for later use

### Label Mapping

```python
LABEL_MAP = {
    "chat": 0,      # General conversation queries
    "meta": 1,      # Meta-queries about the system itself
    "rag": 2,       # Retrieval-augmented generation queries
}
```

## Functions Reference

### `load_data(data_dir)`

Loads training data from text files.

**Parameters:**
- `data_dir` (str): Directory containing training data files

**Returns:**
- `texts` (list): All text samples
- `labels` (list): Corresponding numeric labels

**Expected File Structure:**
```
data_dir/
  ├── chat.txt
  ├── meta.txt
  └── rag.txt
```

Each file contains one training example per line.

### `train_and_save(data_dir, out_dir="tf-idf_classifier")`

Trains the classification pipeline and saves models.

**Parameters:**
- `data_dir` (str): Directory containing training data
- `out_dir` (str): Directory to save trained models

**Outputs:**
- `tfidf.joblib`: Trained TF-IDF vectorizer
- `classifier.joblib`: Trained logistic regression model

## Hyperparameters Guide

### TF-IDF Vectorizer Hyperparameters

#### 1. `ngram_range=(1, 2)`

**What it does:** Specifies the range of n-grams to extract.
- `(1, 2)` means both unigrams (single words) and bigrams (two-word phrases)

**Tuning options:**
- `(1, 1)`: Only single words (simpler, faster)
- `(1, 2)`: Single words + two-word phrases (current setting)
- `(1, 3)`: Includes three-word phrases (more context, slower)
- `(2, 2)`: Only bigrams (loses individual word information)

**Effects of changing:**

| Setting | Training Speed | Model Size | Accuracy Potential | Best For |
|---------|---------------|------------|-------------------|----------|
| `(1, 1)` | Fastest | Smallest | Lower | Simple classification tasks |
| `(1, 2)` | Medium | Medium | Good | Balanced performance (default) |
| `(1, 3)` | Slower | Larger | Higher (if enough data) | Complex patterns, larger datasets |
| `(2, 3)` | Slowest | Largest | Variable | Phrase-heavy classification |

**When to increase:** You have lots of training data and phrases like "thank you" or "what is" are important for classification.

**When to decrease:** You have limited data, training is too slow, or overfitting occurs.

---

#### 2. `max_features=20000`

**What it does:** Limits vocabulary to the top 20,000 most frequent terms.

**Tuning options:**
- `5000-10000`: Smaller vocabulary (faster, less memory)
- `20000`: Current setting (balanced)
- `50000+`: Larger vocabulary (more features)
- `None`: No limit (use all terms)

**Effects of changing:**

| Value | Memory Usage | Training Speed | Underfitting Risk | Overfitting Risk |
|-------|--------------|----------------|-------------------|------------------|
| 5,000 | Low | Fast | Higher | Lower |
| 20,000 | Medium | Medium | Medium | Medium |
| 50,000 | High | Slow | Lower | Higher |
| None | Very High | Slowest | Lowest | Highest |

**When to increase:** Your vocabulary is rich and diverse, or you're seeing underfitting (low training accuracy).

**When to decrease:** Memory constraints, training too slow, or overfitting (high training accuracy, low validation accuracy).

---

#### 3. `lowercase=True`

**What it does:** Converts all text to lowercase before processing.

**Options:**
- `True`: "Hello" and "hello" are treated the same (current setting)
- `False`: "Hello" and "hello" are different features

**Effects of changing to False:**
- **Pros:** Preserves case information (e.g., "US" vs "us")
- **Cons:** Doubles vocabulary size, requires more data, may overfit

**When to change:** Case distinctions are meaningful (e.g., proper nouns, acronyms).

---

#### 4. `strip_accents="unicode"`

**What it does:** Removes accent marks (é → e, ñ → n).

**Options:**
- `"unicode"`: Remove accents using Unicode normalization (current)
- `"ascii"`: Remove accents using ASCII approximations
- `None`: Keep accents

**Effects of changing:**

| Setting | Effect | Use Case |
|---------|--------|----------|
| `"unicode"` | Normalizes characters | Multilingual text |
| `"ascii"` | Aggressive normalization | English-focused |
| `None` | Preserves accents | Language-specific models |

**When to change:** You need to preserve language-specific characters or your data is purely English.

---

### Logistic Regression Hyperparameters

#### 5. `max_iter=1000`

**What it does:** Maximum number of iterations for the optimization algorithm.

**Tuning options:**
- `100-500`: Faster training, may not converge
- `1000`: Current setting (usually sufficient)
- `2000+`: Slower but ensures convergence

**Effects of changing:**

| Value | Training Time | Convergence | Use Case |
|-------|---------------|-------------|----------|
| 100 | Very Fast | May fail | Quick prototyping |
| 500 | Fast | Usually OK | Standard datasets |
| 1000 | Medium | Reliable | Default (recommended) |
| 5000 | Slow | Guaranteed | Complex/large datasets |

**Warning signs you need to increase:**
- Convergence warnings during training
- Model performance improves with higher values

**When to decrease:** Training is unnecessarily slow and model converges quickly.

---

#### 6. `solver='lbfgs'`

**What it does:** Optimization algorithm for finding model weights.

**Options:**

| Solver | Speed | Memory | Best For |
|--------|-------|--------|----------|
| `'lbfgs'` | Medium | Low | Small/medium datasets (current) |
| `'saga'` | Fast | Medium | Large datasets, L1 regularization |
| `'liblinear'` | Fast | Low | Small datasets, binary classification |
| `'newton-cg'` | Slow | Low | High accuracy needed |
| `'sag'` | Fast | Low | Large datasets |

**Effects of changing:**

- **'lbfgs'** (current): Good default, handles multiclass well
- **'saga'**: Better for >100k samples, supports L1 penalty
- **'liblinear'**: Better for small datasets (<10k samples)

**When to change:** You have a very large dataset (use 'saga') or very small dataset (use 'liblinear').

---

### Additional Hyperparameters You Can Add

#### Regularization: `C` (not currently specified)

**Default value:** `C=1.0`

**What it does:** Controls regularization strength (inverse).
- Higher C = Less regularization = More complex model
- Lower C = More regularization = Simpler model

**Tuning options:**
```python
clf = LogisticRegression(
    C=0.1,  # Strong regularization
    max_iter=1000,
    solver='lbfgs'
)
```

**Effects:**

| C Value | Model Complexity | Overfitting Risk | Underfitting Risk |
|---------|------------------|------------------|-------------------|
| 0.01 | Very Simple | Very Low | High |
| 0.1 | Simple | Low | Medium |
| 1.0 | Balanced | Medium | Medium |
| 10.0 | Complex | High | Low |
| 100.0 | Very Complex | Very High | Very Low |

**How to tune:** Use cross-validation to find optimal C:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, labels)
print(f"Best C: {grid_search.best_params_['C']}")
```

---

#### TF-IDF: `min_df` and `max_df`

**What they do:** Filter terms by document frequency.

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2,      # Ignore terms appearing in < 2 documents
    max_df=0.95,   # Ignore terms appearing in > 95% of documents
    lowercase=True,
    strip_accents="unicode",
)
```

**Effects:**

| Parameter | Purpose | Effect on Vocabulary |
|-----------|---------|---------------------|
| `min_df=2` | Remove very rare terms | Smaller vocabulary, less noise |
| `max_df=0.95` | Remove very common terms | Removes stop words, focuses on discriminative terms |

---

## Hyperparameter Tuning Workflow

### Step 1: Start with Baseline

Use default settings to establish baseline performance:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts, labels = load_data("dataset")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000, solver='lbfgs')
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))
```

### Step 2: Identify Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Low training & test accuracy | Underfitting | Increase `max_features`, use `(1,3)` ngrams, increase `C` |
| High training, low test accuracy | Overfitting | Decrease `max_features`, decrease `C`, add `min_df` |
| Slow training | Too many features | Decrease `max_features`, use `(1,1)` ngrams |
| Convergence warnings | Insufficient iterations | Increase `max_iter` |

### Step 3: Systematic Grid Search

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__max_features': [10000, 20000, 50000],
    'clf__C': [0.1, 1.0, 10.0],
    'clf__solver': ['lbfgs', 'saga']
}

grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=5, 
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(texts, labels)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### Step 4: Validate on Holdout Set

Always test final model on completely unseen data.

---

## Common Scenarios & Recommendations

### Scenario 1: Small Dataset (<1,000 samples)

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),        # Simple features
    max_features=5000,         # Fewer features
    min_df=1,                  # Keep rare terms
    lowercase=True
)

clf = LogisticRegression(
    C=0.1,                     # Strong regularization
    max_iter=1000,
    solver='liblinear'         # Good for small data
)
```

### Scenario 2: Large Dataset (>100,000 samples)

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),        # Richer features
    max_features=50000,        # More features
    min_df=5,                  # Filter noise
    max_df=0.95,               # Remove common terms
    lowercase=True
)

clf = LogisticRegression(
    C=1.0,
    max_iter=500,              # May converge faster
    solver='saga'              # Efficient for large data
)
```

### Scenario 3: Overfitting

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 1),        # Reduce complexity
    max_features=10000,        # Fewer features
    min_df=3,                  # Remove rare terms
    max_df=0.90
)

clf = LogisticRegression(
    C=0.01,                    # Strong regularization
    max_iter=1000,
    solver='lbfgs'
)
```

### Scenario 4: Underfitting

```python
vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),        # More features
    max_features=None,         # No limit
    min_df=1,
    lowercase=True
)

clf = LogisticRegression(
    C=10.0,                    # Less regularization
    max_iter=2000,
    solver='lbfgs'
)
```

---

## Performance Monitoring

### Metrics to Track

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# After training and prediction
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

### Training Time vs Accuracy Trade-off

Monitor these during hyperparameter tuning:

```python
import time

start = time.time()
clf.fit(X_train, y_train)
training_time = time.time() - start

print(f"Training time: {training_time:.2f}s")
print(f"Test accuracy: {clf.score(X_test, y_test):.4f}")
```

---

## Quick Reference Table

| Hyperparameter | Default | Range | Primary Effect |
|----------------|---------|-------|----------------|
| `ngram_range` | `(1,2)` | `(1,1)` to `(1,3)` | Feature richness |
| `max_features` | `20000` | `5000-None` | Model complexity |
| `lowercase` | `True` | `True/False` | Case sensitivity |
| `max_iter` | `1000` | `100-5000` | Convergence |
| `solver` | `'lbfgs'` | Various | Optimization method |
| `C` | `1.0` | `0.01-100` | Regularization strength |
| `min_df` | Not set | `1-10` | Noise filtering |
| `max_df` | Not set | `0.8-1.0` | Stop word removal |

---

## Summary

This text classification system uses TF-IDF and logistic regression to categorize user queries. The main hyperparameters control feature extraction complexity (ngram_range, max_features) and model regularization (C). Start with defaults, monitor performance metrics, and adjust systematically based on whether you're experiencing overfitting, underfitting, or computational constraints.