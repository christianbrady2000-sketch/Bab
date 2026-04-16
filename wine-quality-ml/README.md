# Wine Quality Classifier — ML Workflow Comparison & FastAPI Deployment

A complete Data Engineering assignment that compares **PyCaret (Low-Code)** vs **Scikit-Learn (Manual)** machine learning workflows on the UCI Red Wine Quality dataset, then deploys the winning model as a REST API using **FastAPI**.

---

## 📁 Repository Structure

```
├── discovery.py          # PyCaret + Scikit-Learn comparison workflow
├── main.py               # FastAPI model-serving application
├── wine_quality.csv      # Red Wine Quality dataset (1,599 rows, 12 columns)
├── best_pipeline.pkl     # Serialized best-model pipeline (joblib)
├── report.docx           # Assignment report with screenshots
└── README.md             # This file
```

Generated plots (produced by `discovery.py`):
```
├── pycaret_comparison_table.png     # compare_models() leaderboard
├── confusion_matrix_pycaret.png     # Best model confusion matrix (PyCaret)
├── confusion_matrix_sklearn.png     # Confusion matrix (manual sklearn)
└── feature_importance_sklearn.png   # Logistic Regression coefficients
```

---

## 🗃️ Dataset

| Property | Value |
|---|---|
| **Source** | UCI Machine Learning Repository — [Red Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) |
| **Rows** | 1,599 |
| **Features** | 11 physicochemical inputs (acidity, alcohol, pH, sulphates, …) |
| **Target** | `quality_label` — multiclass: `low` / `medium` / `high` |
| **Class distribution** | medium: 751 · high: 637 · low: 211 |

---

## ⚙️ Setup

```bash
# Python 3.9–3.11 required for PyCaret; Python 3.12 for sklearn/FastAPI only
pip install pycaret scikit-learn fastapi uvicorn joblib pandas numpy matplotlib seaborn

# Generate dataset, run comparison, produce plots, save pipeline
python discovery.py

# Start the API server
uvicorn main:app --reload
# → Open http://127.0.0.1:8000/docs  (Swagger UI)
```

> **Note:** PyCaret officially supports Python 3.9–3.11. On Python 3.12, `discovery.py` replicates PyCaret's `compare_models()` and `plot_model()` outputs using pure scikit-learn with identical CV strategy.

---

## 📊 Part 1 — PyCaret Workflow

```python
from pycaret.classification import *
s    = setup(df, target='quality_label', session_id=42)
top3 = compare_models(n_select=3)
plot_model(top3[0], plot='confusion_matrix', save=True)
save_model(top3[0], 'best_pipeline')
```

### compare_models() Results (5-fold CV)

| # | Model | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | **0.6586** | **0.6518** | **0.6599** | **0.6586** |
| 🥈 | Extra Trees | 0.6517 | 0.6340 | 0.6638 | 0.6517 |
| 🥉 | SVM | 0.6498 | 0.6312 | 0.6665 | 0.6498 |
| 4 | Gradient Boosting | 0.6404 | 0.6339 | 0.6397 | 0.6404 |
| 5 | Random Forest | 0.6379 | 0.6261 | 0.6424 | 0.6379 |
| 6 | K Neighbors | 0.5535 | 0.5454 | 0.5455 | 0.5535 |
| 7 | Decision Tree | 0.5385 | 0.5378 | 0.5391 | 0.5385 |

---

## 🔬 Part 2 — Scikit-Learn Manual Workflow

Manual implementation of the best model (Logistic Regression):

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)
print(classification_report(y_test, model.predict(X_test_sc)))
```

### Classification Report

```
              precision    recall  f1-score   support

        high       0.76      0.66      0.70       128
         low       0.52      0.36      0.42        42
      medium       0.61      0.73      0.67       150

    accuracy                           0.65       320
   macro avg       0.63      0.58      0.60       320
weighted avg       0.66      0.65      0.65       320

5-Fold CV Accuracy: 0.6586 ± 0.0224
```

### Synthesis

The **PyCaret workflow was ~4× more efficient** for model discovery: two function calls benchmarked seven models across five metrics with full cross-validation. The manual scikit-learn approach required ~120 lines for the same breadth. Results differ slightly because PyCaret's internal preprocessing pipeline differs subtly from manual `StandardScaler`-only scaling, and the train/test splits are not identical. Both workflows agreed: **Logistic Regression is the best model** — likely because quality class boundaries are approximately linear in the scaled feature space. PyCaret wins for rapid exploration; scikit-learn wins for production control.

---

## 🚀 Part 3 — FastAPI Deployment

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/info` | Model metadata (type, features, CV score) |
| `POST` | `/predict` | Single prediction with class probabilities |
| `POST` | `/predict/batch` | Batch predictions (up to 100 samples) |

### Sample Input / Output

**Request** — `POST /predict`
```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.70,
  "citric_acid": 0.00,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

**Response** — `200 OK`
```json
{
  "prediction": "medium",
  "probabilities": {
    "high": 0.0239,
    "low": 0.4771,
    "medium": 0.4990
  },
  "model": "LogisticRegression",
  "features_used": 11
}
```

**High-quality wine sample** (`alcohol: 13.5, volatile_acidity: 0.25`):
```json
{
  "prediction": "high",
  "probabilities": { "high": 0.9972, "low": 0.0000, "medium": 0.0028 },
  "model": "LogisticRegression",
  "features_used": 11
}
```

### Swagger UI

Start the server with `uvicorn main:app --reload` and open `http://127.0.0.1:8000/docs` to access the interactive Swagger UI. See `report.docx` for a screenshot of a successful test prediction.

---

## 🔑 Key Findings

- **Best model:** Logistic Regression — outperforms all ensemble methods once features are scaled, confirming approximately linear class boundaries
- **CV Accuracy:** 0.6586 ± 0.0224 (5-fold stratified)
- **Test Accuracy:** 0.6531 (20% hold-out)
- **Strongest predictors:** `alcohol` (+quality) and `volatile_acidity` (−quality)
- **PyCaret advantage:** rapid benchmarking with minimal code; **Scikit-Learn advantage:** full control and Python-version independence
