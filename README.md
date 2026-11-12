# FraudLens

FraudLens is an interactive Streamlit dashboard for demonstrating explainable,
cost-sensitive credit-card fraud detection with a saved XGBoost model.

The app is designed as a research and portfolio project. It lets users inspect
sample transaction predictions, compare validation results, explore threshold
trade-offs, and view both global and transaction-level SHAP explanations.

## Dashboard

- **Transaction Prediction** — Select an example transaction and view its fraud
  score, prediction, decision threshold, known sample label, and raw features.
  The selector can be filtered to All, Genuine, or Fraud examples.
- **Model Analytics** — Compare Logistic Regression, Random Forest, and XGBoost
  on validation data, then inspect the saved final test-set results.
- **Cost Optimization** — Move a threshold slider to explore the saved
  validation false-positive, missed-fraud, precision, recall, and estimated-cost
  trade-offs.
- **Explainable AI** — Review a global SHAP beeswarm plot and generate a local
  SHAP waterfall explanation for the selected transaction.

## Results

### Validation comparison at threshold 0.50

| Model | Precision | Recall | F1-score | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 6.07% | 85.71% | 11.34% | 0.6295 | 0.9614 |
| Random Forest | 83.72% | 73.47% | 78.26% | 0.7557 | 0.9593 |
| XGBoost | 86.36% | 77.55% | 81.72% | 0.7961 | 0.9684 |

### Final test set

The held-out test set contains 56,962 transactions, including 98 fraud cases.

| Metric | Result |
| --- | ---: |
| PR-AUC | 0.8709 |
| ROC-AUC | 0.9758 |
| Precision at 0.50 | 79.81% |
| Recall at 0.50 | 84.69% |

The validation-selected threshold (0.62) improved precision on the test set,
but it increased the estimated test cost from $4,314 to $4,516. FraudLens shows
this result explicitly rather than presenting threshold optimisation as a
universal improvement.

## Project structure

```text
FraudLens/
├── app.py
├── requirements.txt
├── data/
│   └── sample_transactions.csv
├── models/
│   ├── xgb_fraud_model.pkl
│   ├── preprocessing.pkl
│   └── optimal_threshold.pkl
└── results/
    ├── model_comparison.csv
    ├── threshold_optimization.csv
    ├── test_threshold_comparison.csv
    └── shap_beeswarm.png
```

## Run locally

Requires Python 3.9 or later.

```bash
git clone https://github.com/anushkasngh31/fraudlens-explainable-credit-card-fraud-detection
cd FraudLens
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## Using the dashboard

1. In **Transaction Prediction**, use the filter to select a genuine or fraud
   example and inspect its prediction.
2. In **Model Analytics**, compare validation metrics and review final test-set
   performance separately.
3. In **Cost Optimization**, move the threshold slider and examine the saved
   validation trade-off curve.
4. In **Explainable AI**, click **Generate SHAP waterfall** to explain the
   currently selected transaction.

## Important limitations

- This is a research demonstration, not a production banking system.
- The sample transaction labels are shown only to support the demonstration; a
  live prediction system would not know the actual class at decision time.
- The model was trained with class weighting. Its output scores should not be
  interpreted as calibrated real-world fraud probabilities.
- The cost values are hypothetical scenario assumptions: USD 250 per missed
  fraud, USD 15 per false alarm, and USD 3 per detected fraud. They are not
  actual financial losses.
- `V1` through `V28` are anonymized PCA-derived variables, so their business
  interpretation is limited.

## Data and artifacts

The repository includes a small sample dataset and the saved artifacts required
to run the dashboard. The full source dataset is intentionally excluded. Do
not commit confidential, customer, or payment-card data.

## Verification

Before publishing changes, check Python syntax and start the dashboard:

```bash
python -m py_compile app.py
streamlit run app.py
```
## Output

<img width="1380" height="752" alt="image" src="https://github.com/user-attachments/assets/4d1764d0-13eb-48f9-a0e7-a60e15897dd5" />
<img width="1380" height="752" alt="image" src="https://github.com/user-attachments/assets/1918e74c-a958-4a95-be11-5adb423efe41" />


