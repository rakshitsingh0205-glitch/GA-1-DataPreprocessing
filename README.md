# Feature Engineering Capstone — StaySmart Hotels

**Course:** Data Science with Machine Learning  
**Dataset:** Hotel Bookings (119,390 rows × 32 columns)  
**Target:** `is_canceled` (binary classification)

---

## Repository Structure

```
FeatureEngineering_Capstone/
├── FeatureEngineering_Capstone.ipynb   ← Main notebook (all 8 tasks)
├── requirements.txt
├── README.md
├── src/
│   ├── feature_helpers.py              ← Reusable feature engineering functions
│   └── pipelines.py                    ← Modular sklearn pipelines
└── report/
    └── Report.pdf                      ← Final written report with all figures
```

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/FeatureEngineering_Capstone.git
cd FeatureEngineering_Capstone

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook FeatureEngineering_Capstone.ipynb
```

The notebook downloads the dataset automatically from a public URL — no manual download required.

---

## Task Coverage

| Task | Description | Status |
|------|-------------|--------|
| 1 | Baseline model + What is a Feature? | ✅ |
| 2 | Curse of Dimensionality demo | ✅ |
| 3 | Numeric preprocessing (binning, binarization, scaling) | ✅ |
| 4 | KNN distance metrics & scaling impact | ✅ |
| 5 | End-to-end sklearn Pipeline | ✅ |
| 6 | Feature extraction (dates, text, encoding) | ✅ |
| 7 | Feature construction (8+ features, leakage section) | ✅ |
| 8 | Feature importance + selection | ✅ |
| Final | Before vs After comparison table + Executive Summary | ✅ |
