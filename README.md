# 🛒 Will This Visitor Buy? — Purchase Intent Predictor

Most online store visitors leave without buying. This project predicts, from a browsing session, whether a visitor is likely to purchase, so the store can show a discount only to visitors who need it.

## 🔗 Live demo
**[Try the app](https://shopper-inten.streamlit.app/)**

![App screenshot](screenshot.png)

## Results (test set)

| Metric | Baseline (always "no buy") | Best model (Logistic Regression) |
|---|---|---|
| Accuracy | 0.844 | 0.880 |
| Precision | 0.000 | 0.593 |
| Recall | 0.000 | 0.734 |
| F1 | 0.000 | 0.657 |
| ROC-AUC | 0.500 | 0.913 |

**Key lesson:** accuracy was misleading. A model that always predicts "no buy" is 84% accurate but catches zero buyers. Recall was the metric that mattered.

## What I did
- Cleaned the data (removed 125 duplicate rows) and split train/test before any fitting to avoid leakage
- Built a scikit-learn Pipeline: scaling for numeric columns, one-hot encoding for categorical ones
- Compared a baseline with several Logistic Regression variants (class weighting, L1/L2 regularization)
- Tuned hyperparameters with 5-fold cross-validation on the training set only
- Chose the decision threshold on the training data, then evaluated once on the untouched test set
- Deployed a Streamlit app that turns the probability into an action: no action / show coupon / show chat

## Limitations
- The data comes from one online store over one year, so results may not transfer to other stores
- Very unusual inputs (e.g. zero pages viewed) are not handled reliably
- The coupon rule is a simple probability cut-off, not a measured business ROI

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468) — UCI Machine Learning Repository.

## Tools
Python, pandas, scikit-learn, Streamlit
