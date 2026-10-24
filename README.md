# Phishing Email Detector

An NLP project that classifies emails as **Legitimate** or **Phishing** using TF-IDF features with Naive Bayes and Logistic Regression.

## How it works

Raw email text → Cleaning → TF-IDF → Model → Prediction

1. **Cleaning:** lowercase; replace links, email addresses and numbers with `urltoken`, `emailtoken`, `numtoken`; remove symbols
2. **TF-IDF:** English stop words removed, unigrams + bigrams, top 5000 features
3. **Models:** Naive Bayes and Logistic Regression
4. **Prediction:** an interactive prompt returns the phishing probability (threshold 50%)

## Experiment setup

| Item | Value |
|---|---|
| Emails used | 20,000 (random sample, `random_state=42`) |
| Train / Test split | 80% / 20% (16,000 / 4,000), stratified |
| Test set | 1,920 legitimate, 2,080 phishing |

## Results

### Model comparison

| Model | Accuracy |
|---|---|
| Naive Bayes | 95.60% |
| **Logistic Regression** | **97.35%** |

### Naive Bayes

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Legit | 0.93 | 0.98 | 0.96 | 1920 |
| Phishing | 0.98 | 0.94 | 0.96 | 2080 |

Confusion matrix:

| | Predicted Legit | Predicted Phishing |
|---|---|---|
| **Actual Legit** | 1876 | 44 |
| **Actual Phishing** | 132 | 1948 |

### Logistic Regression (best model, saved to `phishing_model.pkl`)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Legit | 0.98 | 0.97 | 0.97 | 1920 |
| Phishing | 0.97 | 0.98 | 0.97 | 2080 |

Confusion matrix:

| | Predicted Legit | Predicted Phishing |
|---|---|---|
| **Actual Legit** | 1861 | 59 |
| **Actual Phishing** | 47 | 2033 |

**Key observation:** Naive Bayes missed 132 phishing emails, while Logistic Regression missed only 47. Missing a phishing email is the more dangerous error, so Logistic Regression is the better choice.

### Live prediction examples

```
Paste email text (or quit): Dear customer, your bank account has been suspended due to unusual activity. Verify your password immediately by clicking the link below http://secure-bank-login.com or your account will be permanently closed.
Result: PHISHING (phishing probability: 95.59%)

Paste email text (or quit): Hi team, please find attached the minutes of yesterday's meeting. Let me know if I missed anything. The project review is scheduled for Monday at 10 am in the seminar hall.
Result: LEGITIMATE (phishing probability: 2.39%)
```

## Dataset

This project expects a file named `phishing_email.csv` in the same folder as `phishing_detector.py`. The dataset is not included in this repository. Obtain it from the dataset provider and place it beside the script before running. The file needs a `text_combined` column and a `label` column (0 = legit, 1 = phishing).

## Setup and run

```bash
pip install pandas scikit-learn joblib
python phishing_detector.py
```

Type or paste an email when prompted, and type `quit` to exit.

## Limitations

- Looks only at word patterns, not URLs, sender address or attachments
- Trained on 20,000 emails from a single dataset, so new scam styles may fool it
- 47 phishing emails and 59 legitimate emails were still misclassified

## Future work

- Train on the full dataset
- Try deep learning models (LSTM, BERT)
- Add URL and sender-domain features
- Tune hyperparameters
