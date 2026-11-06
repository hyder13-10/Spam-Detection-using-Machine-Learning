# 📧 Spam Detection using Machine Learning

A mini project for *Principles of AI* that classifies messages as **spam** or **ham (not spam)** using TF-IDF and classical ML models, with a Streamlit web app for live testing.

## Dataset
[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI / Kaggle): 5,572 messages, 4,825 ham and 747 spam (imbalanced).

## Approach
1. **Load and tidy**: read with `latin-1` encoding, keep the label and text columns
2. **Clean text**: lowercase, remove URLs, remove non-letters, remove stopwords (NLTK)
3. **Split**: 80/20 stratified train/test split
4. **Vectorize**: TF-IDF (fit on the training set only, to avoid data leakage)
5. **Train and compare**: Naive Bayes, Logistic Regression, LinearSVC, Random Forest, Gradient Boosting, AdaBoost
6. **Evaluate**: precision, recall and F1 on the spam class, not just accuracy
7. **Validate**: 5-fold cross-validation using a Pipeline

## Results

| Model | Accuracy | Spam Precision | Spam Recall | Spam F1 |
|---|---|---|---|---|
| **MNB (alpha=0.1)** | **0.983** | 0.965 | **0.907** | **0.935** |
| LinearSVC | 0.978 | 0.970 | 0.867 | 0.915 |
| Random Forest | 0.977 | 1.000 | 0.827 | 0.905 |
| MNB (alpha=1) | 0.966 | 1.000 | 0.747 | 0.855 |
| Gradient Boosting | 0.962 | 0.974 | 0.740 | 0.841 |
| Logistic Regression | 0.954 | 0.954 | 0.693 | 0.803 |
| AdaBoost | 0.940 | 0.919 | 0.607 | 0.731 |

**Best model: Multinomial Naive Bayes (alpha=0.1)**, with a 5-fold CV F1 of **0.927**.

**Key observations**
- Naive Bayes outperformed boosting models because it suits small, sparse text data.
- Accuracy alone is misleading on imbalanced data, so the project focuses on spam recall and F1.
- Trade-off: MNB (alpha=0.1) catches more spam but wrongly flags a few legitimate messages. Random Forest and default MNB had zero false positives but missed more spam.

## Limitations
The model is trained on **SMS-style messages**, so it generalizes poorly to modern email-style text. In my manual tests, a formal inheritance scam slipped through and a genuine subscription receipt was flagged as spam. Future work: add email datasets (Enron / SpamAssassin) and retrain.

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `spam_detection.ipynb`: full analysis and model comparison
- `app.py`: Streamlit web app
- `spam_model.pkl`: trained model (TF-IDF + MNB pipeline)
- `spam.csv`: dataset
- `requirements.txt`: dependencies

## Screenshot
<img width="1917" height="1078" alt="image" src="https://github.com/user-attachments/assets/72cde576-2897-4475-acae-aad3ef83a561" />


## Author
Hyder, B.E. CSE (AI/ML), Sapthagiri NPS University
