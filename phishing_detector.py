import re
import pandas as pd

# 1. Load data
df = pd.read_csv("phishing_email.csv")
df = df.rename(columns={"text_combined": "text"})
df = df.dropna()

# Use 20,000 emails so it runs fast on a laptop
df = df.sample(20000, random_state=42)
print("Emails used:", df.shape[0])

# 2. Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"\d+", " numtoken ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean"] = df["text"].apply(clean_text)

# 3. Compare before and after
print("\nBEFORE:", df["text"].iloc[0][:200])
print("\nAFTER:", df["clean"].iloc[0][:200])
# 4. Split into train and test
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# 5. TF-IDF: turn text into numbers
vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

# 6. Train and evaluate two models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}
for name, model in models.items():
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    print("\n=====", name, "=====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, target_names=["Legit", "Phishing"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    import joblib
best = models["Logistic Regression"]
joblib.dump((vec, best), "phishing_model.pkl")

def predict(email):
    X = vec.transform([clean_text(email)])
    prob = best.predict_proba(X)[0][1]
    label = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    return label, prob

while True:
    msg = input("\nPaste email text (or quit): ")
    if msg.lower() == "quit":
        break
    label, prob = predict(msg)
    print(f"Result: {label} (phishing probability: {prob:.2%})")
    