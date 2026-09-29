import joblib
import numpy as np

from feature_extraction import FEATURE_NAMES

clf = joblib.load("classifier.joblib")
vectorizer = joblib.load("vectorizer.joblib")

vocab = vectorizer.get_feature_names_out()
importances = clf.feature_importances_

n_tfidf = len(vocab)
tfidf_importances = importances[:n_tfidf]
stylo_importances = importances[n_tfidf:]

print("Top 20 TF-IDF words driving the model:")
top_idx = np.argsort(tfidf_importances)[::-1][:20]
for i in top_idx:
    print(f"  {vocab[i]}: {tfidf_importances[i]:.4f}")

print("\nStylometric feature importances:")
for name, imp in zip(FEATURE_NAMES, stylo_importances):
    print(f"  {name}: {imp:.4f}")

total_tfidf = tfidf_importances.sum()
total_stylo = stylo_importances.sum()
print(f"\nTotal TF-IDF importance: {total_tfidf:.4f} ({total_tfidf * 100:.1f}%)")
print(f"Total stylometric importance: {total_stylo:.4f} ({total_stylo * 100:.1f}%)")