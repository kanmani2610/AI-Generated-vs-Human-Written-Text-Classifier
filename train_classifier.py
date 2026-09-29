import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from feature_extraction import extract_features, FEATURE_NAMES, FUNCTION_WORDS


def build_feature_matrix(texts, tfidf_vectorizer, fit=False):
    stylo = np.array([list(extract_features(t).values()) for t in texts])

    if fit:
        tfidf = tfidf_vectorizer.fit_transform(texts)
    else:
        tfidf = tfidf_vectorizer.transform(texts)

    stylo_sparse = csr_matrix(stylo)
    combined = hstack([tfidf, stylo_sparse])
    return combined


def train(data_path, model_out, vectorizer_out, text_col="text", label_col="label"):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[text_col, label_col])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df[text_col].astype(str).tolist(),
        df[label_col].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df[label_col],
    )

    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=sorted(FUNCTION_WORDS), lowercase=True, sublinear_tf=True
    )

    X_train = build_feature_matrix(X_train_text, tfidf_vectorizer, fit=True)
    X_test = build_feature_matrix(X_test_text, tfidf_vectorizer, fit=False)

    stylo_train = np.array([list(extract_features(t).values()) for t in X_train_text])
    feature_stats = {
        name: (float(np.mean(stylo_train[:, i])), float(np.std(stylo_train[:, i]) + 1e-9))
        for i, name in enumerate(FEATURE_NAMES)
    }
    joblib.dump(feature_stats, "feature_stats.joblib")

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(clf, model_out)
    joblib.dump(tfidf_vectorizer, vectorizer_out)
    print(f"Saved model to {model_out}")
    print(f"Saved vectorizer to {vectorizer_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with text,label columns")
    parser.add_argument("--model_out", default="classifier.joblib")
    parser.add_argument("--vectorizer_out", default="vectorizer.joblib")
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    args = parser.parse_args()

    train(args.data, args.model_out, args.vectorizer_out, args.text_col, args.label_col)