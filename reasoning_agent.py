import os
import json
import joblib
from scipy.sparse import hstack, csr_matrix
import numpy as np

from feature_extraction import extract_features, FEATURE_NAMES, FEATURE_DESCRIPTIONS, sentence_split


class ReasoningAgent:
    def __init__(self, model_path, vectorizer_path, stats_path="feature_stats.joblib", use_llm=False):
        self.clf = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.feature_stats = joblib.load(stats_path)
        self.use_llm = use_llm

    def _build_vector(self, text):
        feats = extract_features(text)
        stylo = np.array([[feats[name] for name in FEATURE_NAMES]])
        tfidf = self.vectorizer.transform([text])
        combined = hstack([tfidf, csr_matrix(stylo)])
        return combined, feats

    def predict(self, text):
        vec, feats = self._build_vector(text)
        pred = self.clf.predict(vec)[0]
        proba = self.clf.predict_proba(vec)[0]
        confidence = float(np.max(proba))

        importances = self._get_stylo_importance()
        top_signals = self._rank_signals(feats, importances)

        explanation = (
            self._llm_explanation(text, pred, confidence, top_signals)
            if self.use_llm
            else self._rule_based_explanation(pred, confidence, top_signals)
        )

        return {
            "prediction": str(pred),
            "confidence": round(confidence, 4),
            "features": feats,
            "top_signals": top_signals,
            "explanation": explanation,
        }

    def _get_stylo_importance(self):
        n_stylo = len(FEATURE_NAMES)
        full_importances = self.clf.feature_importances_
        stylo_importances = full_importances[-n_stylo:]
        return dict(zip(FEATURE_NAMES, stylo_importances))

    def _rank_signals(self, feats, importances, top_k=3):
        scored = []
        for name in FEATURE_NAMES:
            mean, std = self.feature_stats[name]
            z = (feats[name] - mean) / std
            importance = importances.get(name, 0.0)
            score = importance * abs(z)
            scored.append((name, z, importance, score, mean))

        scored.sort(key=lambda x: x[3], reverse=True)

        signals = []
        for name, z, importance, score, mean in scored[:top_k]:
            direction = "higher" if z > 0 else "lower"
            signals.append({
                "feature": name,
                "value": round(feats[name], 4),
                "typical_value": round(mean, 4),
                "z_score": round(float(z), 2),
                "direction": direction,
                "importance": round(float(importance), 4),
                "reason": (
                    f"{FEATURE_DESCRIPTIONS[name]} is {direction} than typical "
                    f"({feats[name]:.3f} vs avg {mean:.3f})"
                ),
            })
        return signals

    def predict_document(self, text, chunk_size=7, min_chunk_size=3):
        sentences = sentence_split(text)

        chunks = []
        i = 0
        while i < len(sentences):
            chunk_sentences = sentences[i:i + chunk_size]
            if len(chunk_sentences) < min_chunk_size and chunks:
                chunks[-1].extend(chunk_sentences)
            else:
                chunks.append(chunk_sentences)
            i += chunk_size

        chunk_results = []
        ai_count = 0

        for idx, chunk_sentences in enumerate(chunks):
            chunk_text = " ".join(chunk_sentences)
            if len(chunk_text.strip()) < 20:
                continue

            result = self.predict(chunk_text)
            is_ai = result["prediction"].lower() == "ai"
            if is_ai:
                ai_count += 1

            chunk_results.append({
                "chunk_index": idx,
                "sentence_count": len(chunk_sentences),
                "preview": chunk_text[:180],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "top_signals": result["top_signals"],
            })

        total = len(chunk_results)
        ai_fraction = ai_count / total if total > 0 else 0.0

        if ai_fraction >= 0.7:
            overall_verdict = "ai"
        elif ai_fraction <= 0.3:
            overall_verdict = "human"
        else:
            overall_verdict = "mixed"

        return {
            "overall_verdict": overall_verdict,
            "ai_fraction": round(ai_fraction, 4),
            "total_chunks": total,
            "ai_chunks": ai_count,
            "human_chunks": total - ai_count,
            "chunks": chunk_results,
        }

    def _rule_based_explanation(self, pred, confidence, signals):
        verdict = "AI-generated" if str(pred).lower() in ("1", "ai", "true") else "human-written"
        reasons = "; ".join(s["reason"] for s in signals)
        return (
            f"Classified as {verdict} with {confidence * 100:.1f}% confidence. "
            f"Most unusual signals for this text: {reasons}."
        )

    def _llm_explanation(self, text, pred, confidence, signals):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return self._rule_based_explanation(pred, confidence, signals)

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        signal_summary = "\n".join(
            f"- {s['feature']}: {s['value']} ({s['reason']})" for s in signals
        )
        prompt = (
            f"A text classifier predicted this text is '{pred}' with "
            f"{confidence * 100:.1f}% confidence.\n\n"
            f"Contributing linguistic signals:\n{signal_summary}\n\n"
            f"Text excerpt: {text[:500]}\n\n"
            "Write a concise 2-3 sentence explanation for a non-technical reader "
            "justifying this verdict based on the signals above."
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


if __name__ == "__main__":
    import sys

    agent = ReasoningAgent("classifier.joblib", "vectorizer.joblib")
    sample_text = sys.argv[1] if len(sys.argv) > 1 else "Paste your text here."
    result = agent.predict(sample_text)
    print(json.dumps(result, indent=2))