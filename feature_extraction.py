import re
import numpy as np
from collections import Counter

_gpt2_model = None
_gpt2_tokenizer = None


def _load_gpt2():
    global _gpt2_model, _gpt2_tokenizer
    if _gpt2_model is None:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        _gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        _gpt2_model.eval()
    return _gpt2_model, _gpt2_tokenizer


def compute_perplexity_features(text):
    import torch

    model, tokenizer = _load_gpt2()
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings["input_ids"]

    if input_ids.shape[1] < 2:
        return 0.0, 0.0

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    mean_loss = token_losses.mean().item()
    std_loss = token_losses.std().item() if token_losses.numel() > 1 else 0.0
    perplexity = float(np.exp(min(mean_loss, 20)))
    return perplexity, float(std_loss)

FUNCTION_WORDS = set("""
a an the and or but if while although because since until when where
this that these those is are was were be been being have has had do
does did will would shall should may might must can could of in on
at by for with about against between into through during before after
above below to from up down out off over under again further then
once here there all any both each few more most other some such no
nor not only own same so than too very s t just
""".split())


def sentence_split(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


def word_split(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def burstiness(sentences):
    lengths = [len(word_split(s)) for s in sentences]
    if len(lengths) < 2:
        return 0.0
    mean = np.mean(lengths)
    std = np.std(lengths)
    if mean == 0:
        return 0.0
    return float(std / mean)


def type_token_ratio(words):
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def function_word_ratio(words):
    if not words:
        return 0.0
    count = sum(1 for w in words if w in FUNCTION_WORDS)
    return count / len(words)


def punctuation_ratio(text):
    if not text:
        return 0.0
    punct = sum(1 for c in text if c in ',;:-()"\'')
    return punct / len(text)


def avg_word_length(words):
    if not words:
        return 0.0
    return float(np.mean([len(w) for w in words]))


def char_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text.lower())
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return float(-sum(p * np.log2(p) for p in probs))


def repetition_score(words):
    if len(words) < 4:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(bigrams)


FEATURE_NAMES = [
    "avg_sentence_length",
    "sentence_length_std",
    "burstiness",
    "type_token_ratio",
    "function_word_ratio",
    "punctuation_ratio",
    "avg_word_length",
    "char_entropy",
    "repetition_score",
    "perplexity",
    "perplexity_burstiness",
]

FEATURE_DESCRIPTIONS = {
    "avg_sentence_length": "average sentence length",
    "sentence_length_std": "variation in sentence length",
    "burstiness": "burstiness (sentence-length variability)",
    "type_token_ratio": "vocabulary diversity",
    "function_word_ratio": "function-word usage",
    "punctuation_ratio": "punctuation density",
    "avg_word_length": "average word length",
    "char_entropy": "character-level entropy",
    "repetition_score": "phrase repetition",
    "perplexity": "text predictability to a language model (perplexity)",
    "perplexity_burstiness": "variation in per-word predictability",
}


def extract_features(text):
    sentences = sentence_split(text)
    words = word_split(text)
    sent_lengths = [len(word_split(s)) for s in sentences] or [0]
    perplexity, perplexity_burstiness = compute_perplexity_features(text)

    return {
        "avg_sentence_length": float(np.mean(sent_lengths)),
        "sentence_length_std": float(np.std(sent_lengths)),
        "burstiness": burstiness(sentences),
        "type_token_ratio": type_token_ratio(words),
        "function_word_ratio": function_word_ratio(words),
        "punctuation_ratio": punctuation_ratio(text),
        "avg_word_length": avg_word_length(words),
        "char_entropy": char_entropy(text),
        "repetition_score": repetition_score(words),
        "perplexity": perplexity,
        "perplexity_burstiness": perplexity_burstiness,
    }


def features_to_vector(feat_dict):
    return np.array([feat_dict[name] for name in FEATURE_NAMES])