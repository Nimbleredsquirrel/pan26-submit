import re
import string
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def tokenize_words(txt):
    return re.findall(r"\b\w+\b", txt.lower())


def tokenize_sents(txt):
    sents = re.split(r"(?<=[.!?])\s+", txt.strip())
    return [s for s in sents if s.strip()]


def tokenize_words_raw(txt):
    # preserve case for lowercase ratio
    return re.findall(r"\b\w+\b", txt)


def mttr(words, win=100):
    if len(words) < win:
        return len(set(words)) / len(words) if words else 0.0
    ttrs = []
    for i in range(len(words) - win + 1):
        w = words[i:i + win]
        ttrs.append(len(set(w)) / win)
    return float(np.mean(ttrs))


def hapax_rate(words):
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c == 1) / len(words)


def burstiness(sents):
    if len(sents) < 2:
        return 0.0
    lengths = [len(tokenize_words(s)) for s in sents]
    mu = np.mean(lengths)
    if mu == 0:
        return 0.0
    return float(np.std(lengths) / mu)


def bigram_uniq(words):
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


def verb_ratio(txt):
    words = tokenize_words(txt)
    if not words:
        return 0.0
    aux = {"is", "are", "was", "were", "be", "been", "being",
           "have", "has", "had", "do", "does", "did", "will",
           "would", "could", "should", "may", "might", "shall",
           "can", "must", "get", "got", "make", "made", "said",
           "go", "went", "come", "came", "know", "knew", "think",
           "thought", "see", "saw", "look", "looked", "take", "took"}
    ed_ing = sum(1 for w in words if w.endswith(("ed", "ing")) and len(w) > 4)
    aux_cnt = sum(1 for w in words if w in aux)
    return (ed_ing + aux_cnt) / len(words)


def lowercase_ratio(txt):
    words = tokenize_words_raw(txt)
    if not words:
        return 0.0
    return sum(1 for w in words if w.islower()) / len(words)


def punct_dens(txt):
    if not txt:
        return 0.0
    return sum(1 for c in txt if c in string.punctuation) / len(txt)


def spelling_err_rate(txt):
    words = tokenize_words_raw(txt)
    if not words:
        return 0.0
    errs = sum(1 for w in words if re.search(r"(.)\1{2,}", w))
    return errs / len(words)


def stylometric_ftrs(txt):
    words = tokenize_words(txt)
    sents = tokenize_sents(txt)
    n_words = len(words)
    n_sents = max(len(sents), 1)
    n_chars = len(txt)

    return {
        "n_chars": n_chars,
        "n_wrds": n_words,
        "n_snts": n_sents,
        "avg_wrd_ln": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_snt_ln": n_words / n_sents,
        "ttr": len(set(words)) / n_words if n_words else 0.0,
        "mttr": mttr(words),
        "hapax_rate": hapax_rate(words),
        "burstiness": burstiness(sents),
        "bigram_uniq": bigram_uniq(words),
        "verb_ratio": verb_ratio(txt),
        "lowercase_ratio": lowercase_ratio(txt),
        "punct_dens": punct_dens(txt),
        "spelling_err_rate": spelling_err_rate(txt),
    }


class GLTRFeatures:
    def __init__(self, mdl_nm="gpt2", device=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"loading {mdl_nm} on {device}")
        self.tok = AutoTokenizer.from_pretrained(mdl_nm)
        self.model = AutoModelForCausalLM.from_pretrained(mdl_nm)
        self.model.to(device)
        self.model.eval()

    def get_ftrs(self, txt, max_len=512):
        inputs = self.tok(
            txt, return_tensors="pt", max_length=max_len,
            truncation=True
        ).to(self.device)

        ids = inputs.input_ids
        if ids.shape[1] < 2:
            return self._empty_ftrs()

        with torch.no_grad():
            logits = self.model(**inputs).logits  # (1, L, V)

        log_p = F.log_softmax(logits[0, :-1, :], dim=-1)  # (L-1, V)
        act_ids = ids[0, 1:]  # (L-1,)

        ranks = (log_p > log_p.gather(1, act_ids.unsqueeze(1))).sum(dim=1)
        ranks = ranks.float().cpu().numpy()

        n = len(ranks)
        return {
            "gltr_frac_top10":   float((ranks < 10).sum() / n),
            "gltr_frac_top100":  float((ranks < 100).sum() / n),
            "gltr_frac_top1000": float((ranks < 1000).sum() / n),
            "gltr_frac_tail":    float((ranks >= 1000).sum() / n),
            "gltr_mean_rank":    float(ranks.mean()),
            "gltr_median_rank":  float(np.median(ranks)),
            "gltr_rank_std":     float(ranks.std()),
        }

    def _empty_ftrs(self):
        return {k: 0.0 for k in [
            "gltr_frac_top10", "gltr_frac_top100", "gltr_frac_top1000",
            "gltr_frac_tail", "gltr_mean_rank", "gltr_median_rank", "gltr_rank_std"
        ]}


def get_all_ftrs(txt, gltr=None):
    features = stylometric_ftrs(txt)
    if gltr is not None:
        features.update(gltr.get_ftrs(txt))
    return features


def ftrs_to_vec(features):
    return np.array(list(features.values()), dtype=np.float32)


FTR_NAMES_STYLO = list(stylometric_ftrs("dummy text here").keys())
FTR_NAMES_GLTR = [
    "gltr_frac_top10", "gltr_frac_top100", "gltr_frac_top1000",
    "gltr_frac_tail", "gltr_mean_rank", "gltr_median_rank", "gltr_rank_std"
]
FTR_NAMES_ALL = FTR_NAMES_STYLO + FTR_NAMES_GLTR


# ── Fingerprint features (Tier-1 expansion) ──────────────────────────────────

_RLHF_VOCAB = {
    "delve", "tapestry", "testament", "realm", "vibrant", "nuanced",
    "comprehensive", "pivotal", "intricate", "crucial", "paramount",
    "multifaceted", "underscore", "leverage", "foster", "elevate",
    "transformative", "innovative", "seamless", "dynamic", "holistic",
    "synergy", "endeavor", "facilitate", "navigate", "robust",
    "showcase", "spearhead", "streamline", "optimize", "harness",
}

_SYCOPHANCY = {
    "great question", "excellent question", "certainly", "absolutely",
    "of course", "i'd be happy to", "i'll be happy to",
    "thank you for asking", "i hope this helps", "feel free to ask",
}

_DOWNTONERS = {
    "somewhat", "slightly", "rather", "fairly", "quite",
    "relatively", "moderately", "arguably", "perhaps", "maybe",
}

_SAFETY_PHRASES = {
    "i cannot", "i can't", "i'm unable", "i must note",
    "it's important to note", "as an ai", "as a language model",
    "i should mention", "please note that",
}


def fingerprint_ftrs(txt):
    words = tokenize_words(txt)
    sents = tokenize_sents(txt)
    n_words = max(len(words), 1)
    n_sents = max(len(sents), 1)
    txt_lower = txt.lower()

    # <think> block (DeepSeek-R1 artefact)
    think_block = int(bool(re.search(r"<think>", txt, re.IGNORECASE)))

    # \boxed{} (math model artefact)
    boxed_ans = int("\\boxed{" in txt)

    # sycophantic phrases per sentence
    sycoph_rate = sum(1 for p in _SYCOPHANCY if p in txt_lower) / n_sents

    # safety/refusal preamble in first 300 chars
    intro = txt_lower[:300]
    safety_pre = int(any(p in intro for p in _SAFETY_PHRASES))

    # RLHF vocabulary rate
    rlhf_rate = sum(1 for w in words if w in _RLHF_VOCAB) / n_words

    # downtoner rate (Llama-2 hallmark)
    downtoner_rate = sum(1 for w in words if w in _DOWNTONERS) / n_words

    # em dash (—) per sentence (GPT/OpenAI hallmark)
    em_dash_rate = txt.count("—") / n_sents

    # markdown elements per sentence
    md_hits = len(re.findall(r"#{1,6}\s|\*\*|`|\n[-*]\s", txt))
    md_density = md_hits / n_sents

    # stop-word ratio (AI tends to be leaner)
    _STOP = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
        "or", "but", "is", "are", "was", "were", "be", "been", "being",
        "it", "this", "that", "with", "by", "from", "as", "not", "so",
        "if", "do", "did", "does", "has", "have", "had", "will", "would",
        "could", "should", "may", "might", "can",
    }
    stop_ratio = sum(1 for w in words if w in _STOP) / n_words

    # paragraph-level burstiness (humans vary paragraph length more)
    paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
    if len(paras) >= 2:
        para_lens = [len(tokenize_words(p)) for p in paras]
        mu = float(np.mean(para_lens))
        para_burst = float(np.std(para_lens) / mu) if mu else 0.0
    else:
        para_burst = 0.0

    # suffix-stripped vocabulary ratio (rough stemming, no NLTK)
    _SUFFIXES = ("ing", "tion", "ness", "ment", "er", "ly", "ies", "ed", "es", "s")

    def _stem(w):
        for s in _SUFFIXES:
            if w.endswith(s) and len(w) > len(s) + 3:
                return w[: -len(s)]
        return w

    stems = [_stem(w) for w in words]
    stem_voc_rat = len(set(stems)) / len(stems) if stems else 0.0

    return {
        "think_block":    float(think_block),
        "boxed_ans":      float(boxed_ans),
        "sycoph_rate":    sycoph_rate,
        "safety_pre":     float(safety_pre),
        "rlhf_rate":      rlhf_rate,
        "downtoner_rate": downtoner_rate,
        "em_dash_rate":   em_dash_rate,
        "md_density":     md_density,
        "stop_ratio":     stop_ratio,
        "para_burst":     para_burst,
        "stem_voc_rat":   stem_voc_rat,
    }


FTR_NAMES_FINGERPRINT = list(fingerprint_ftrs("dummy text here").keys())
