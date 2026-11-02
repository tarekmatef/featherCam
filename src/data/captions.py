"""
Caption splitting, cleaning, typo detection, and NLP statistics for CUB-200-2011.

In Multimodal-Fatima/CUB_train and CUB_test (Reed et al., 2016), each record's
`description` field is a single newline-separated string containing 10 crowdsourced
captions. This module splits, cleans, normalizes, and audits those descriptions.
"""

from collections import Counter
import re
import string
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
import unicodedata

import numpy as np

# Domain-specific typo & UK->US spelling normalization table observed in crowdsourced
# CUB-200-2011 bird descriptions (Reed et al., 2016).
COMMON_CUB_TYPOS: Dict[str, str] = {
    # Bird anatomy typos
    "feathres": "feathers",
    "featheres": "feathers",
    "fethers": "feathers",
    "feahters": "feathers",
    "featers": "feathers",
    "beack": "beak",
    "beek": "beak",
    "baek": "beak",
    "billl": "bill",
    "biil": "bill",
    "brest": "breast",
    "braest": "breast",
    "bresat": "breast",
    "underbely": "underbelly",
    "bely": "belly",
    "belley": "belly",
    "throath": "throat",
    "thoat": "throat",
    "traot": "throat",
    "abdoman": "abdomen",
    "abodmen": "abdomen",
    "eyering": "eye ring",
    "eyebrow": "eyebrow",
    "eyebrows": "eyebrows",
    "wingbars": "wing bars",
    "wingbar": "wing bar",
    "tarsus": "tarsus",
    "rectrices": "rectrices",
    "superciliary": "superciliary",
    # General & color typos + UK/US standardization
    "brid": "bird",
    "bir": "bird",
    "birs": "bird",
    "brd": "bird",
    "teh": "the",
    "hte": "the",
    "wih": "with",
    "wiht": "with",
    "wtih": "with",
    "wit": "with",
    "adn": "and",
    "nad": "and",
    "has": "has",
    "ahve": "have",
    "whie": "white",
    "whte": "white",
    "wihte": "white",
    "blak": "black",
    "balck": "black",
    "blck": "black",
    "brwn": "brown",
    "borwn": "brown",
    "brow": "brown",
    "yello": "yellow",
    "yelow": "yellow",
    "yellowish": "yellowish",
    "ornage": "orange",
    "oragne": "orange",
    "grey": "gray",
    "greyish": "grayish",
    "colour": "color",
    "coloured": "colored",
    "colours": "colors",
    "colred": "colored",
    "coloration": "coloration",
    "smal": "small",
    "samll": "small",
    "meduim": "medium",
    "mediun": "medium",
    "longg": "long",
    "shortt": "short",
    "pointy": "pointy",
    "poined": "pointed",
    "curved": "curved",
    "speckeled": "speckled",
    "stripied": "striped",
    "stiped": "striped",
}

ENGLISH_STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "very",
    "was",
    "which",
    "with",
}

_NON_ALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s\-']")
_HYPHEN_OR_APOSTROPHE_NOISE_RE = re.compile(r"(?:^[\-']+|[\-']+$|\s[\-']+|[\-']+\s)")
_MULTI_SPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_WEIRD_SYMBOL_RE = re.compile(r"[^a-zA-Z0-9\s.,;:'\"?!\-()]")


def split_raw_description(description: Optional[str]) -> Tuple[List[str], int]:
    """
    Split a raw CUB `description` field on newlines into individual captions.

    Args:
        description: Raw newline-separated string holding (typically 10) captions.

    Returns:
        Tuple `(captions, empty_line_count)` where `captions` are stripped non-empty
        caption strings and `empty_line_count` counts blank lines discarded.
    """
    if description is None:
        return [], 0
    text = str(description).replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return [], len(text.split("\n"))

    raw_lines = text.split("\n")
    captions: List[str] = []
    empty_lines = 0
    for line in raw_lines:
        stripped = line.strip()
        if stripped:
            captions.append(stripped)
        else:
            empty_lines += 1
    return captions, empty_lines


def tokenize_raw(caption: str) -> List[str]:
    """Basic whitespace tokenization of a raw caption without cleaning (for raw vocab baseline)."""
    return [tok for tok in caption.strip().split() if tok]


def clean_caption(
    caption: str,
    lowercase: bool = True,
    normalize_punctuation: bool = True,
    fix_typos: bool = True,
    typo_map: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Clean and normalize a single bird caption.

    Preprocessing steps:
      1. Unicode NFKC normalization and whitespace stripping.
      2. Lowercasing (unifies case variants such as 'This'/'this').
      3. Hyphen separation & punctuation removal (prevents token fragmentation such as
         'wings.' vs 'wings' or 'black-and-white' vs 'black and white').
      4. Domain-aware typo and UK->US spelling correction (`COMMON_CUB_TYPOS`).
      5. Whitespace collapsing.

    Returns:
        Tuple `(cleaned_text, replacements)` where `replacements` lists `(old_word, new_word)`
        typo corrections applied to this caption.
    """
    text = unicodedata.normalize("NFKC", str(caption)).strip()
    if lowercase:
        text = text.lower()

    if normalize_punctuation:
        # Replace hyphens/slashes between words with spaces so compound adjectives tokenize cleanly
        text = text.replace("-", " ").replace("/", " ")
        # Strip noisy punctuation/symbols while keeping alphanumeric tokens and apostrophes
        text = _NON_ALLOWED_CHARS_RE.sub(" ", text)
        text = _HYPHEN_OR_APOSTROPHE_NOISE_RE.sub(" ", text)

    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    replacements: List[Tuple[str, str]] = []
    if fix_typos and text:
        active_map = typo_map if typo_map is not None else COMMON_CUB_TYPOS
        cleaned_tokens: List[str] = []
        for tok in text.split(" "):
            bare = tok.strip("'")
            if bare in active_map and active_map[bare] != bare:
                corrected = active_map[bare]
                replacements.append((bare, corrected))
                cleaned_tokens.extend(corrected.split(" "))
            elif bare:
                cleaned_tokens.append(bare)
        text = " ".join(cleaned_tokens)

    return text, replacements


def preprocess_description_column(
    descriptions: Sequence[Optional[str]],
    lowercase: bool = True,
    normalize_punctuation: bool = True,
    fix_typos: bool = True,
    min_tokens: int = 2,
    deduplicate_per_image: bool = True,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    """
    Split and clean the `description` column across an entire dataset split.

    Args:
        descriptions: Sequence of raw newline-separated description strings (one per image).
        lowercase: Convert captions to lowercase.
        normalize_punctuation: Remove punctuation and normalize compounds.
        fix_typos: Apply `COMMON_CUB_TYPOS` corrections.
        min_tokens: Drop degenerate noisy fragments shorter than `min_tokens` words.
        deduplicate_per_image: Remove exact duplicate captions within the same image.

    Returns:
        `(cleaned_captions_per_image, cleaning_audit)`
    """
    all_cleaned: List[List[str]] = []
    typo_counter: Counter = Counter()
    total_raw_captions = 0
    total_empty_lines = 0
    dropped_too_short = 0
    dropped_duplicates = 0

    for desc in descriptions:
        raw_caps, empty_cnt = split_raw_description(desc)
        total_raw_captions += len(raw_caps)
        total_empty_lines += empty_cnt

        image_cleaned: List[str] = []
        seen_in_image: Set[str] = set()

        for raw_c in raw_caps:
            cleaned_c, reps = clean_caption(
                raw_c,
                lowercase=lowercase,
                normalize_punctuation=normalize_punctuation,
                fix_typos=fix_typos,
            )
            for old_w, new_w in reps:
                typo_counter[f"{old_w} -> {new_w}"] += 1

            tokens = cleaned_c.split() if cleaned_c else []
            if len(tokens) < min_tokens:
                dropped_too_short += 1
                continue

            if deduplicate_per_image:
                if cleaned_c in seen_in_image:
                    dropped_duplicates += 1
                    continue
                seen_in_image.add(cleaned_c)

            image_cleaned.append(cleaned_c)

        all_cleaned.append(image_cleaned)

    total_cleaned_captions = sum(len(caps) for caps in all_cleaned)
    audit = {
        "num_records": len(descriptions),
        "total_raw_captions": total_raw_captions,
        "total_cleaned_captions": total_cleaned_captions,
        "empty_lines_removed": total_empty_lines,
        "dropped_too_short": dropped_too_short,
        "dropped_intra_image_duplicates": dropped_duplicates,
        "total_typo_fixes_applied": int(sum(typo_counter.values())),
        "top_typo_fixes": dict(typo_counter.most_common(30)),
    }
    return all_cleaned, audit


def detect_corpus_typos(
    word_counts: Counter,
    max_rare_freq: int = 2,
    min_anchor_freq: int = 30,
    min_word_len: int = 4,
    max_candidates: int = 40,
) -> List[Dict[str, Any]]:
    """
    Algorithmically detect likely remaining misspellings in a corpus vocabulary.

    Identifies rare alphabetic tokens (`count <= max_rare_freq`, `len >= min_word_len`)
    that lie at Damerau-Levenshtein edit distance 1 (deletion, transposition,
    substitution, or insertion) from a high-frequency corpus word (`count >= min_anchor_freq`),
    excluding simple plural/tense suffixes ('s', 'd').
    """
    anchors: Set[str] = {
        w for w, c in word_counts.items() if c >= min_anchor_freq and len(w) >= min_word_len and w.isalpha()
    }
    if not anchors:
        return []

    alphabet = string.ascii_lowercase
    suspected: List[Dict[str, Any]] = []

    for word, freq in word_counts.items():
        if freq > max_rare_freq or len(word) < min_word_len or not word.isalpha() or word in anchors:
            continue

        best_match: Optional[str] = None
        best_anchor_freq = -1

        # 1. Deletions (word has 1 extra char)
        candidates: Set[str] = {word[:i] + word[i + 1 :] for i in range(len(word))}
        # 2. Adjacent transpositions
        candidates.update(
            word[:i] + word[i + 1] + word[i] + word[i + 2 :] for i in range(len(word) - 1)
        )
        # 3. Single-char substitutions
        for i in range(len(word)):
            prefix, suffix = word[:i], word[i + 1 :]
            for ch in alphabet:
                if ch != word[i]:
                    candidates.add(prefix + ch + suffix)
        # 4. Single-char insertions (word is missing 1 char)
        for i in range(len(word) + 1):
            prefix, suffix = word[:i], word[i:]
            for ch in alphabet:
                candidates.add(prefix + ch + suffix)

        for cand in candidates:
            if cand not in anchors:
                continue
            # Ignore regular English inflections like 'feather' <-> 'feathers' or 'curve' <-> 'curved'
            if cand == word + "s" or word == cand + "s" or cand == word + "d" or word == cand + "d":
                continue
            cand_freq = word_counts[cand]
            if cand_freq > best_anchor_freq:
                best_anchor_freq = cand_freq
                best_match = cand

        if best_match is not None:
            suspected.append(
                {
                    "typo": word,
                    "typo_count": int(freq),
                    "suggested_word": best_match,
                    "suggested_count": int(best_anchor_freq),
                }
            )

    suspected.sort(key=lambda item: (-item["typo_count"], -item["suggested_count"], item["typo"]))
    return suspected[:max_candidates]


def analyze_captions(
    descriptions: Sequence[Optional[str]],
    short_threshold: int = 4,
    long_threshold: int = 25,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    """
    Perform comprehensive EDA on a sequence of CUB `description` strings.

    Analyzes raw vs. cleaned caption counts, captions-per-image anomalies, token/char
    length distributions, vocabulary statistics, noisy captions, and corpus typos.

    Returns:
        `(caption_stats_report, cleaned_captions_per_image)`
    """
    raw_captions_per_image: List[List[str]] = []
    raw_vocab: Counter = Counter()
    raw_word_lengths: List[int] = []
    noisy_examples: Dict[str, List[str]] = {
        "very_short": [],
        "very_long": [],
        "contains_digits": [],
        "weird_symbols": [],
        "intra_image_duplicates": [],
    }
    non_ten_caption_records: List[Dict[str, int]] = []

    for idx, desc in enumerate(descriptions):
        caps, _ = split_raw_description(desc)
        raw_captions_per_image.append(caps)
        if len(caps) != 10:
            non_ten_caption_records.append({"record_index": idx, "num_captions": len(caps)})

        seen_raw: Set[str] = set()
        for c in caps:
            toks = tokenize_raw(c)
            raw_vocab.update(toks)
            raw_word_lengths.append(len(toks))

            if len(toks) < short_threshold and len(noisy_examples["very_short"]) < 10:
                noisy_examples["very_short"].append(c)
            if len(toks) > long_threshold and len(noisy_examples["very_long"]) < 10:
                noisy_examples["very_long"].append(c)
            if _DIGIT_RE.search(c) and len(noisy_examples["contains_digits"]) < 10:
                noisy_examples["contains_digits"].append(c)
            if _WEIRD_SYMBOL_RE.search(c) and len(noisy_examples["weird_symbols"]) < 10:
                noisy_examples["weird_symbols"].append(c)
            lowered = c.lower()
            if lowered in seen_raw and len(noisy_examples["intra_image_duplicates"]) < 10:
                noisy_examples["intra_image_duplicates"].append(c)
            seen_raw.add(lowered)

    cleaned_per_image, cleaning_audit = preprocess_description_column(
        descriptions,
        lowercase=True,
        normalize_punctuation=True,
        fix_typos=True,
        min_tokens=2,
        deduplicate_per_image=True,
    )

    cleaned_vocab: Counter = Counter()
    content_vocab: Counter = Counter()
    cleaned_word_lengths: List[int] = []
    cleaned_char_lengths: List[int] = []

    for caps in cleaned_per_image:
        for c in caps:
            toks = c.split()
            cleaned_vocab.update(toks)
            content_vocab.update(t for t in toks if t not in ENGLISH_STOPWORDS)
            cleaned_word_lengths.append(len(toks))
            cleaned_char_lengths.append(len(c))

    # Also inspect pre-typo-fix normalized vocabulary to discover additional corpus typos
    unfixed_vocab: Counter = Counter()
    for caps in raw_captions_per_image:
        for c in caps:
            norm_c, _ = clean_caption(c, lowercase=True, normalize_punctuation=True, fix_typos=False)
            if norm_c:
                unfixed_vocab.update(norm_c.split())

    detected_typos = detect_corpus_typos(unfixed_vocab)

    raw_counts_arr = np.array([len(c) for c in raw_captions_per_image], dtype=np.int32)
    clean_counts_arr = np.array([len(c) for c in cleaned_per_image], dtype=np.int32)
    raw_len_arr = np.array(raw_word_lengths if raw_word_lengths else [0], dtype=np.float64)
    clean_len_arr = np.array(cleaned_word_lengths if cleaned_word_lengths else [0], dtype=np.float64)
    clean_char_arr = np.array(cleaned_char_lengths if cleaned_char_lengths else [0], dtype=np.float64)

    raw_vocab_size = len(raw_vocab)
    clean_vocab_size = len(cleaned_vocab)
    vocab_reduction_pct = (
        100.0 * (raw_vocab_size - clean_vocab_size) / max(raw_vocab_size, 1)
    )
    hapax_count = sum(1 for _, cnt in cleaned_vocab.items() if cnt == 1)

    length_hist_counts = Counter(cleaned_word_lengths)
    captions_per_img_dist = Counter(int(x) for x in raw_counts_arr.tolist())

    report: Dict[str, Any] = {
        "num_records": len(descriptions),
        "captions_per_image": {
            "raw_min": int(raw_counts_arr.min()) if len(raw_counts_arr) else 0,
            "raw_max": int(raw_counts_arr.max()) if len(raw_counts_arr) else 0,
            "raw_mean": float(raw_counts_arr.mean()) if len(raw_counts_arr) else 0.0,
            "cleaned_min": int(clean_counts_arr.min()) if len(clean_counts_arr) else 0,
            "cleaned_max": int(clean_counts_arr.max()) if len(clean_counts_arr) else 0,
            "cleaned_mean": float(clean_counts_arr.mean()) if len(clean_counts_arr) else 0.0,
            "exact_10_raw_records": int((raw_counts_arr == 10).sum()) if len(raw_counts_arr) else 0,
            "non_10_raw_records_count": len(non_ten_caption_records),
            "non_10_raw_records_sample": non_ten_caption_records[:15],
            "raw_distribution": {str(k): int(v) for k, v in sorted(captions_per_img_dist.items())},
        },
        "word_length_stats": {
            "raw_mean": float(raw_len_arr.mean()),
            "raw_std": float(raw_len_arr.std()),
            "raw_min": int(raw_len_arr.min()),
            "raw_max": int(raw_len_arr.max()),
            "cleaned_mean": float(clean_len_arr.mean()),
            "cleaned_median": float(np.median(clean_len_arr)),
            "cleaned_std": float(clean_len_arr.std()),
            "cleaned_min": int(clean_len_arr.min()),
            "cleaned_max": int(clean_len_arr.max()),
            "cleaned_p5": float(np.percentile(clean_len_arr, 5)),
            "cleaned_p95": float(np.percentile(clean_len_arr, 95)),
            "cleaned_char_mean": float(clean_char_arr.mean()),
            "raw_lengths": raw_word_lengths,
            "cleaned_lengths": cleaned_word_lengths,
            "cleaned_length_histogram": {str(k): int(v) for k, v in sorted(length_hist_counts.items())},
        },
        "vocabulary_stats": {
            "raw_vocab_size": raw_vocab_size,
            "cleaned_vocab_size": clean_vocab_size,
            "vocab_reduction_pct": float(vocab_reduction_pct),
            "total_cleaned_tokens": int(sum(cleaned_vocab.values())),
            "hapax_legomena_count": int(hapax_count),
            "top_30_words": dict(cleaned_vocab.most_common(30)),
            "top_30_descriptive_words": dict(content_vocab.most_common(30)),
        },
        "cleaning_audit": cleaning_audit,
        "noisy_caption_audit": {
            "short_caption_count": int(sum(1 for l in raw_word_lengths if l < short_threshold)),
            "long_caption_count": int(sum(1 for l in raw_word_lengths if l > long_threshold)),
            "examples": noisy_examples,
        },
        "suspected_corpus_typos": detected_typos,
    }
    return report, cleaned_per_image

