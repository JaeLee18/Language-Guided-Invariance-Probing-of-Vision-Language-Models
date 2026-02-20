import os
import json
import re
import random
from collections import OrderedDict

# =========================
# Config
# =========================
_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSONL = os.path.join(_DIR, "lgip_coco_native5_train2017.jsonl")
OUTPUT_JSONL = os.path.join(_DIR, "lgip_coco_native5_train2017_with_perturb.jsonl")

random.seed(42)

MAX_SAME_PER_CAP = 6
MAX_DIFF_PER_CAP = 6

COLORS = [
    "red", "blue", "green", "yellow", "black",
    "white", "brown", "gray", "orange", "pink", "purple"
]

NUMBER_WORDS = ["one", "two", "three", "four", "five"]

OBJECTS = [
    "dog", "cat", "horse", "car", "bus", "train",
    "person", "bird", "boat", "bicycle", "truck"
]


# =========================
# Paraphrase (meaning-preserving) functions
# =========================

def template_paraphrases(caption: str):
    """Template-based paraphrases that preserve meaning while varying style."""
    templates = [
        "a photo of {}",
        "an image of {}",
        "a picture of {}",
        "{}",
        "{} in the scene",
        "a scene showing {}",
    ]
    return [t.format(caption) for t in templates]


def simple_prefix_paraphrases(caption: str):
    """Paraphrases that prepend a short phrase."""
    prefixes = [
        "In this image, {}",
        "In the picture, {}",
        "This image shows {}",
    ]
    return [p.format(caption) for p in prefixes]


def rule_based_paraphrases(caption: str):
    """Combine paraphrase sources and deduplicate."""
    outs = []
    outs += template_paraphrases(caption)
    outs += simple_prefix_paraphrases(caption)

    # Filter out very short results
    filtered = []
    for s in outs:
        t = s.strip()
        if len(t) < 5:
            continue
        filtered.append(t)

    # Deduplicate while preserving order
    uniq = list(OrderedDict.fromkeys(filtered))
    return uniq


# =========================
# Semantic flip (meaning-changing) functions
# =========================

def _regex_flip_one(caption: str, candidates, word_list):
    """Replace one occurrence of a word from word_list with another from candidates."""
    pattern = r"\b(" + "|".join(word_list) + r")\b"
    match = re.search(pattern, caption)
    if not match:
        return None

    orig = match.group(1)
    others = [w for w in candidates if w != orig]
    if not others:
        return None

    new = random.choice(others)
    new_caption = re.sub(pattern, new, caption, count=1)
    return new_caption


def flip_color(caption: str):
    return _regex_flip_one(caption, COLORS, COLORS)


def flip_number(caption: str):
    return _regex_flip_one(caption, NUMBER_WORDS, NUMBER_WORDS)


def flip_object(caption: str):
    return _regex_flip_one(caption, OBJECTS, OBJECTS)


def rule_based_flips(caption: str):
    """Attempt semantic flips by substituting color / number / object words."""
    outs = []

    for fn in [flip_color, flip_number, flip_object]:
        try:
            new_cap = fn(caption)
        except Exception:
            new_cap = None
        if new_cap is not None and new_cap != caption:
            outs.append(new_cap)

    filtered = []
    for s in outs:
        t = s.strip()
        if len(t) < 5:
            continue
        filtered.append(t)

    uniq = list(OrderedDict.fromkeys(filtered))
    return uniq


# =========================
# main
# =========================

def process_lgip_jsonl(input_path: str, output_path: str):
    total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            total += 1

            orig_caps = sample.get("orig_captions", [])
            same_caps = {}
            diff_caps = {}

            for idx, cap in enumerate(orig_caps):
                same_list = rule_based_paraphrases(cap)
                diff_list = rule_based_flips(cap)

                # Remove entries identical to the original
                same_list = [s for s in same_list if s.strip() != cap.strip()]
                diff_list = [d for d in diff_list if d.strip() != cap.strip()]

                # Subsample if too many
                if len(same_list) > MAX_SAME_PER_CAP:
                    same_list = random.sample(same_list, MAX_SAME_PER_CAP)
                if len(diff_list) > MAX_DIFF_PER_CAP:
                    diff_list = random.sample(diff_list, MAX_DIFF_PER_CAP)

                same_caps[str(idx)] = same_list
                diff_caps[str(idx)] = diff_list

            sample["same_caps"] = same_caps
            sample["diff_caps"] = diff_caps

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processed {total} samples.")
    print(f"Saved with perturbations to: {output_path}")


if __name__ == "__main__":
    process_lgip_jsonl(INPUT_JSONL, OUTPUT_JSONL)
