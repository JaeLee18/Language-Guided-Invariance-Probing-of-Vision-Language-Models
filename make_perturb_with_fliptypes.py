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
OUTPUT_JSONL = os.path.join(_DIR, "lgip_coco_native5_train2017_with_fliptypes.jsonl")

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
# Paraphrase (same-meaning) functions
# =========================

def template_paraphrases(caption: str):
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
    prefixes = [
        "In this image, {}",
        "In the picture, {}",
        "This image shows {}",
    ]
    return [p.format(caption) for p in prefixes]


def rule_based_paraphrases(caption: str):
    outs = []
    outs += template_paraphrases(caption)
    outs += simple_prefix_paraphrases(caption)

    filtered = []
    for s in outs:
        t = s.strip()
        if len(t) < 5:
            continue
        filtered.append(t)

    uniq = list(OrderedDict.fromkeys(filtered))
    return uniq

# =========================
# Flip (diff) functions -- returns (text, flip_type)
# =========================

def _regex_flip_one(caption: str, word_list, flip_type: str):
    """Replace one word from word_list with another; return (new_caption, flip_type)."""
    pattern = r"\b(" + "|".join(word_list) + r")\b"
    match = re.search(pattern, caption)
    if not match:
        return None

    orig = match.group(1)
    others = [w for w in word_list if w != orig]
    if not others:
        return None

    new = random.choice(others)
    new_caption = re.sub(pattern, new, caption, count=1)
    return new_caption, flip_type


def flip_color(caption: str):
    return _regex_flip_one(caption, COLORS, "color")


def flip_number(caption: str):
    return _regex_flip_one(caption, NUMBER_WORDS, "number")


def flip_object(caption: str):
    return _regex_flip_one(caption, OBJECTS, "object")


def rule_based_flips_with_type(caption: str):
    """Return list of (flip_text, flip_type) where flip_type in {"color","number","object"}."""
    outs = []

    for fn in [flip_color, flip_number, flip_object]:
        try:
            res = fn(caption)
        except Exception:
            res = None
        if res is None:
            continue
        new_cap, flip_type = res
        if new_cap is not None and new_cap != caption:
            outs.append((new_cap, flip_type))

    # Filter short results and deduplicate
    filtered = []
    for s, t in outs:
        s2 = s.strip()
        if len(s2) < 5:
            continue
        filtered.append((s2, t))

    # Deduplicate by text (keep first occurrence's type)
    uniq_dict = OrderedDict()
    for s, t in filtered:
        if s not in uniq_dict:
            uniq_dict[s] = t

    return list(uniq_dict.items())

# =========================
# main
# =========================

def process_lgip_with_fliptypes(input_path: str, output_path: str):
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
            diff_caps_color = {}
            diff_caps_number = {}
            diff_caps_object = {}

            for idx, cap in enumerate(orig_caps):
                # same (paraphrase)
                s_list = rule_based_paraphrases(cap)
                s_list = [s for s in s_list if s.strip() != cap.strip()]
                if len(s_list) > MAX_SAME_PER_CAP:
                    s_list = random.sample(s_list, MAX_SAME_PER_CAP)

                # diff (flip + type)
                flips = rule_based_flips_with_type(cap)
                flips = [(txt, t) for (txt, t) in flips if txt.strip() != cap.strip()]
                if len(flips) > MAX_DIFF_PER_CAP:
                    flips = random.sample(flips, MAX_DIFF_PER_CAP)

                diff_list_all = [txt for (txt, _) in flips]

                # Split by type
                c_list, n_list, o_list = [], [], []
                for txt, t in flips:
                    if t == "color":
                        c_list.append(txt)
                    elif t == "number":
                        n_list.append(txt)
                    elif t == "object":
                        o_list.append(txt)

                key = str(idx)
                same_caps[key] = s_list
                diff_caps[key] = diff_list_all
                diff_caps_color[key] = c_list
                diff_caps_number[key] = n_list
                diff_caps_object[key] = o_list

            sample["same_caps"] = same_caps
            sample["diff_caps"] = diff_caps
            sample["diff_caps_color"] = diff_caps_color
            sample["diff_caps_number"] = diff_caps_number
            sample["diff_caps_object"] = diff_caps_object

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processed {total} samples.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    process_lgip_with_fliptypes(INPUT_JSONL, OUTPUT_JSONL)
