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
OUTPUT_JSONL = os.path.join(_DIR, "rebuttal_enhanced_lgip_coco_native5_train2017_with_fliptypes.jsonl")

random.seed(42)

MAX_SAME_PER_CAP = 6
MAX_DIFF_PER_CAP = 6
MAX_COMBINED_PER_CAP = 6  # For combined transformations

COLORS = [
    "red", "blue", "green", "yellow", "black",
    "white", "brown", "gray", "orange", "pink", "purple"
]

NUMBER_WORDS = ["one", "two", "three", "four", "five"]

OBJECTS = [
    "dog", "cat", "horse", "car", "bus", "train",
    "person", "bird", "boat", "bicycle", "truck"
]

# Synonym dictionary for advanced paraphrasing
SYNONYM_DICT = {
    'cat': ['feline', 'kitty'],
    'dog': ['canine', 'puppy', 'hound'],
    'person': ['individual', 'human', 'man', 'woman'],
    'sits': ['rests', 'perches'],
    'stands': ['stands', 'poses'],
    'large': ['big', 'huge', 'enormous'],
    'small': ['little', 'tiny', 'miniature'],
    'red': ['crimson', 'scarlet'],
    'blue': ['azure', 'navy'],
    'green': ['emerald', 'lime'],
}

# =========================
# Advanced Paraphrase functions
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


def passive_voice_paraphrases(caption: str):
    """
    Convert active voice sentences to passive voice where possible
    """
    paraphrases = []

    # Simple rule-based passive voice conversion
    # Pattern: Subject + verb + object -> Object + be + past_participle + by + subject
    patterns = [
        # "A cat sits on a computer" -> "A computer has a cat sitting on it"
        (r'^A (\w+) (is|are|sits|sitting|stands|standing|lies|lying) (on|in|at|by|next to|behind|in front of) (.+)\.$',
         r'A \4 \2 \3 \1.'),
        # "cat sits on computer" -> "computer has cat sitting on it"
        (r'^(\w+) (is|are|sits|sitting|stands|standing|lies|lying) (on|in|at|by|next to|behind|in front of) (.+)\.$',
         r'\4 \2 \3 \1.'),
        # "A train is going down the tracks" -> "The tracks have a train going down them"
        (r'^A (\w+) (is|are) (going|running|moving) (down|along|through) (.+)\.$',
         r'\5 \2 \3 \1 \4 them.'),
    ]

    for pattern, replacement in patterns:
        match = re.match(pattern, caption.strip(), re.IGNORECASE)
        if match:
            passive = re.sub(pattern, replacement, caption.strip(), flags=re.IGNORECASE)
            if passive != caption.strip():
                paraphrases.append(passive.capitalize())

    return paraphrases


def synonym_based_paraphrases(caption: str):
    """
    Replace words with synonyms while preserving meaning
    """
    paraphrases = []
    words = caption.lower().split()

    # Try substituting 1-2 words with synonyms
    for i, word in enumerate(words):
        if word in SYNONYM_DICT:
            for synonym in SYNONYM_DICT[word]:
                new_words = words.copy()
                new_words[i] = synonym
                new_caption = ' '.join(new_words).capitalize()
                if new_caption != caption:
                    paraphrases.append(new_caption)

    # Try substituting two words if possible
    for i, word1 in enumerate(words):
        if word1 in SYNONYM_DICT:
            for j, word2 in enumerate(words[i+1:], i+1):
                if word2 in SYNONYM_DICT:
                    for syn1 in SYNONYM_DICT[word1]:
                        for syn2 in SYNONYM_DICT[word2]:
                            new_words = words.copy()
                            new_words[i] = syn1
                            new_words[j] = syn2
                            new_caption = ' '.join(new_words).capitalize()
                            if new_caption != caption:
                                paraphrases.append(new_caption)

    return paraphrases


def advanced_paraphrases(caption: str):
    """
    Combine multiple paraphrasing techniques
    """
    all_paraphrases = []

    # Existing template-based paraphrases
    all_paraphrases.extend(template_paraphrases(caption))
    all_paraphrases.extend(simple_prefix_paraphrases(caption))

    # Add passive voice paraphrases
    all_paraphrases.extend(passive_voice_paraphrases(caption))

    # Add synonym-based paraphrases
    all_paraphrases.extend(synonym_based_paraphrases(caption))

    # Remove duplicates and filter
    unique_paraphrases = list(set(p.strip() for p in all_paraphrases))
    filtered = [p for p in unique_paraphrases if len(p) >= 5 and p.lower() != caption.lower()]

    return filtered


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
# Combined Transformations (Paraphrase + Flip)
# =========================

def generate_combined_transformations(caption: str):
    """
    Generate transformations that combine paraphrase + semantic flip
    Returns list of dicts with 'text', 'combined_type', and 'intermediate_para'
    """
    combined_transformations = []

    # First, generate paraphrases of the original caption
    paraphrases = advanced_paraphrases(caption)

    # For each paraphrase, apply semantic flips
    for para in paraphrases[:5]:  # Limit to avoid explosion
        flips = rule_based_flips_with_type(para)
        for flip_text, flip_type in flips:
            # Mark these as combined transformations
            combined_transformations.append({
                'text': flip_text,
                'combined_type': f'para+{flip_type}',
                'intermediate_para': para
            })

    return combined_transformations


# =========================
# main
# =========================

def process_enhanced_lgip_with_fliptypes(input_path: str, output_path: str):
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
            combined_caps = {}  # New: combined transformations

            for idx, cap in enumerate(orig_caps):
                # same (advanced paraphrases)
                s_list = advanced_paraphrases(cap)
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

                # combined transformations (paraphrase + flip)
                combined_list = generate_combined_transformations(cap)
                combined_list = combined_list[:MAX_COMBINED_PER_CAP] if len(combined_list) > MAX_COMBINED_PER_CAP else combined_list

                key = str(idx)
                same_caps[key] = s_list
                diff_caps[key] = diff_list_all
                diff_caps_color[key] = c_list
                diff_caps_number[key] = n_list
                diff_caps_object[key] = o_list
                combined_caps[key] = combined_list

            sample["same_caps"] = same_caps
            sample["diff_caps"] = diff_caps
            sample["diff_caps_color"] = diff_caps_color
            sample["diff_caps_number"] = diff_caps_number
            sample["diff_caps_object"] = diff_caps_object
            sample["combined_caps"] = combined_caps  # New field

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processed {total} samples.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    process_enhanced_lgip_with_fliptypes(INPUT_JSONL, OUTPUT_JSONL)
