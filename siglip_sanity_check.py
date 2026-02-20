import os
import json
import random
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from PIL import Image
from transformers import AutoModel, AutoProcessor

# =========================
# Config
# =========================

_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(_DIR, "lgip_coco_native5_train2017_with_perturb.jsonl")

MODEL_ID = "google/siglip-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EXAMPLES = 10
MAX_FLIPS_PER_CAP = 5

random.seed(42)


# =========================
# SigLIP wrapper
# =========================

class SigLipVLM:
    def __init__(self, model_id: str, device: str = "cuda"):
        print(f"[INFO] Loading SigLIP model_id={model_id} on {device}")
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)  # (1, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0]  # (D,)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True,
                                return_tensors="pt").to(self.device)
        feats = self.model.get_text_features(**inputs)  # (N, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats  # (N, D)


# =========================
# Utils
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """a: (D,), b: (D,)"""
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)[0])


# =========================
# main: sanity check
# =========================

def main():
    print("[INFO] Loading JSONL...")
    samples = load_jsonl(JSONL_PATH)
    print(f"[INFO] Loaded {len(samples)} samples")

    vlm = SigLipVLM(MODEL_ID, DEVICE)

    # Select only samples that have at least one diff caption
    valid_samples = []
    for s in samples:
        diff_caps = s.get("diff_caps", {})
        has_diff = any(len(v) > 0 for v in diff_caps.values())
        if has_diff:
            valid_samples.append(s)

    print(f"[INFO] Samples with at least one diff caption: {len(valid_samples)}")

    random.shuffle(valid_samples)
    selected = valid_samples[:NUM_EXAMPLES]

    for idx_sample, sample in enumerate(selected):
        image_path = sample["image_path"]
        orig_captions = sample["orig_captions"]
        same_caps = sample.get("same_caps", {})
        diff_caps = sample.get("diff_caps", {})

        print("\n" + "=" * 80)
        print(f"[Sample {idx_sample+1}] image_path: {image_path}")

        try:
            img_feat = vlm.encode_image(image_path)
        except Exception as e:
            print(f"[WARN] Failed to load/encode image: {e}")
            continue

        # Pick a caption index that has diff_caps
        candidate_indices = [i for i, _ in enumerate(orig_captions)
                             if len(diff_caps.get(str(i), [])) > 0]
        if not candidate_indices:
            print("[INFO] No diff_caps for this sample, skip.")
            continue

        cap_idx = random.choice(candidate_indices)
        base_cap = orig_captions[cap_idx]
        flips = diff_caps[str(cap_idx)]

        if len(flips) > MAX_FLIPS_PER_CAP:
            flips = random.sample(flips, MAX_FLIPS_PER_CAP)

        # Encode original + flipped captions
        texts = [base_cap] + flips
        text_feats = vlm.encode_text(texts)  # (1 + K, D)
        base_feat = text_feats[0]
        flip_feats = text_feats[1:]

        base_sim = cosine_sim(img_feat, base_feat)

        print(f"\n[Original caption index {cap_idx}]")
        print(f"  ORIG: {base_cap}")
        print(f"  sim(image, ORIG) = {base_sim:.4f}")

        print("\n  Flipped captions:")
        for flip_text, flip_feat in zip(flips, flip_feats):
            flip_sim = cosine_sim(img_feat, flip_feat)
            delta = flip_sim - base_sim
            direction = "+" if delta > 0 else "-"
            print(f"    - FLIP: {flip_text}")
            print(f"      sim(image, FLIP) = {flip_sim:.4f}  ({direction}{abs(delta):.4f} vs ORIG)")


if __name__ == "__main__":
    main()
