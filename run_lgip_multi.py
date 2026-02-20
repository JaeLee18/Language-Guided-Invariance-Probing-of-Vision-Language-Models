import os
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import open_clip
from transformers import AutoModel, AutoProcessor


# =========================================
# Config
# =========================================

_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(_DIR, "lgip_coco_native5_train2017_with_perturb.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEXT_BATCH_SIZE = 64

MODELS = [
    # ======================
    # 1) Classic / Baseline CLIP (OpenAI)
    # ======================
    {
        "name": "openclip_ViT-B-16_openai",
        "type": "openclip",
        "model_name": "ViT-B-16",
        "pretrained": "openai",
    },
    {
        "name": "openclip_ViT-L-14_openai",
        "type": "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "openai",
    },

    # ======================
    # 2) Large OpenCLIP (LAION-2B)
    # ======================
    {
        "name": "openclip_ViT-L-14_laion2b_s32b_b82k",
        "type": "openclip",
        "model_name": "ViT-L-14",
        "pretrained": "laion2b_s32b_b82k",
    },
    {
        "name": "openclip_ViT-H-14_laion2b_s32b_b79k",
        "type": "openclip",
        "model_name": "ViT-H-14",
        "pretrained": "laion2b_s32b_b79k",
    },

    # ======================
    # 3) EVA02-CLIP
    # ======================
    {
        "name": "openclip_EVA02-L-14_merged2b_s4b_b131k",
        "type": "openclip",
        "model_name": "EVA02-L-14",
        "pretrained": "merged2b_s4b_b131k",
    },

    # ======================
    # 4) SigLIP (Google)
    # ======================
    {
        "name": "siglip_google_siglip-base-patch16-224",
        "type": "siglip",
        "model_id": "google/siglip-base-patch16-224",
    },
    {
        "name": "siglip_google_siglip-base-patch16-384",
        "type": "siglip",
        "model_id": "google/siglip-base-patch16-384",
    },
    {
        "name": "siglip_google_siglip-large-patch16-384",
        "type": "siglip",
        "model_id": "google/siglip-large-patch16-384",
    },

    # ======================
    # 5) SigLIP2
    # ======================
    {
        "name": "siglip2_google_siglip2-base-patch16-224",
        "type": "siglip",
        "model_id": "google/siglip2-base-patch16-224",
    },
]



# =========================================
# Common interface
# =========================================

class BaseVLM:
    def encode_image(self, image_path: str) -> torch.Tensor:
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError


# =========================================
# OpenCLIP (CLIP / EVA02-CLIP) wrapper
# =========================================

class OpenClipVLM(BaseVLM):
    def __init__(self, model_name: str, pretrained: str, device: str = "cuda"):
        print(f"[INFO] Loading OpenCLIP model={model_name}, pretrained={pretrained} on {device}")
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)  # (1, 3, H, W)
        feat = self.model.encode_image(img_t)                       # (1, D)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0]                                              # (D,)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0)
        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.encode_text(tokens)                  # (B, D)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
        all_feats = torch.cat(all_feats, dim=0)                     # (N, D)
        return all_feats


# =========================================
# SigLIP (HuggingFace transformers) wrapper
# =========================================

class SigLipVLM(BaseVLM):
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
        outputs = self.model.get_image_features(**inputs)  # (1, D)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs[0]                                  # (D,)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0)
        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            inputs = self.processor(text=batch, padding=True, truncation=True,
                                   return_tensors="pt").to(self.device)
            outputs = self.model.get_text_features(**inputs)  # (B, D)
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            all_feats.append(outputs)
        all_feats = torch.cat(all_feats, dim=0)               # (N, D)
        return all_feats


# =========================================
# Utils / Metrics
# =========================================

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def cosine_sim_single_to_many(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    vec: (D,)
    mat: (N, D)
    return: (N,)  # vec vs mat[i] cosine similarity
    """
    if mat.numel() == 0:
        return torch.empty(0, device=vec.device)
    vec = vec.unsqueeze(0)  # (1, D)
    return F.cosine_similarity(vec, mat, dim=-1)


def compute_lgip_metrics(vlm: BaseVLM, jsonl_path: str, model_tag: str) -> Dict[str, Any]:
    """
    Read JSONL with same_caps / diff_caps and compute
    invariance_error, sensitivity_score, sensitivity_pos_rate (LGIP metrics).
    """

    invariance_diffs = []      # |base - same| for same-meaning pairs
    sensitivity_gaps = []      # base - diff for diff-meaning pairs
    sensitivity_correct = 0    # count where base > diff
    sensitivity_total = 0

    num_samples = 0

    for sample in tqdm(load_jsonl(jsonl_path), desc=f"LGIP [{model_tag}]", total=None):
        image_path = sample["image_path"]
        orig_captions = sample["orig_captions"]  # list[str]
        same_caps = sample.get("same_caps", {})  # dict[str, list[str]]
        diff_caps = sample.get("diff_caps", {})  # dict[str, list[str]]

        # Image feature
        try:
            img_feat = vlm.encode_image(image_path)  # (D,)
        except Exception as e:
            print(f"[WARN] [{model_tag}] Failed to load/encode image {image_path}: {e}")
            continue

        # Original caption features
        try:
            orig_feats = vlm.encode_text(orig_captions)  # (5, D)
        except Exception as e:
            print(f"[WARN] [{model_tag}] Failed to encode orig captions for image {image_path}: {e}")
            continue

        if orig_feats.numel() == 0:
            continue

        for idx, base_feat in enumerate(orig_feats):
            base_score = F.cosine_similarity(
                img_feat.unsqueeze(0),
                base_feat.unsqueeze(0),
                dim=-1
            )[0]  # scalar

            key = str(idx)

            # 1) invariance: same_caps (same meaning)
            caps_same = same_caps.get(key, [])
            if caps_same:
                same_feats = vlm.encode_text(caps_same)  # (K, D)
                same_scores = cosine_sim_single_to_many(img_feat, same_feats)  # (K,)
                diffs = (base_score - same_scores).abs()  # (K,)
                invariance_diffs.extend(diffs.cpu().tolist())

            # 2) sensitivity: diff_caps (different meaning)
            caps_diff = diff_caps.get(key, [])
            if caps_diff:
                diff_feats = vlm.encode_text(caps_diff)  # (K, D)
                diff_scores = cosine_sim_single_to_many(img_feat, diff_feats)  # (K,)
                gaps = base_score - diff_scores  # (K,)
                sensitivity_gaps.extend(gaps.cpu().tolist())

                correct_mask = (gaps > 0).cpu()
                sensitivity_correct += int(correct_mask.sum().item())
                sensitivity_total += int(correct_mask.numel())

        num_samples += 1

    # Aggregate metrics
    if len(invariance_diffs) == 0:
        invariance_error = None
    else:
        invariance_error = float(sum(invariance_diffs) / len(invariance_diffs))

    if len(sensitivity_gaps) == 0:
        sensitivity_mean = None
    else:
        sensitivity_mean = float(sum(sensitivity_gaps) / len(sensitivity_gaps))

    if sensitivity_total == 0:
        sensitivity_pos_rate = None
    else:
        sensitivity_pos_rate = sensitivity_correct / sensitivity_total

    metrics = {
        "model_tag": model_tag,
        "num_samples": num_samples,
        "num_invariance_pairs": len(invariance_diffs),
        "num_sensitivity_pairs": len(sensitivity_gaps),
        "invariance_error_mean": invariance_error,         # lower is better
        "sensitivity_gap_mean": sensitivity_mean,          # higher is better
        "sensitivity_pos_rate": sensitivity_pos_rate,      # base > diff ratio
    }

    return metrics


# =========================================
# main: evaluate multiple models
# =========================================

def main():
    all_metrics = []

    for cfg in MODELS:
        model_tag = cfg["name"]
        mtype = cfg["type"]

        if mtype == "openclip":
            vlm = OpenClipVLM(
                model_name=cfg["model_name"],
                pretrained=cfg["pretrained"],
                device=DEVICE,
            )
        elif mtype == "siglip":
            vlm = SigLipVLM(
                model_id=cfg["model_id"],
                device=DEVICE,
            )
        else:
            print(f"[WARN] Unknown model type: {mtype}, skip")
            continue

        metrics = compute_lgip_metrics(vlm, JSONL_PATH, model_tag)
        all_metrics.append(metrics)

        print("\n=== LGIP Metrics [{}] ===".format(model_tag))
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("\n")

        # Save per-model metrics
        base = os.path.splitext(JSONL_PATH)[0]
        out_path = base + f"__LGIP_{model_tag}.metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Saved metrics for {model_tag} to: {out_path}")

        del vlm
        torch.cuda.empty_cache()

    # Save combined summary
    base = os.path.splitext(JSONL_PATH)[0]
    summary_path = base + "__LGIP_ALL.metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[INFO] Saved all metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()
