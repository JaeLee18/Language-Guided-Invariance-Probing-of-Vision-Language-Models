import os
import json
from typing import List, Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip
from transformers import AutoModel, AutoProcessor

_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(_DIR, "lgip_coco_native5_train2017_with_fliptypes.jsonl")

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


# ================ Common interface ================

class BaseVLM:
    def encode_image(self, image_path: str) -> torch.Tensor:
        raise NotImplementedError
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        raise NotImplementedError

# ================ OpenCLIP wrapper ================

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
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img_t)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0]

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0)
        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

# ================ SigLIP wrapper ================

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
        feats = self.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0]

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0)
        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            inputs = self.processor(text=batch, padding=True, truncation=True,
                                   return_tensors="pt").to(self.device)
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats)
        return torch.cat(all_feats, dim=0)

# ================ Utils & Metrics ================

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def cosine_sim_single_to_many(vec: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    if mat.numel() == 0:
        return torch.empty(0, device=vec.device)
    vec = vec.unsqueeze(0)
    return F.cosine_similarity(vec, mat, dim=-1)

def compute_lgip_by_fliptype(vlm: BaseVLM, jsonl_path: str, model_tag: str) -> Dict[str, Any]:
    # Global accumulators
    inv_diffs_global = []
    gap_global = []
    correct_global = 0
    total_global = 0

    # Per-type accumulators
    type_stats = {
        "color": {
            "gaps": [],
            "correct": 0,
            "total": 0,
        },
        "number": {
            "gaps": [],
            "correct": 0,
            "total": 0,
        },
        "object": {
            "gaps": [],
            "correct": 0,
            "total": 0,
        },
    }

    num_samples = 0

    for sample in tqdm(load_jsonl(jsonl_path), desc=f"LGIP flip-type [{model_tag}]"):
        image_path = sample["image_path"]
        orig_caps = sample["orig_captions"]

        same_caps = sample.get("same_caps", {})
        diff_caps = sample.get("diff_caps", {})
        diff_caps_color = sample.get("diff_caps_color", {})
        diff_caps_number = sample.get("diff_caps_number", {})
        diff_caps_object = sample.get("diff_caps_object", {})

        try:
            img_feat = vlm.encode_image(image_path)
        except Exception as e:
            print(f"[WARN] {model_tag}: failed image {image_path}: {e}")
            continue

        try:
            orig_feats = vlm.encode_text(orig_caps)
        except Exception as e:
            print(f"[WARN] {model_tag}: failed text encode: {e}")
            continue

        if orig_feats.numel() == 0:
            continue

        for idx, base_feat in enumerate(orig_feats):
            key = str(idx)
            base_score = F.cosine_similarity(
                img_feat.unsqueeze(0),
                base_feat.unsqueeze(0),
                dim=-1
            )[0]

            # invariance: same (global only)
            caps_same = same_caps.get(key, [])
            if caps_same:
                same_feats = vlm.encode_text(caps_same)
                same_scores = cosine_sim_single_to_many(img_feat, same_feats)
                diffs = (base_score - same_scores).abs()
                inv_diffs_global.extend(diffs.cpu().tolist())

            # sensitivity: global diff
            caps_diff_all = diff_caps.get(key, [])
            if caps_diff_all:
                diff_feats_all = vlm.encode_text(caps_diff_all)
                diff_scores_all = cosine_sim_single_to_many(img_feat, diff_feats_all)
                gaps_all = base_score - diff_scores_all
                gap_global.extend(gaps_all.cpu().tolist())
                correct_global += int((gaps_all > 0).sum().item())
                total_global += int(gaps_all.numel())

            # Per-type sensitivity
            for tname, caps_dict in [
                ("color", diff_caps_color),
                ("number", diff_caps_number),
                ("object", diff_caps_object),
            ]:
                caps_t = caps_dict.get(key, [])
                if not caps_t:
                    continue
                feats_t = vlm.encode_text(caps_t)
                scores_t = cosine_sim_single_to_many(img_feat, feats_t)
                gaps_t = base_score - scores_t
                type_stats[tname]["gaps"].extend(gaps_t.cpu().tolist())
                type_stats[tname]["correct"] += int((gaps_t > 0).sum().item())
                type_stats[tname]["total"] += int(gaps_t.numel())

        num_samples += 1

    # Summarize
    def summarize_gap(gaps: List[float], correct: int, total: int):
        if len(gaps) == 0:
            return {"gap_mean": None, "pos_rate": None, "count": 0}
        gap_mean = float(sum(gaps) / len(gaps))
        pos_rate = None if total == 0 else correct / total
        return {
            "gap_mean": gap_mean,
            "pos_rate": pos_rate,
            "count": len(gaps),
        }

    inv_err = None if len(inv_diffs_global) == 0 else float(sum(inv_diffs_global) / len(inv_diffs_global))
    global_sens = summarize_gap(gap_global, correct_global, total_global)

    out = {
        "model_tag": model_tag,
        "num_samples": num_samples,
        "invariance_error_mean": inv_err,
        "global_sensitivity": global_sens,
        "by_flip_type": {
            t: summarize_gap(stat["gaps"], stat["correct"], stat["total"])
            for t, stat in type_stats.items()
        },
    }
    return out

# ================ main ================

def main():
    all_results = []
    for cfg in MODELS:
        if cfg["type"] == "openclip":
            vlm = OpenClipVLM(cfg["model_name"], cfg["pretrained"], DEVICE)
        elif cfg["type"] == "siglip":
            vlm = SigLipVLM(cfg["model_id"], DEVICE)
        else:
            print(f"Unknown type: {cfg['type']}")
            continue

        res = compute_lgip_by_fliptype(vlm, JSONL_PATH, cfg["name"])
        all_results.append(res)

        print("\n=== Result [{}] ===".format(cfg["name"]))
        print(f"  invariance_error_mean: {res['invariance_error_mean']}")
        print(f"  global_sensitivity: {res['global_sensitivity']}")
        print("  by_flip_type:")
        for t, s in res["by_flip_type"].items():
            print(f"    {t}: gap_mean={s['gap_mean']}, pos_rate={s['pos_rate']}, count={s['count']}")

        del vlm
        torch.cuda.empty_cache()

    out_path = os.path.splitext(JSONL_PATH)[0] + "__LGIP_fliptypes.metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print("\n[INFO] Saved all flip-type metrics to:", out_path)


if __name__ == "__main__":
    main()
