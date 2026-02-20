import os
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import gc

import open_clip
from transformers import AutoModel, AutoProcessor

# Updated to use the enhanced JSONL file
_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(_DIR, "rebuttal_enhanced_lgip_coco_native5_train2017_with_fliptypes.jsonl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================
# VRAM OPTIMIZATION SETTINGS FOR MAXIMUM GPU UTILIZATION
# =================================================================
IMAGE_BATCH_SIZE = 32      # Process multiple images simultaneously (adjust based on GPU memory)
TEXT_BATCH_SIZE = 512     # Large text batches for efficient GPU utilization
USE_MIXED_PRECISION = True  # Use FP16 for 2x faster inference with minimal accuracy loss
CACHE_IMAGES = True       # Pre-load images into memory for faster access

# Performance Tips:
# - IMAGE_BATCH_SIZE: Increase if you have more VRAM (e.g., 16 for 24GB GPUs)
# - TEXT_BATCH_SIZE: Larger is better, but may hit model token limits
# - Mixed precision: Usually 2x speedup with <1% accuracy loss
# - Monitor GPU usage with nvidia-smi during runs

# Models to evaluate
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

        # Enable mixed precision if requested
        self.use_amp = USE_MIXED_PRECISION and device == "cuda"
        if self.use_amp:
            print(f"[INFO] Using mixed precision (FP16) for faster inference")
            self.scaler = torch.cuda.amp.GradScaler()

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img_t = self.preprocess(img).unsqueeze(0).to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                feat = self.model.encode_image(img_t)
        else:
            feat = self.model.encode_image(img_t)

        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0]

    @torch.no_grad()
    def encode_images_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Batch encode multiple images for maximum VRAM utilization"""
        if len(image_paths) == 0:
            return torch.empty(0, 0, device=self.device)

        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img_t = self.preprocess(img)
            images.append(img_t)

        batch_images = torch.stack(images).to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                feats = self.model.encode_image(batch_images)
        else:
            feats = self.model.encode_image(batch_images)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0, device=self.device)

        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            tokens = self.tokenizer(batch).to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    feats = self.model.encode_text(tokens)
            else:
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

        # Enable mixed precision if requested
        self.use_amp = USE_MIXED_PRECISION and device == "cuda"
        if self.use_amp:
            print(f"[INFO] Using mixed precision (FP16) for faster inference")

    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                feats = self.model.get_image_features(**inputs)
        else:
            feats = self.model.get_image_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0]

    @torch.no_grad()
    def encode_images_batch(self, image_paths: List[str]) -> torch.Tensor:
        """Batch encode multiple images for maximum VRAM utilization"""
        if len(image_paths) == 0:
            return torch.empty(0, 0, device=self.device)

        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        if self.use_amp:
            with torch.cuda.amp.autocast():
                feats = self.model.get_image_features(**inputs)
        else:
            feats = self.model.get_image_features(**inputs)

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if len(texts) == 0:
            return torch.empty(0, 0, device=self.device)

        all_feats = []
        for i in range(0, len(texts), TEXT_BATCH_SIZE):
            batch = texts[i:i + TEXT_BATCH_SIZE]
            inputs = self.processor(text=batch, padding=True, truncation=True,
                                   return_tensors="pt").to(self.device)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    feats = self.model.get_text_features(**inputs)
            else:
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

# -----------------------
# Helpers
# -----------------------

def safe_mean(xs: List[float]) -> Optional[float]:
    return float(sum(xs) / len(xs)) if xs else None

def safe_std(xs: List[float]) -> Optional[float]:
    if len(xs) <= 1:
        return 0.0 if len(xs) == 1 else None
    return float(torch.std(torch.tensor(xs)))

def weighted_mean(values: List[float], weights: List[float]) -> Optional[float]:
    if not values:
        return None
    wsum = float(sum(weights))
    if wsum <= 0:
        return None
    return float(sum(v * w for v, w in zip(values, weights)) / wsum)

def weighted_pos_rate(gaps: List[float], weights: List[float]) -> Optional[float]:
    if not gaps:
        return None
    wsum = float(sum(weights))
    if wsum <= 0:
        return None
    pos_w = float(sum(w for g, w in zip(gaps, weights) if g > 0))
    return pos_w / wsum

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))

def quantile_thresholds(xs: List[float], qs: Tuple[float, float]) -> Tuple[float, float]:
    # returns (q_low, q_high)
    if not xs:
        return (0.0, 1.0)
    t = torch.tensor(xs)
    q1 = float(torch.quantile(t, qs[0]))
    q2 = float(torch.quantile(t, qs[1]))
    return q1, q2

def strength_bin(s: float, q_low: float, q_high: float) -> str:
    if s <= q_low:
        return "weak"
    if s <= q_high:
        return "medium"
    return "strong"


# -----------------------
# Flip parsing
# -----------------------

def normalize_flip_type(name: str) -> str:
    # make consistent keys
    name = name.lower().strip()
    name = name.replace("flip_", "")
    name = name.replace("diff_caps_", "")
    name = name.replace("para+", "para+")
    return name

def extract_flip_items(sample: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    """
    Returns a list of dict items:
      {"text": <flip_caption>, "flip_type": <type>, "meta": {...}}
    Supports:
      - sample["diff_caps_*"][key] = [str, str, ...]
      - sample["diff_caps"][key] = [str, ...]
      - sample["diff_caps"][key] = [{"text":..., "flip_type":..., ...}, ...]
    """
    items: List[Dict[str, Any]] = []

    # 1) unified diff_caps
    diff_caps = sample.get("diff_caps", {})
    if isinstance(diff_caps, dict) and key in diff_caps:
        payload = diff_caps.get(key, [])
        if payload and isinstance(payload[0], dict):
            for it in payload:
                if "text" in it:
                    items.append({
                        "text": it["text"],
                        "flip_type": normalize_flip_type(it.get("flip_type", "all")),
                        "meta": {k: v for k, v in it.items() if k not in ["text", "flip_type"]}
                    })
        else:
            for s in payload:
                items.append({"text": s, "flip_type": "all", "meta": {}})

    # 2) specialized diff_caps_* fields, if present
    for k, v in sample.items():
        if not isinstance(k, str):
            continue
        if not k.startswith("diff_caps_"):
            continue
        flip_type = normalize_flip_type(k)  # e.g., diff_caps_color -> color
        caps_dict = v
        if isinstance(caps_dict, dict) and key in caps_dict:
            for s in caps_dict.get(key, []):
                items.append({"text": s, "flip_type": flip_type, "meta": {}})

    return items


# -----------------------
# Main evaluation
# -----------------------

def compute_strength_aware_lgip(
    vlm,
    jsonl_path: str,
    model_tag: str,
    # Type priors approximate human-perceived severity. Tune if needed.
    type_prior: Optional[Dict[str, float]] = None,
    # Blend between type prior and semantic distance between orig and flip captions
    alpha_type: float = 0.5,
    # Strength bins by quantiles (weak <= q25, medium <= q75, strong > q75)
    bin_quantiles: Tuple[float, float] = (0.25, 0.75),
) -> Dict[str, Any]:

    if type_prior is None:
        type_prior = {
            "color": 0.25,
            "number": 0.55,
            "object": 1.00,
            "spatial": 0.80,   # if you add spatial flips
            "all": 0.60,
        }

    # Global
    inv_diffs_global: List[float] = []
    gap_global: List[float] = []
    strength_global: List[float] = []
    correct_global = 0
    total_global = 0

    # Weighted global
    gap_global_w: List[float] = []
    w_global: List[float] = []

    # Per-type, per-bin
    # stats[type][bin] accumulates gaps + weights + counts
    stats = defaultdict(lambda: defaultdict(lambda: {
        "gaps": [],
        "weights": [],
        "total": 0,
        "correct": 0,
    }))

    # Paraphrase invariance split (simple vs advanced)
    inv_simple: List[float] = []
    inv_advanced: List[float] = []

    num_samples = 0

    # Pre-load all samples for batching
    all_samples = list(load_jsonl(jsonl_path))

    # Process samples in batches for maximum VRAM utilization
    for batch_start in tqdm(range(0, len(all_samples), IMAGE_BATCH_SIZE),
                           desc=f"Strength-aware LGIP [{model_tag}]"):

        batch_samples = all_samples[batch_start:batch_start + IMAGE_BATCH_SIZE]
        batch_image_paths = [s["image_path"] for s in batch_samples]

        try:
            # Batch encode all images in this batch
            if hasattr(vlm, 'encode_images_batch'):
                batch_img_feats = vlm.encode_images_batch(batch_image_paths)
            else:
                # Fallback for single image encoding
                batch_img_feats = []
                for img_path in batch_image_paths:
                    feat = vlm.encode_image(img_path)
                    batch_img_feats.append(feat)
                batch_img_feats = torch.stack(batch_img_feats)

        except Exception as e:
            print(f"[WARN] {model_tag}: failed batch {batch_start}-{batch_start+len(batch_samples)}: {e}")
            continue

        # Process each sample in the batch
        for sample_idx, sample in enumerate(batch_samples):
            img_feat = batch_img_feats[sample_idx]
            orig_caps = sample["orig_captions"]
            same_caps = sample.get("same_caps", {})

            try:
                orig_feats = vlm.encode_text(orig_caps)
            except Exception as e:
                print(f"[WARN] {model_tag}: failed text encoding for sample: {e}")
                continue

            if orig_feats.numel() == 0:
                continue

            for idx, base_feat in enumerate(orig_feats):
                key = str(idx)

                base_score = F.cosine_similarity(
                    img_feat.unsqueeze(0),
                    base_feat.unsqueeze(0),
                    dim=-1
                )[0].item()

                # -------------
                # (A) Paraphrase invariance, now exportable
                # -------------
                caps_same = same_caps.get(key, [])
                if caps_same:
                    # classify simple vs advanced paraphrase
                    simple_paras, advanced_paras = [], []
                    for cap in caps_same:
                        is_simple = any(phrase in cap.lower() for phrase in [
                            "a photo of", "an image of", "a picture of",
                            "in this image", "in the picture", "this image shows"
                        ])
                        (simple_paras if is_simple else advanced_paras).append(cap)

                    if simple_paras:
                        feats = vlm.encode_text(simple_paras)
                        scores = cosine_sim_single_to_many(img_feat, feats).detach().cpu().tolist()
                        inv_simple.extend([abs(base_score - s) for s in scores])
                        inv_diffs_global.extend([abs(base_score - s) for s in scores])

                    if advanced_paras:
                        feats = vlm.encode_text(advanced_paras)
                        scores = cosine_sim_single_to_many(img_feat, feats).detach().cpu().tolist()
                        inv_advanced.extend([abs(base_score - s) for s in scores])
                        inv_diffs_global.extend([abs(base_score - s) for s in scores])

                # -------------
                # (B) Semantic flips: strength-aware + error-type stats
                # -------------
                flip_items = extract_flip_items(sample, key)
                if not flip_items:
                    continue

                flip_texts = [it["text"] for it in flip_items]
                flip_types = [it["flip_type"] for it in flip_items]

                flip_feats = vlm.encode_text(flip_texts)
                flip_scores = cosine_sim_single_to_many(img_feat, flip_feats).detach().cpu().tolist()

                # semantic distance between orig caption embedding and flip caption embedding
                # uses same text encoder (model-specific), but correlates well with semantic change
                base_feat_norm = base_feat / (base_feat.norm() + 1e-12)
                flip_feat_norm = flip_feats / (flip_feats.norm(dim=-1, keepdim=True) + 1e-12)
                text_cos = (flip_feat_norm @ base_feat_norm).detach().cpu().tolist()
                text_dist = [1.0 - float(c) for c in text_cos]  # [0..2] but typically [0..1]

                # compute gaps and strength
                gaps = [base_score - s for s in flip_scores]
                for g in gaps:
                    gap_global.append(g)
                    correct_global += 1 if g > 0 else 0
                    total_global += 1

                # normalize text_dist to [0,1] conservatively
                # (cos can be negative; clamp dist to [0,1] keeps strength stable)
                text_dist01 = [clip01(d) for d in text_dist]

                strengths = []
                weights = []
                for ftype, d01 in zip(flip_types, text_dist01):
                    prior = float(type_prior.get(ftype, type_prior.get("all", 0.6)))
                    s = alpha_type * prior + (1.0 - alpha_type) * d01
                    s = clip01(s)
                    strengths.append(s)
                    # use strength directly as weight, but avoid zero
                    weights.append(max(1e-4, s))

                strength_global.extend(strengths)
                gap_global_w.extend(gaps)
                w_global.extend(weights)

                # store per-type temporary, bin later after we know thresholds
                for ftype, g, w, s in zip(flip_types, gaps, weights, strengths):
                    stats[ftype]["__all__"]["gaps"].append(g)
                    stats[ftype]["__all__"]["weights"].append(w)
                    stats[ftype]["__all__"]["total"] += 1
                    stats[ftype]["__all__"]["correct"] += 1 if g > 0 else 0
                    stats[ftype]["__all__"].setdefault("strengths", []).append(s)

        num_samples += 1

    # -----------------------
    # Bin thresholds (global)
    # -----------------------
    q_low, q_high = quantile_thresholds(strength_global, bin_quantiles)

    # -----------------------
    # Finalize per-type bins
    # -----------------------
    per_type_summary = {}
    for ftype, bins in stats.items():
        all_bin = bins.get("__all__", {})
        gaps = all_bin.get("gaps", [])
        weights = all_bin.get("weights", [])
        strengths = all_bin.get("strengths", [])

        # allocate to bins
        for g, w, s in zip(gaps, weights, strengths):
            b = strength_bin(s, q_low, q_high)
            bins[b]["gaps"].append(g)
            bins[b]["weights"].append(w)
            bins[b]["total"] += 1
            bins[b]["correct"] += 1 if g > 0 else 0

        # summarize each bin + overall
        def summarize_bin(bins_dict: Dict, bin_name: str) -> Dict[str, Any]:
            bb = bins_dict.get(bin_name, {})
            bg = bb.get("gaps", [])
            bw = bb.get("weights", [])
            return {
                "count": len(bg),
                "gap_mean": safe_mean(bg),
                "gap_std": safe_std(bg),
                "pos_rate": (bb["correct"] / bb["total"]) if bb.get("total", 0) > 0 else None,
                "w_gap_mean": weighted_mean(bg, bw),
                "w_pos_rate": weighted_pos_rate(bg, bw),
            }

        per_type_summary[ftype] = {
            "overall": summarize_bin(bins, "__all__"),
            "weak": summarize_bin(bins, "weak"),
            "medium": summarize_bin(bins, "medium"),
            "strong": summarize_bin(bins, "strong"),
        }

    # -----------------------
    # Global summaries
    # -----------------------
    result = {
        "model_tag": model_tag,
        "num_samples": num_samples,

        # invariance
        "global_invariance_error_mean": safe_mean(inv_diffs_global),
        "paraphrase_simple_inv_error_mean": safe_mean(inv_simple),
        "paraphrase_advanced_inv_error_mean": safe_mean(inv_advanced),
        "paraphrase_simple_count": len(inv_simple),
        "paraphrase_advanced_count": len(inv_advanced),

        # unweighted sensitivity
        "global_sensitivity": {
            "count": len(gap_global),
            "gap_mean": safe_mean(gap_global),
            "gap_std": safe_std(gap_global),
            "pos_rate": (correct_global / total_global) if total_global > 0 else None,
        },

        # strength-aware weighted sensitivity
        "global_strength": {
            "count": len(strength_global),
            "strength_mean": safe_mean(strength_global),
            "q_low": q_low,
            "q_high": q_high,
        },
        "global_weighted_sensitivity": {
            "count": len(gap_global_w),
            "w_gap_mean": weighted_mean(gap_global_w, w_global),
            "w_pos_rate": weighted_pos_rate(gap_global_w, w_global),
        },

        # error-type stats
        "error_type_stats": per_type_summary,
    }

    return result

# ================ Analysis & Summary Functions ================

# -----------------------
# Printing
# -----------------------

def print_strength_aware_report(results: List[Dict[str, Any]]):
    print("\n" + "=" * 90)
    print("STRENGTH-AWARE LGIP REPORT (weighted + error-type + bins)")
    print("=" * 90)

    for r in results:
        print(f"\nModel: {r['model_tag']}")
        print(f"  Samples: {r['num_samples']}")

        print("  Invariance:")
        print(f"    Global InvErr: {r['global_invariance_error_mean']:.6f}" if r["global_invariance_error_mean"] is not None else "    Global InvErr: None")
        print(f"    Simple InvErr: {r['paraphrase_simple_inv_error_mean']:.6f} (N={r['paraphrase_simple_count']})"
              if r["paraphrase_simple_inv_error_mean"] is not None else f"    Simple InvErr: None (N={r['paraphrase_simple_count']})")
        print(f"    Adv InvErr:    {r['paraphrase_advanced_inv_error_mean']:.6f} (N={r['paraphrase_advanced_count']})"
              if r["paraphrase_advanced_inv_error_mean"] is not None else f"    Adv InvErr:    None (N={r['paraphrase_advanced_count']})")

        gs = r["global_sensitivity"]
        print("  Sensitivity (unweighted):")
        print(f"    Gap mean: {gs['gap_mean']:+.6f} | PosRate: {gs['pos_rate']:.2%} | N={gs['count']}")

        gw = r["global_weighted_sensitivity"]
        gstr = r["global_strength"]
        print("  Strength-aware (weighted):")
        print(f"    Strength mean: {gstr['strength_mean']:.4f} | bins q25={gstr['q_low']:.4f}, q75={gstr['q_high']:.4f}")
        print(f"    W-Gap mean: {gw['w_gap_mean']:+.6f} | W-PosRate: {gw['w_pos_rate']:.2%} | N={gw['count']}")

        print("  Error-type stats (overall + bins):")
        # show common types first if present
        preferred = ["color", "number", "object", "spatial", "all"]
        types = [t for t in preferred if t in r["error_type_stats"]] + \
                [t for t in r["error_type_stats"].keys() if t not in preferred]
        for t in types:
            s = r["error_type_stats"][t]
            o = s["overall"]
            print(f"    [{t}] N={o['count']}, Gap={o['gap_mean']:+.4f}, Pos={o['pos_rate']:.1%}, WGap={o['w_gap_mean']:+.4f}, WPos={o['w_pos_rate']:.1%}")
            for b in ["weak", "medium", "strong"]:
                bb = s[b]
                if bb["count"] > 0:
                    print(f"      - {b:6} N={bb['count']}, Gap={bb['gap_mean']:+.4f}, Pos={bb['pos_rate']:.1%}, WGap={bb['w_gap_mean']:+.4f}, WPos={bb['w_pos_rate']:.1%}")

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

        result = compute_strength_aware_lgip(
            vlm,
            JSONL_PATH,
            cfg["name"],
            alpha_type=0.5,  # tune 0.3 to rely more on text distance, 0.7 to rely more on priors
        )
        all_results.append(result)

        del vlm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # Clear CPU memory too

    # Detailed analysis across all models
    print_strength_aware_report(all_results)

    # Save results
    output_path = os.path.splitext(JSONL_PATH)[0] + "__ENHANCED_LGIP_analysis.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[INFO] Saved enhanced analysis to: {output_path}")


if __name__ == "__main__":
    main()
