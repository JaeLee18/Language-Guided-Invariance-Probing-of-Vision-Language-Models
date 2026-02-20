# LGIP: Linguistic-Graphic Invariance and Perturbation

[![arXiv](https://img.shields.io/badge/arXiv-2511.13494-b31b1b.svg)](https://arxiv.org/abs/2511.13494)
[![Journal](https://img.shields.io/badge/Pattern%20Recognition%20Letters-2026-blue.svg)](https://doi.org/10.1016/j.patrec.2026.02.012)

Evaluation of language robustness in Vision-Language Models (VLMs). Measures how well VLMs stay invariant to meaning-preserving paraphrases while detecting semantically different caption perturbations.

## Required Data

Download **COCO 2017 Train images** and place them so the image directory is:

```
LGIP/
  train2017/
    train2017/
      000000000009.jpg
      000000000025.jpg
      ...
```

- Download link: https://cocodataset.org/#download (2017 Train images, ~18GB)

You also need the base JSONL dataset `lgip_coco_native5_train2017.jsonl` which contains 40,000 COCO image entries with 5 captions each. This file is the input for all perturbation generation scripts.

## Dependencies

```
pip install torch open_clip_torch transformers Pillow tqdm
```

A CUDA GPU is expected. Scripts fall back to CPU if unavailable, but evaluation will be very slow.

## Scripts

### Caption Perturbation Generation

These scripts take the base JSONL and produce augmented versions with paraphrases and semantic flips.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `diff_gen_caption.py` | `lgip_coco_native5_train2017.jsonl` | `*_with_perturb.jsonl` | Basic perturbation: template paraphrases + color/number/object flips |
| `make_perturb_with_fliptypes.py` | `lgip_coco_native5_train2017.jsonl` | `*_with_fliptypes.jsonl` | Same as above but labels each flip with its type (color/number/object) |
| `rebuttal_enhanced_perturb_with_fliptypes.py` | `lgip_coco_native5_train2017.jsonl` | `rebuttal_enhanced_*_with_fliptypes.jsonl` | Extended: adds synonym substitution, passive voice paraphrases, and combined (paraphrase+flip) transformations |

### LGIP Evaluation

These scripts load VLM models, encode images and captions, compute cosine similarities, and output metric JSON files.

| Script | Input | Output | Description |
|--------|-------|--------|-------------|
| `run_lgip_multi.py` | `*_with_perturb.jsonl` | `*.metrics.json` per model + `*_ALL.metrics.json` | Main evaluation across 9 models (CLIP, OpenCLIP, EVA02, SigLIP, SigLIP2) |
| `run_lgip_by_fliptype.py` | `*_with_fliptypes.jsonl` | `*.metrics.json` per model | Evaluation with metrics broken down by flip type |
| `rebuttal_enhanced_lgip_by_transformation.py` | `rebuttal_enhanced_*_with_fliptypes.jsonl` | `*_ENHANCED_LGIP_analysis.json` | Strength-aware evaluation with per-type bins, weighted metrics, and simple vs. advanced paraphrase breakdown |

### Utility

| Script | Description |
|--------|-------------|
| `siglip_sanity_check.py` | Quick spot-check: loads SigLIP, picks 10 random samples, prints cosine similarities for original vs. flipped captions |

## Pipeline

```
lgip_coco_native5_train2017.jsonl
  |
  +--> diff_gen_caption.py ---------> *_with_perturb.jsonl -----> run_lgip_multi.py
  |
  +--> make_perturb_with_fliptypes.py -> *_with_fliptypes.jsonl -> run_lgip_by_fliptype.py
  |
  +--> rebuttal_enhanced_perturb_with_fliptypes.py -> rebuttal_enhanced_*_with_fliptypes.jsonl
                                                         |
                                                         +--> rebuttal_enhanced_lgip_by_transformation.py
```

## Configuration

All scripts use paths relative to the script directory (via `os.path.dirname(__file__)`), so they work out of the box as long as data files are placed alongside the scripts. Model lists, batch sizes, and other constants are configured at the top of each file.

## Models Evaluated

| Category | Models |
|----------|--------|
| OpenAI CLIP | ViT-B/16, ViT-L/14 |
| OpenCLIP (LAION-2B) | ViT-L/14, ViT-H/14 |
| EVA02-CLIP | EVA02-L-14 |
| SigLIP | base-224, base-384, large-384 |
| SigLIP2 | base-224 |

Model weights are downloaded automatically from HuggingFace Hub on first run.

## Metrics

- **invariance_error_mean** -- avg |sim(image, original) - sim(image, paraphrase)| -- lower is better
- **sensitivity_gap_mean** -- avg (sim(image, original) - sim(image, flip)) -- higher is better
- **sensitivity_pos_rate** -- fraction where original caption scores higher than flipped -- higher is better

## Citation

```bibtex
@article{JOONGLEE2026108,
  title = {Language-guided invariance probing of vision–language models},
  journal = {Pattern Recognition Letters},
  volume = {202},
  pages = {108-113},
  year = {2026},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2026.02.012},
}
```
