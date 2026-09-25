# Extensions: transfer to CLIP/SigLIP + VLMs, and defense evaluation

This branch extends the embedding-space patch replica with (1) two more encoder families in the
ensemble, (2) a black-box transfer test against chat vision-language models, and (3) a defense
evaluation. It is an **academic study** run entirely offline on held-out Imagenette validation
images — the target class is a harmless **banana**, and no real device, camera, or product is
involved or targeted. The goal is to show *how* embedding-based image AI can be fooled and *whether*
simple preprocessing defenses stop it.

## What was added

| File | Change | Why |
|------|--------|-----|
| `src/patchattack/models.py` | `OpenClipEmbedder` + `clip_b32` / `siglip_b16` loaders; `default_device()` (cuda→mps→cpu) | Add CLIP ViT-B/32 and SigLIP ViT-B/16 — the vision towers most VLMs are built on — as ensemble members / transfer targets |
| `src/patchattack/eval_vlm.py` | **new** | Black-box transfer test: paste the patch on held-out photos, ask a chat VLM "what is the main object?", count answers mentioning the target |
| `src/patchattack/eval_defense.py` | **new** | Run six preprocessing defenses *before* the model and measure how much the "banana" rate drops |
| `src/patchattack/train.py` | `save_checkpoint()` + checkpoint every preview step and on `KeyboardInterrupt` | An interrupted run previously saved nothing; long runs are now recoverable |
| `src/patchattack/eval.py`, `viz.py` | `default_device()` default; `plot_vlm_hit_rate()` | Portability + a VLM transfer plot |
| `pyproject.toml` | optional `vlm` extra (`open_clip_torch`, `transformers>=5`, `accelerate`, `num2words`) | Keeps the heavy CLIP/VLM deps optional |

## How to run

```bash
uv pip install -e ".[vlm]"        # adds open_clip + transformers for the new evals

# 1. Train a patch, now optionally including CLIP/SigLIP in the ensemble
python -m patchattack.train  # (add clip_b32 / siglip_b16 to the ensemble config to include them)

# 2. Transfer to unseen CLIP/SigLIP (models the patch never trained on)
python -m patchattack.eval --patch-path outputs/patches/<run>/patch.pt \
    --eval-ensemble clip_b32 siglip_b16

# 3. Transfer to chat VLMs (SmolVLM-256M/500M, Qwen2.5-VL-3B by default)
python -m patchattack.eval_vlm --patch-path outputs/patches/<run>/patch.pt

# 4. Defense evaluation (JPEG / blur / median / segmentation) on both the
#    embedding classifier and one VLM
python -m patchattack.eval_defense --patch-path outputs/patches/<run>/patch.pt
```

Each writes CSVs (and plots) under `outputs/plots/`.

## Controls (so a rise isn't just "any sticker")

Every eval compares three conditions on the *same* placements: the **clean** image (no patch), the
**adversarial** patch, and a **random-noise** patch of identical size/position. Only the adversarial
pattern should move the prediction.

## Headline results (target = banana, evaluated on held-out Imagenette val)

Patch trained on 5 CNN/DINOv2 backbones, then tested on **unseen** encoders and VLMs. "Says banana"
rate at 25% patch area (clean & random-patch controls were ~0% throughout):

| Model (unseen at train time) | Type | Says banana |
|---|---|---|
| CLIP ViT-B/32 | image encoder | ~88% |
| SigLIP ViT-B/16 | image encoder | ~74% |
| SmolVLM-500M | chat VLM | ~91% |
| SmolVLM-256M | chat VLM | ~75% |
| Qwen2.5-VL-3B | chat VLM | ~61% |

**Defenses** (embedding classifier, adversarial banana-rate averaged over 6 encoders; lower = better):

| Defense | Family | Banana rate |
|---|---|---|
| none | — | ~93% |
| JPEG q30 / blur / median | blind input transform | ~94–96% (no help; blur is worse) |
| segmentation, blind localiser | find + remove | ~92% (barely) |
| segmentation, **perfect** localiser | oracle upper bound | **~2%** |

**Takeaway.** The attack transfers across architectures it never saw, including full chat VLMs.
Cheap blind defenses (blur/JPEG/median) do not stop it. Removing the patch works — but only with
near-perfect knowledge of *where* the patch is, which blind localisation does not reliably provide.
For any consequential decision, this argues for keeping a human in the loop.
