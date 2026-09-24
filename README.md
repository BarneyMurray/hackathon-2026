# Adversarial Patch — embedding-space replica

Replicates Brown et al., ["Adversarial Patch"](https://arxiv.org/abs/1712.09665) (NeurIPS 2017
workshop) — a universal, location/scale/rotation-robust image patch trained via
Expectation-over-Transformation (EOT) — retargeted at a modern **on-device model ensemble** and
an **embedding-space** attack formulation instead of the original's classifier cross-entropy.

## Method

- **Models**: ResNet18, MobileNetV3-Large, EfficientNet-B0 (mobile-scale CNNs), plus DINOv2
  ViT-S/14 and ViT-B/14 — five on-device-realistic vision models, nothing closed-source or
  oversized.
- **Attack formulation**: each model is treated purely as an **embedding extractor** (CNN
  penultimate pooled features; DINOv2 CLS token). The patch is trained to maximize **cosine
  similarity** between the patched image's embedding and a precomputed target-class centroid
  embedding, summed across the ensemble — not classifier logit-matching. This is more
  architecture-agnostic and matches how these models are actually deployed (retrieval/similarity
  search), and it's what lets DINOv2 (which has no classification head at all) slot into the same
  attack as the CNNs.
- **EOT training**: the patch is composited onto random background images via a differentiable
  `affine_grid`/`grid_sample` warp (random rotation, log-uniform-sampled scale, random or
  corner-anchored translation), and trained by gradient descent — only the patch's own pixels are
  a trainable parameter, every backbone stays frozen.
- **Target**: `banana` by default (reference photos in `data/banana_refs/`, fetched from Wikimedia
  Commons), but any target class works by pointing `--target-name`/`--target-refs-dir` at a
  different reference-image folder.

## Setup

```bash
uv venv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -e . --no-deps
uv pip install requests pillow numpy tqdm matplotlib pandas
```

> The `cu130` index targets NVIDIA Blackwell-class GPUs (e.g. GB10/DGX Spark). On other hardware,
> use the standard PyTorch install instructions for your CUDA version instead — nothing else in
> this codebase is Blackwell-specific.

Download the background dataset and banana reference photos:

```bash
bash scripts/download_imagenette.sh
python3 scripts/fetch_reference_images.py    # ~40 train / ~18 eval banana photos, Wikimedia Commons
```

## Producing a patch against the initial 5-model ensemble

```bash
python3 -m patchattack.train \
    --run-name paper_replica \
    --ensemble resnet18 mobilenet_v3_large efficientnet_b0 dinov2_vits14 dinov2_vitb14 \
    --target-name banana \
    --steps 3000 --batch-size 16 \
    --area-min 0.01 --area-max 0.20
```

This is the direct paper replica: **fully random patch placement** (matching the original paper's
setting — no positional bias toward where a real target object would be) across the whole
ensemble, banana as the target class. Outputs land in `outputs/patches/paper_replica/`: the
trained patch (`patch.pt`, plus periodic PNG previews), a training-curve plot, and a CSV log of
loss / per-model cosine similarity.

Add `--corner` for a corner-anchored placement variant (patch confined to a fixed bottom-left
region rather than anywhere in frame — a more realistic sticker-placement assumption for some
deployment scenarios, at the cost of being a less direct paper replica).

## Evaluating a trained patch

```bash
python3 -m patchattack.eval --patch-path outputs/patches/paper_replica/patch.pt --max-images 400
```

Reproduces the paper's Figure 3: attack success rate (patch composited onto held-out background
images, classified via nearest-centroid over an 11-way gallery — the target class plus
Imagenette's 10 classes as distractors) swept across patch sizes from 0.5% to 25% of image area.
Saves a CSV and a plot to `outputs/plots/`.

`--eval-ensemble` lets you test a patch against different models than it was trained on (black-box
transfer), and `--background-dataset {imagenette,coco,food_items}` swaps the background image
domain used for evaluation (see `patchattack/data.py` — `food_items` needs its own curated image
pool, not included here).

## Layout

```
src/patchattack/
  models.py                # ensemble loading + unified embed(x) interface
  transforms.py             # EOT sampling + differentiable grid_sample compositing
  patch.py                  # Patch nn.Module: sigmoid-parameterized pixels + circular/square mask
  reference_embeddings.py   # per-model target centroid + eval-gallery centroids (cached)
  data.py                   # background image datasets (Imagenette / COCO / food-item pool)
  train.py                  # EOT training loop, CLI entrypoint
  eval.py                   # nearest-centroid success-rate eval, Fig-3-style sweep
  viz.py                    # patch preview / training-curve plotting
scripts/
  fetch_reference_images.py # Wikimedia Commons reference-photo downloader (generic, any target class)
  download_imagenette.sh
1712.09665v2.pdf             # the paper being replicated
```
