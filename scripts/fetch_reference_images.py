"""Download banana reference photos from Wikimedia Commons for the target-embedding
centroid (train split) and eval-gallery centroid (eval split). No auth needed.

Usage: python scripts/fetch_reference_images.py [--n-train 40] [--n-eval 18]
"""
from __future__ import annotations

import argparse
import io
import random
import time
from pathlib import Path

import requests
from PIL import Image

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_CATEGORIES = ["Category:Bananas", "Category:Cavendish bananas", "Category:Banana"]
USER_AGENT = "adversarial-patch-research/0.1 (research use; contact: barneymurray.0@gmail.com)"
BAD_NAME_HINTS = ("map", "diagram", "logo", "icon", "chart", "graph", "flag", "svg", "chemical", "structure")


def list_category_files(category: str, limit: int = 100) -> list[dict]:
    files = []
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": min(limit, 50),
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 1024,
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT}
    while True:
        resp = requests.get(COMMONS_API, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            infos = page.get("imageinfo")
            if infos:
                files.append({"title": page.get("title", ""), **infos[0]})
        if len(files) >= limit or "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.2)
    return files[:limit]


def is_usable(info: dict) -> bool:
    mime = info.get("mime", "")
    if mime not in ("image/jpeg", "image/png"):
        return False
    w, h = info.get("width", 0), info.get("height", 0)
    if min(w, h) < 256:
        return False
    if max(w, h) / max(1, min(w, h)) > 3.0:
        return False
    title_lower = info.get("title", "").lower()
    if any(hint in title_lower for hint in BAD_NAME_HINTS):
        return False
    return True


def download(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest, "JPEG", quality=92)
        return True
    except Exception as e:
        print(f"  skip {url}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent.parent / "data" / "banana_refs")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = ap.parse_args()

    candidates = []
    seen_titles = set()
    for cat in args.categories:
        print(f"listing {cat} ...")
        for info in list_category_files(cat, limit=100):
            if info["title"] in seen_titles:
                continue
            seen_titles.add(info["title"])
            if is_usable(info):
                candidates.append(info)
        time.sleep(0.2)

    print(f"found {len(candidates)} usable candidates after filtering")
    need = args.n_train + args.n_eval
    if len(candidates) < need:
        print(f"WARNING: only {len(candidates)} usable images found, need {need}. "
              f"Proceeding with what's available (splitting proportionally).")

    random.Random(args.seed).shuffle(candidates)

    train_dir = args.out / "train"
    eval_dir = args.out / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    n_train_actual = min(args.n_train, max(1, len(candidates) - args.n_eval)) if len(candidates) < need else args.n_train
    train_candidates = candidates[:n_train_actual]
    eval_candidates = candidates[n_train_actual:n_train_actual + args.n_eval]

    def fetch_all(cands, dest_dir, label):
        ok = 0
        for i, info in enumerate(cands):
            url = info.get("thumburl") or info["url"]
            dest = dest_dir / f"{label}_{i:03d}.jpg"
            if download(url, dest):
                ok += 1
            time.sleep(0.15)
        return ok

    n_ok_train = fetch_all(train_candidates, train_dir, "train")
    n_ok_eval = fetch_all(eval_candidates, eval_dir, "eval")

    print(f"downloaded {n_ok_train} train images -> {train_dir}")
    print(f"downloaded {n_ok_eval} eval images -> {eval_dir}")
    if n_ok_train < 15 or n_ok_eval < 8:
        raise SystemExit(
            f"ERROR: too few images downloaded (train={n_ok_train}, eval={n_ok_eval}). "
            "Centroid embeddings need a reasonable sample size -- check network access "
            "or widen the CATEGORIES list."
        )


if __name__ == "__main__":
    main()
