"""Live webcam demo: point your camera at something, and watch a panel of vision models from
different generations say what they see -- with and without the adversarial patch.

    python -m patchattack.live_demo --patch-path outputs/patches/<run>/patch.pt
    open http://127.0.0.1:8000        (print the patch from http://127.0.0.1:8000/print)

Two ways to apply the patch:
- digital: the server pastes the patch onto each camera frame at a spot you click, so the
  "camera only" and "with patch" columns are the exact same frame except for the patch;
- physical: print the patch, choose "camera only", and hold the printout next to an object.

Controls (same size and position as the patch, to show it is the *pattern* that matters and
not just "there's something new in frame"): a real photo of the target, and random noise.

Every model is labelled "trained against" (in the patch's training ensemble) or "never seen"
(pure transfer), read from the patch checkpoint's own config.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from patchattack.eval import load_patch
from patchattack.models import default_device
from patchattack.patch import Patch
from patchattack.reference_embeddings import BANANA_REFS
from patchattack.transforms import apply_patch, fixed_eot_params

HTML_PATH = Path(__file__).with_name("live_demo.html")
FRAME_SIZE = 448  # the browser sends a centre-square crop at this size

DEFAULT_LABELS = [
    "a banana",
    "an apple",
    "a bag of crisps",
    "a bottle",
    "a cup",
    "a phone",
    "a book",
    "a keyboard",
    "a person",
    "a hand",
    "a toaster",
    "a plant",
]
DEFAULT_VLMS = ["HuggingFaceTB/SmolVLM-500M-Instruct"]
VLM_PROMPT = "What is the main object in this image? Answer in a few words."


@dataclass
class ModelInfo:
    key: str
    name: str
    era: str
    task: str
    ensemble_name: str | None = (
        None  # name in the patch-training ensemble, if it could be in it
    )


# ---------------------------------------------------------------- closed-set classifiers
class ImageNetClassifier:
    def __init__(self, ctor, weights, device):
        self.model = ctor(weights=weights).eval().to(device)
        self.categories = weights.meta["categories"]
        self.size = weights.transforms().crop_size[0]
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[list[tuple[str, float]]]:
        x = F.interpolate(
            x, size=self.size, mode="bilinear", antialias=True, align_corners=False
        )
        probs = self.model((x - self.mean) / self.std).softmax(-1)
        top = probs.topk(3, dim=-1)
        return [
            [(self.categories[i], p) for p, i in zip(ps.tolist(), ids.tolist())]
            for ps, ids in zip(top.values, top.indices)
        ]


# ---------------------------------------------------------------- open-vocabulary zero-shot
class ZeroShot:
    """CLIP-family image/text matching over a user-editable label list."""

    def __init__(self, arch: str, pretrained: str, device: str, labels: list[str]):
        import open_clip

        self.model = (
            open_clip.create_model(arch, pretrained=pretrained).eval().to(device)
        )
        self.tokenizer = open_clip.get_tokenizer(arch)
        cfg = self.model.visual.preprocess_cfg
        size = cfg["size"]
        self.size = size[0] if isinstance(size, (tuple, list)) else size
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.device = device
        self.set_labels(labels)

    @torch.no_grad()
    def set_labels(self, labels: list[str]) -> None:
        self.labels = labels
        tokens = self.tokenizer([f"a photo of {label}" for label in labels]).to(
            self.device
        )
        self.text = F.normalize(self.model.encode_text(tokens), dim=-1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> list[list[tuple[str, float]]]:
        x = F.interpolate(
            x, size=self.size, mode="bilinear", antialias=True, align_corners=False
        )
        img = F.normalize(self.model.encode_image((x - self.mean) / self.std), dim=-1)
        probs = (100.0 * img @ self.text.T).softmax(-1)
        top = probs.topk(min(3, len(self.labels)), dim=-1)
        return [
            [(self.labels[i], p) for p, i in zip(ps.tolist(), ids.tolist())]
            for ps, ids in zip(top.values, top.indices)
        ]


# ---------------------------------------------------------------- detection + segmentation
class Detector:
    def __init__(self, weights: str, device: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device

    def __call__(self, frames_uint8: list[np.ndarray]):
        """frames: HxWx3 RGB uint8. Returns (per-frame detections, per-frame annotated RGB)."""
        # ultralytics expects BGR numpy arrays
        results = self.model.predict(
            [f[:, :, ::-1] for f in frames_uint8],
            device=self.device,
            conf=0.25,
            verbose=False,
        )
        dets, plots = [], []
        for r in results:
            names = r.names
            dets.append(
                sorted(
                    (
                        (names[int(c)], float(p))
                        for c, p in zip(r.boxes.cls, r.boxes.conf)
                    ),
                    key=lambda t: -t[1],
                )[:4]
            )
            plots.append(r.plot()[:, :, ::-1])
        return dets, plots


# ---------------------------------------------------------------- chat VLM
class ChatVLM:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.model = (
            AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
            .to(device)
            .eval()
        )
        self.device = device

    @torch.no_grad()
    def ask(self, image: Image.Image, prompt: str = VLM_PROMPT) -> str:
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conv,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        return self.processor.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


# ---------------------------------------------------------------- stickers (patch + controls)
def photo_patch(path: Path, size: int, device: str) -> Patch:
    """A Patch whose pixels are a real photo, centre-cropped to the same circle as the adversarial
    patch -- the 'just stick a picture of a banana on it' control."""
    img = Image.open(path).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    p = Patch(size, "circle").to(device)
    with torch.no_grad():
        x = pil_to_tensor(img).float().div(255).clamp(1e-3, 1 - 1e-3).to(device)
        p.raw.copy_(torch.logit(x))
    return p


class Demo:
    def __init__(self, args):
        self.device = args.device
        self.lock = (
            threading.Lock()
        )  # MPS is not safe to drive from two threads at once
        self.stickers: dict[str, Patch] = {}
        self.target = "banana"
        ensemble: list[str] = []
        if args.patch_path:
            patch, cfg = load_patch(args.patch_path, self.device)
            self.stickers["patch"] = patch
            self.target = cfg.get("target_name", "banana")
            ensemble = cfg.get("ensemble", [])
            size, shape = cfg["patch_size"], cfg["patch_shape"]
            self.patch_info = {
                "run": args.patch_path.parent.name,
                "ensemble": ensemble,
                "physical": cfg.get("camera_strength", 0) > 0,
            }
        else:
            size, shape = 200, "circle"
            self.patch_info = None
        torch.manual_seed(0)
        self.stickers["noise"] = Patch(size, shape, init="random").to(self.device)
        refs = sorted((Path(args.target_refs_dir) / "eval").glob("*.jpg"))
        if refs:
            self.stickers["photo"] = photo_patch(
                refs[args.target_photo_idx % len(refs)], size, self.device
            )

        self.labels = list(DEFAULT_LABELS)
        tv = torchvision.models
        print("loading models (first run downloads weights) ...")
        specs = [
            (
                ModelInfo(
                    "resnet50",
                    "ResNet-50",
                    "2015 · CNN",
                    "ImageNet-1k classifier",
                    "resnet50",
                ),
                lambda: ImageNetClassifier(
                    tv.resnet50, tv.ResNet50_Weights.IMAGENET1K_V2, self.device
                ),
            ),
            (
                ModelInfo(
                    "mobilenet_v3_large",
                    "MobileNetV3-L",
                    "2019 · on-device CNN",
                    "ImageNet-1k classifier",
                    "mobilenet_v3_large",
                ),
                lambda: ImageNetClassifier(
                    tv.mobilenet_v3_large,
                    tv.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
                    self.device,
                ),
            ),
            (
                ModelInfo(
                    "vit_b_16",
                    "ViT-B/16",
                    "2020 · Vision Transformer",
                    "ImageNet-1k classifier",
                    "vit_b_16",
                ),
                lambda: ImageNetClassifier(
                    tv.vit_b_16, tv.ViT_B_16_Weights.IMAGENET1K_V1, self.device
                ),
            ),
            (
                ModelInfo(
                    "clip_b32",
                    "CLIP ViT-B/32",
                    "2021 · vision-language",
                    "zero-shot over your labels",
                    "clip_b32",
                ),
                lambda: ZeroShot(
                    "ViT-B-32-quickgelu", "openai", self.device, self.labels
                ),
            ),
            (
                ModelInfo(
                    "siglip2_b16",
                    "SigLIP 2 ViT-B/16",
                    "2025 · vision-language",
                    "zero-shot over your labels",
                    "siglip2_b16",
                ),
                lambda: ZeroShot("ViT-B-16-SigLIP2", "webli", self.device, self.labels),
            ),
        ]
        if not args.no_detector:
            specs.append(
                (
                    ModelInfo(
                        "yolo",
                        "YOLO11-seg",
                        "2024 · detector",
                        "COCO detection + segmentation",
                    ),
                    lambda: Detector(args.detector, self.device),
                )
            )
        if getattr(args, "light", False):
            # keep only the responsive models so the whole panel runs per-frame on a laptop CPU/MPS
            keep = {"mobilenet_v3_large", "clip_b32"}
            specs = [s for s in specs if s[0].key in keep]
        self.models = {}
        self.infos = []
        for info, make in specs:
            t0 = time.time()
            self.models[info.key] = make()
            self.infos.append({**info.__dict__, "seen": info.ensemble_name in ensemble})
            print(f"  {info.name:20s} {time.time() - t0:5.1f}s")

        self.vlms = {}
        for model_id in args.vlm:
            t0 = time.time()
            self.vlms[model_id] = ChatVLM(model_id, self.device)
            name = model_id.split("/")[-1]
            self.infos.append(
                {
                    "key": f"vlm:{model_id}",
                    "name": name,
                    "era": "2024-25 · chat VLM",
                    "task": f'asked: "{VLM_PROMPT}"',
                    "ensemble_name": None,
                    "seen": False,
                }
            )
            print(f"  {name:20s} {time.time() - t0:5.1f}s")
        self.vlm_answers: dict[str, dict[str, str]] = {}
        self.vlm_frames: dict[str, torch.Tensor] = {}
        self.stop = threading.Event()
        if self.vlms:
            threading.Thread(target=self._vlm_loop, daemon=True).start()

    # ---- compositing
    def compose(
        self,
        frame: torch.Tensor,
        sticker: str,
        x: float,
        y: float,
        area: float,
        angle: float,
    ) -> torch.Tensor:
        """frame: (1,3,S,S) in [0,1]; x/y in [0,1] of the frame; area = fraction of frame area."""
        if sticker not in self.stickers:
            return frame
        S = frame.shape[-1]
        params = fixed_eot_params(1, S, area, angle, x * S, y * S, self.device)
        return apply_patch(self.stickers[sticker], frame, params).clamp(0, 1)

    # ---- inference
    @torch.no_grad()
    def infer(self, frame: torch.Tensor, settings: dict) -> dict:
        sticker = settings.get("sticker", "none")
        views = {"clean": frame}
        if sticker != "none":
            views["patched"] = self.compose(
                frame,
                sticker,
                settings.get("x", 0.72),
                settings.get("y", 0.72),
                settings.get("area", 0.12),
                settings.get("angle", 0.0),
            )
        names = list(views)
        batch = torch.cat([views[n] for n in names])
        results: dict[str, dict] = {}
        timings = {}
        annotated: dict[str, np.ndarray] = {}
        with self.lock:
            for key, model in self.models.items():
                t0 = time.time()
                if isinstance(model, Detector):
                    uint8 = [
                        (v[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        for v in batch.split(1)
                    ]
                    dets, plots = model(uint8)
                    results[key] = dict(zip(names, dets))
                    annotated = dict(zip(names, plots))
                else:
                    results[key] = dict(zip(names, model(batch)))
                timings[key] = round((time.time() - t0) * 1000)
        for n in names:
            self.vlm_frames[n] = views[n][0].detach().cpu()
        for model_id, answers in self.vlm_answers.items():
            results[f"vlm:{model_id}"] = {
                n: [(a, 1.0)] for n, a in answers.items() if n in names
            }

        show = names[-1]
        view = (
            annotated[show]
            if "yolo" in self.models and settings.get("overlay", True)
            else (views[show][0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )
        return {
            "type": "result",
            "views": names,
            "results": results,
            "timings": timings,
            "image": _jpeg_b64(view),
            "target": self.target,
        }

    def _vlm_loop(self) -> None:
        """Answers are slow (~1s+), so the VLM re-asks about the latest frame in a background loop."""
        while not self.stop.is_set():
            frames = dict(self.vlm_frames)
            if not frames:
                time.sleep(0.2)
                continue
            for model_id, vlm in self.vlms.items():
                answers = {}
                for view, x in frames.items():
                    with self.lock:
                        answers[view] = vlm.ask(to_pil_image(x))
                self.vlm_answers[model_id] = answers
            time.sleep(0.05)  # let the fast loop grab the lock between rounds

    def set_labels(self, labels: list[str]) -> None:
        self.labels = labels
        with self.lock:
            for model in self.models.values():
                if isinstance(model, ZeroShot):
                    model.set_labels(labels)


def _jpeg_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _decode_frame(data: bytes, device: str) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    if img.size != (FRAME_SIZE, FRAME_SIZE):
        img = img.resize((FRAME_SIZE, FRAME_SIZE), Image.Resampling.BILINEAR)
    return pil_to_tensor(img).float().div(255).unsqueeze(0).to(device)


def build_app(demo: Demo):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI()

    @app.get("/")
    def index():
        return HTMLResponse(HTML_PATH.read_text())

    @app.get("/sticker/{name}.png")
    def sticker_png(name: str, px: int = 1024):
        """The sticker as a transparent PNG, upsampled for printing."""
        p = demo.stickers[name]
        rgba = torch.cat([p.pixels(), p.mask]).detach().unsqueeze(0).cpu()
        rgba = F.interpolate(rgba, size=px, mode="bicubic", align_corners=False).clamp(
            0, 1
        )[0]
        buf = io.BytesIO()
        to_pil_image(rgba).save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.get("/print")
    def print_page():
        sizes = [5, 8, 12, 16]
        imgs = "".join(
            f'<figure><img src="/sticker/patch.png" style="width:{cm}cm;height:{cm}cm">'
            f"<figcaption>{cm} cm</figcaption></figure>"
            for cm in sizes
        )
        return HTMLResponse(
            "<!doctype html><title>Print patch</title><style>body{font:14px system-ui;margin:1cm}"
            "figure{display:inline-block;margin:.5cm;text-align:center}</style>"
            "<p>Print at 100% scale (no 'fit to page'), matte paper if you can - glossy glare hurts. "
            "Cut out a circle and hold it next to or on an object.</p>" + imgs
        )

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "models": demo.infos,
                    "labels": demo.labels,
                    "stickers": list(demo.stickers),
                    "target": demo.target,
                    "patch": demo.patch_info,
                }
            )
        )
        settings: dict = {"sticker": "patch" if "patch" in demo.stickers else "none"}
        try:
            while True:
                msg = await sock.receive()
                if msg.get("text"):
                    data = json.loads(msg["text"])
                    if "labels" in data:
                        await asyncio.to_thread(
                            demo.set_labels,
                            [s for s in data.pop("labels") if s.strip()],
                        )
                    settings.update(data)
                elif msg.get("bytes"):
                    frame = _decode_frame(msg["bytes"], demo.device)
                    out = await asyncio.to_thread(demo.infer, frame, settings)
                    await sock.send_text(json.dumps(out))
                elif msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass

    return app


def selftest(demo: Demo, image_path: Path, settings: dict) -> None:
    """Headless check: run one still image through the whole pipeline and print what each model says."""
    img = Image.open(image_path).convert("RGB")
    side = min(img.size)
    img = img.crop(
        (
            (img.width - side) // 2,
            (img.height - side) // 2,
            (img.width + side) // 2,
            (img.height + side) // 2,
        )
    )
    buf = io.BytesIO()
    img.resize((FRAME_SIZE, FRAME_SIZE)).save(buf, format="JPEG")
    frame = _decode_frame(buf.getvalue(), demo.device)
    out = demo.infer(frame, settings)
    for n in out["views"]:
        if n in demo.vlm_frames:
            for model_id, vlm in demo.vlms.items():
                out["results"].setdefault(f"vlm:{model_id}", {})[n] = [
                    (vlm.ask(to_pil_image(demo.vlm_frames[n])), 1.0)
                ]
    for info in demo.infos:
        r = out["results"].get(info["key"], {})
        line = "  |  ".join(
            f"{v}: " + ", ".join(f"{lbl} {p:.2f}" for lbl, p in r.get(v, [])[:2])
            for v in out["views"]
        )
        print(f"{info['name']:22s} {'SEEN ' if info['seen'] else 'unseen'}  {line}")
    print("timings ms:", out["timings"])
    Image.open(io.BytesIO(base64.b64decode(out["image"]))).save(
        image_path.with_suffix(".demo.jpg")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-path", type=Path, default=None)
    ap.add_argument(
        "--target-refs-dir",
        default=str(BANANA_REFS),
        help="its eval/ photos provide the 'real photo of the target' control sticker",
    )
    ap.add_argument("--target-photo-idx", type=int, default=0)
    ap.add_argument(
        "--vlm",
        nargs="*",
        default=DEFAULT_VLMS,
        help="HF chat-VLM ids, e.g. Qwen/Qwen2.5-VL-3B-Instruct; pass --vlm with no ids to disable",
    )
    ap.add_argument("--detector", default="yolo11n-seg.pt")
    ap.add_argument("--no-detector", action="store_true")
    ap.add_argument(
        "--light",
        action="store_true",
        help="laptop-friendly: load only MobileNet + CLIP, and disable YOLO + the chat VLM, "
        "so every panel updates per frame without bogging the machine down",
    )
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--selftest", type=Path, default=None, help="run one image headlessly and exit"
    )
    ap.add_argument(
        "--sticker", default="patch", help="selftest only: patch | photo | noise | none"
    )
    ap.add_argument("--area", type=float, default=0.12, help="selftest only")
    args = ap.parse_args()
    if args.light:
        args.no_detector = True
        args.vlm = []

    demo = Demo(args)
    if args.selftest:
        selftest(
            demo,
            args.selftest,
            {"sticker": args.sticker, "area": args.area, "x": 0.7, "y": 0.7},
        )
        return
    import uvicorn

    print(f"open http://{args.host}:{args.port}   (printable patch: /print)")
    uvicorn.run(build_app(demo), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
