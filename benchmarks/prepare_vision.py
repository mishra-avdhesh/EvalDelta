"""CIFAR-10 DeltaBench source from public pretrained checkpoints (GPU used only for caching).

    python benchmarks/prepare_vision.py

Checkpoints: chenyaofo/pytorch-cifar-models (BSD-3-Clause), loaded via torch.hub. The pool is
the CIFAR-10 test set (10,000 images). Pipeline versions (fp16, JPEG ingestion, low-resolution
camera) are labelled ``pipeline`` and are not counted as model upgrades.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent / "data"
RAW = ROOT / "raw"
HUB = "chenyaofo/pytorch-cifar-models"
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# (version id, change type, description, hub model, pipeline)
LADDER = [
    ("v01", "architecture", "ResNet-20", "cifar10_resnet20", "fp32"),
    ("v02", "architecture", "ResNet-32 (deeper)", "cifar10_resnet32", "fp32"),
    ("v03", "architecture", "ResNet-44 (deeper)", "cifar10_resnet44", "fp32"),
    ("v04", "architecture", "ResNet-56 (deeper)", "cifar10_resnet56", "fp32"),
    ("v05", "architecture", "VGG-13-BN (architecture switch)", "cifar10_vgg13_bn", "fp32"),
    ("v06", "architecture", "VGG-16-BN", "cifar10_vgg16_bn", "fp32"),
    (
        "v07",
        "architecture",
        "MobileNetV2 x1.4 (mobile deployment)",
        "cifar10_mobilenetv2_x1_4",
        "fp32",
    ),
    (
        "v08",
        "compression",
        "MobileNetV2 x1.0 (width compression)",
        "cifar10_mobilenetv2_x1_0",
        "fp32",
    ),
    (
        "v09",
        "compression",
        "MobileNetV2 x0.75 (further compression)",
        "cifar10_mobilenetv2_x0_75",
        "fp32",
    ),
    ("v10", "pipeline", "ResNet-56 with fp16 inference", "cifar10_resnet56", "fp16"),
    ("v11", "pipeline", "ResNet-56 with JPEG q=70 ingestion", "cifar10_resnet56", "jpeg70"),
    ("v12", "architecture", "RepVGG-A2", "cifar10_repvgg_a2", "fp32"),
    ("v13", "architecture", "ShuffleNetV2 x2.0", "cifar10_shufflenetv2_x2_0", "fp32"),
    ("v14", "architecture", "VGG-19-BN", "cifar10_vgg19_bn", "fp32"),
    (
        "v15",
        "pipeline",
        "ResNet-56 behind a 24px low-resolution camera pipeline",
        "cifar10_resnet56",
        "lowres24",
    ),
]


HF_TEST = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet"


def load_test() -> tuple[torch.Tensor, np.ndarray]:
    """CIFAR-10 test split (10,000 images) from the Hugging Face parquet mirror."""
    import urllib.request

    path = RAW / "cifar10_test.parquet"
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF_TEST, path)
    df = pd.read_parquet(path)
    imgs = np.stack(
        [np.asarray(Image.open(io.BytesIO(r["bytes"])).convert("RGB")) for r in df["img"]]
    )
    x = torch.tensor(imgs).permute(0, 3, 1, 2).float() / 255.0
    return x, df["label"].to_numpy()


def pipeline(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind in {"fp32", "fp16"}:
        return x
    if kind == "jpeg70":
        out = []
        for img in (x * 255).byte().permute(0, 2, 3, 1).numpy():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=70)
            out.append(np.asarray(Image.open(buf)))
        return torch.tensor(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    if kind == "lowres24":
        small = F.interpolate(x, size=24, mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(small, size=32, mode="bilinear", align_corners=False)
    raise ValueError(kind)


@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor, half: bool, device: str) -> np.ndarray:
    model = model.to(device).eval()
    if half:
        model = model.half()
    probs = []
    for i in range(0, len(x), 1000):
        b = ((x[i : i + 1000] - MEAN) / STD).to(device)
        if half:
            b = b.half()
        probs.append(torch.softmax(model(b).float(), 1).cpu())
    return torch.cat(probs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-seconds", type=float, default=7200)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument(
        "--through",
        choices=[entry[0] for entry in LADDER],
        default=LADDER[-1][0],
        help="Assemble a documented prefix when later checkpoint downloads fail",
    )
    args = ap.parse_args()
    if args.max_seconds <= 0:
        ap.error("max-seconds must be positive")
    socket.setdefaulttimeout(90)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.require_cuda and device != "cuda":
        raise RuntimeError("CUDA required but unavailable; refusing CPU fallback")
    print(f"torch={torch.__version__}, device={device}", flush=True)
    t_start = time.time()
    selected = LADDER[: next(i for i, entry in enumerate(LADDER) if entry[0] == args.through) + 1]
    out = ROOT / "cifar10"
    if (out / "matrix.parquet").exists() and (out / "versions.json").exists():
        existing = json.loads((out / "versions.json").read_text())
        if [entry["id"] for entry in existing["versions"]] == [entry[0] for entry in selected]:
            print("Complete CIFAR-10 cache already exists; no GPU work needed", flush=True)
            return
    partial = ROOT / "cifar10_partial"
    partial.mkdir(parents=True, exist_ok=True)
    x, y = load_test()
    input_hash = hashlib.sha256((RAW / "cifar10_test.parquet").read_bytes()).hexdigest()
    cols: dict[str, np.ndarray] = {}
    vmeta = []
    for k, (vid, change, desc, hub, pipe) in enumerate(selected, start=1):
        cache = partial / f"{vid}.npz"
        meta_path = partial / f"{vid}.json"
        signature = {
            "cache_format": 1,
            "input_sha256": input_hash,
            "hub": HUB,
            "model": hub,
            "pipeline": pipe,
        }
        if cache.exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            if metadata.get("cache_signature") != signature:
                raise ValueError(f"stale cache signature for {vid}; inspect before replacing it")
            with np.load(cache, allow_pickle=False) as stored:
                p = stored["probabilities"]
            print(f"[cifar10] {vid}: reused completed cache", flush=True)
        else:
            if time.time() - t_start >= args.max_seconds:
                raise TimeoutError("prediction-cache wall-time budget exhausted; rerun to resume")
            t0 = time.time()
            print(f"[cifar10] {vid}: loading {hub}, pipeline={pipe}, device={device}", flush=True)
            model = torch.hub.load(HUB, hub, pretrained=True, trust_repo=True, verbose=False)
            inference_start = time.time()
            p = predict(model, pipeline(x, pipe), pipe == "fp16", device)
            inference_seconds = time.time() - inference_start
            metadata = {
                "id": vid,
                "release_index": k,
                "change_type": change,
                "description": desc,
                "checkpoint": f"{HUB}:{hub}",
                "pipeline": pipe,
                "seconds": round(time.time() - t0, 3),
                "inference_wall_seconds": round(inference_seconds, 3),
                "device": device,
                "cache_signature": signature,
            }
            temporary = cache.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, probabilities=p)
            temporary.replace(cache)
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        if p.shape != (len(y), len(CLASSES)) or not np.isfinite(p).all():
            raise ValueError(f"invalid prediction cache for {vid}")
        pred = p.argmax(1)
        cols[f"loss__{vid}"] = (pred != y).astype(float)
        cols[f"conf__{vid}"] = p.max(1)
        cols[f"ptrue__{vid}"] = p[np.arange(len(y)), y]
        cols[f"pred__{vid}"] = pred.astype(str)
        vmeta.append(metadata)
        print(f"[cifar10] {vid}: cached {len(y)} outcomes", flush=True)
    bright = x.mean((1, 2, 3)).numpy()
    base = pd.DataFrame(
        {
            "sample_id": [f"cifar10_test_{i}" for i in range(len(y))],
            "slice": [CLASSES[c] for c in y],
            "label": y.astype(str),
            "estimated_candidate_cost": 1.0,
            "f_brightness_z": (bright - bright.mean()) / bright.std(),
        }
    )
    out = ROOT / "cifar10"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([base, pd.DataFrame(cols)], axis=1).to_parquet(
        out / "matrix.tmp.parquet", index=False
    )
    gpu_seconds = time.time() - t_start if device == "cuda" else 0.0
    (out / "versions.json").write_text(
        json.dumps(
            {
                "source": "cifar10",
                "versions": vmeta,
                "meta": {
                    "task": "image_classification",
                    "model_family": "cifar10_cnn",
                    "license": "CIFAR-10 (Krizhevsky 2009; no explicit license); checkpoints "
                    "BSD-3-Clause (chenyaofo/pytorch-cifar-models); derived outcomes only",
                    "provenance": "CIFAR-10 test split (huggingface.co/datasets/uoft-cs/cifar10); torch.hub "
                    + HUB,
                    "slices": "true class",
                    "ladder_complete": len(selected) == len(LADDER),
                    "omitted_versions": [entry[0] for entry in LADDER[len(selected) :]],
                    "gpu_seconds": round(gpu_seconds, 1),
                },
            },
            indent=2,
        )
    )
    (out / "matrix.tmp.parquet").replace(out / "matrix.parquet")
    print(f"done in {time.time() - t_start:.0f}s on {device}", flush=True)


if __name__ == "__main__":
    main()
