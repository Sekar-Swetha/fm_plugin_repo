"""Download SD1.5 + ControlNet-OpenPose weights for MotionEditor.

Pure stdlib urllib (no hf CLI / xet — those hang on some lab networks).
Streams each file, skips files already present, falls back .safetensors->.bin.

Run from motionEditor/MotionEditor:
    python3 download_weights.py
"""

import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
CN = "https://huggingface.co/lllyasviel/sd-controlnet-openpose/resolve/main"
SD = "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main"


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  skip (exists) {dest.relative_to(CKPT)}  {dest.stat().st_size/1e6:.1f} MB")
        return True
    print(f"  GET {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                if total % (50 << 20) < (1 << 20):
                    print(f"    ... {total/1e6:.0f} MB", flush=True)
    except Exception as e:
        print(f"  FAIL: {e}")
        if dest.exists():
            dest.unlink()
        return False
    print(f"  saved {dest.stat().st_size/1e6:.1f} MB  -> {dest.relative_to(CKPT)}")
    return True


def fetch_weight(base: str, rel_dir: str, local_dir: Path) -> bool:
    """Try diffusion/model .safetensors, fall back to .bin."""
    name = "diffusion_pytorch_model" if rel_dir in ("unet", "vae", "") else "model"
    url_dir = f"{base}/{rel_dir}" if rel_dir else base
    if fetch(f"{url_dir}/{name}.safetensors", local_dir / f"{name}.safetensors"):
        return True
    print("  (.safetensors not found, trying .bin)")
    return fetch(f"{url_dir}/{name}.bin", local_dir / f"{name}.bin")


def main():
    ok = True

    print("== ControlNet-OpenPose ==")
    cn = CKPT / "sd-controlnet-openpose"
    ok &= fetch(f"{CN}/config.json", cn / "config.json")
    ok &= fetch_weight(CN, "", cn)   # diffusion_pytorch_model.safetensors|bin at root

    print("== Stable Diffusion v1.5 ==")
    sd = CKPT / "stable-diffusion-v1-5"
    ok &= fetch(f"{SD}/model_index.json", sd / "model_index.json")
    ok &= fetch(f"{SD}/scheduler/scheduler_config.json", sd / "scheduler/scheduler_config.json")
    for fn in ("vocab.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json"):
        ok &= fetch(f"{SD}/tokenizer/{fn}", sd / "tokenizer" / fn)
    ok &= fetch(f"{SD}/feature_extractor/preprocessor_config.json",
                sd / "feature_extractor/preprocessor_config.json")
    ok &= fetch(f"{SD}/text_encoder/config.json", sd / "text_encoder/config.json")
    ok &= fetch_weight(SD, "text_encoder", sd / "text_encoder")
    ok &= fetch(f"{SD}/unet/config.json", sd / "unet/config.json")
    ok &= fetch_weight(SD, "unet", sd / "unet")
    ok &= fetch(f"{SD}/vae/config.json", sd / "vae/config.json")
    ok &= fetch_weight(SD, "vae", sd / "vae")

    print("\n== Summary ==")
    for p in sorted(CKPT.rglob("*")):
        if p.is_file():
            print(f"  {p.stat().st_size/1e6:8.1f} MB  {p.relative_to(CKPT)}")
    print("DONE" if ok else "SOME FILES FAILED — re-run to resume (existing files are skipped)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
