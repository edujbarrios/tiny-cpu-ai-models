#!/usr/bin/env python3
"""End-to-end smoke test for a registered model; Python standard library only."""

import json
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path


def create_test_image(path, width=64, height=64):
    """Produce an RGB PNG without depending on Pillow or other packages."""
    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    # Solid-red RGB image, 64 by 64; valid PNG with no external assets.
    scanline = b"\x00" + b"\xf0\x20\x20" * width
    pixels = scanline * height
    header = b"\x89PNG\r\n\x1a\n"
    header += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    path.write_bytes(header + chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


def main(model_id):
    root = Path(__file__).resolve().parents[1]
    models = json.loads((root / "models.json").read_text(encoding="utf-8"))
    if model_id not in models:
        raise SystemExit("Unknown model: " + model_id)
    is_vision = models[model_id]["type"] == "vision"
    if models[model_id].get("status") == "research":
        raise SystemExit("Research-only model cannot be smoke-tested: " + model_id)
    cmd = [
        sys.executable, str(root / "run.py"),
        model_id,
        models[model_id].get("smoke_prompt", (
            "Describe this image in one short sentence." if is_vision
            else "Reply in one short sentence: what is a CPU?"
        )),
        "--max-tokens", "48",
    ]
    with tempfile.TemporaryDirectory() as temp:
        if is_vision:
            image = Path(temp) / "sample.png"
            create_test_image(image)
            cmd.extend(["--image", str(image)])
        print("Running smoke test: {}".format(model_id), flush=True)
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace",
        )
        print(result.stdout, flush=True)
        if result.returncode:
            return result.returncode
        rates = re.findall(r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", result.stdout)
        if not rates or float(rates[-1]) <= 0:
            print("ERROR: Model produced no measurable output tokens.", file=sys.stderr)
            return 1
        if is_vision and "Loaded media from" not in result.stdout:
            print("ERROR: Image input was not actually loaded.", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tests/smoke_models.py MODEL_ID")
    raise SystemExit(main(sys.argv[1]))
