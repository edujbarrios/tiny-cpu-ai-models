#!/usr/bin/env python3
"""End-to-end smoke test for a registered model; Python standard library only."""

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
    cmd = [
        sys.executable, str(Path(__file__).resolve().parents[1] / "run.py"),
        model_id,
        "Describe this image in one short sentence." if "vlm" in model_id
        else "Reply in one short sentence: what is a CPU?",
        "--max-tokens", "48",
    ]
    with tempfile.TemporaryDirectory() as temp:
        if "vlm" in model_id:
            image = Path(temp) / "sample.png"
            create_test_image(image)
            cmd.extend(["--image", str(image)])
        print("Running smoke test: {}".format(model_id), flush=True)
        return subprocess.call(cmd)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tests/smoke_models.py MODEL_ID")
    raise SystemExit(main(sys.argv[1]))
