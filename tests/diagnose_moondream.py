#!/usr/bin/env python3
"""Diagnostic CPU image inference for Moondream; does not modify the public model status."""

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run
from smoke_models import create_test_image


def main():
    model = run.load_catalog()["moondream-2b"]
    engine = run.ensure_runtime()
    weights = run.ensure_artifact(model)
    projector = run.ensure_artifact(model["mmproj"])
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "sample.png"
        create_test_image(image)
        args = SimpleNamespace(
            ctx_size=None,
            max_tokens=64,
            threads=2,
            image=str(image),
        )
        command = run.build_command(
            engine,
            model,
            weights,
            "Describe this image in one short sentence.",
            args,
            projector,
        )
        variants = [
            ("legacy-vicuna", command + ["--no-jinja"]),
            ("auto-jinja", [p for i, p in enumerate(command)
                            if i not in (command.index("--chat-template"),
                                         command.index("--chat-template") + 1)]),
            ("legacy-default", [p for i, p in enumerate(command)
                                 if i not in (command.index("--chat-template"),
                                              command.index("--chat-template") + 1)] + ["--no-jinja"]),
        ]
        for name, cmd in variants:
            print("=== Testing " + name + " ===", flush=True)
            result = subprocess.run(
                cmd, cwd=str(engine.parent), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                timeout=180,
            )
            print(result.stdout[-12000:], flush=True)
            rates = re.findall(r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", result.stdout)
            valid = (
                result.returncode == 0
                and "Loaded media from" in result.stdout
                and rates and float(rates[-1]) > 0
                and "Failed to tokenize prompt" not in result.stdout
            )
            print("RESULT {}: {}".format(name, "PASS" if valid else "FAIL"), flush=True)
            if valid:
                return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
