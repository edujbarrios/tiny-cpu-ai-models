#!/usr/bin/env python3
"""Tiny CPU AI Models: download and run GGUF models on CPUs with no pip dependencies."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CATALOG = ROOT / "models.json"
RUNTIMES = ROOT / "runtimes.json"
USER_AGENT = "tiny-cpu-ai-models/0.1 (CPU-only, no-hf-cli)"


def info(message):
    print(message, file=sys.stderr, flush=True)


def request(url):
    if not url.startswith("https://"):
        raise RuntimeError("Only HTTPS downloads are allowed.")
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, dest, expected_sha256):
    """Atomic downloads, SHA-256 verification, and local cache reuse."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise RuntimeError("A valid SHA-256 checksum is required to verify downloads.")

    if dest.is_file() and file_sha256(dest).lower() == expected_sha256.lower():
        info("✓ Already downloaded: " + dest.name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    info("↓ Downloading " + dest.name)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".downloading-", dir=dest.parent, delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(request(url), timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    digest.update(chunk)

        if digest.hexdigest().lower() != expected_sha256.lower():
            raise RuntimeError(
                "SHA-256 mismatch while downloading " + dest.name
                + ". Check the source or model manifest."
            )
        os.replace(temp_path, dest)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def platform_archive_suffix():
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else (
        "x64" if machine in ("x86_64", "amd64", "x64") else None
    )
    if not arch:
        raise RuntimeError("Unsupported architecture: " + machine)
    if system == "linux":
        return system + "-" + arch, "-bin-ubuntu-" + arch + ".tar.gz"
    if system == "darwin":
        return "macos-" + arch, "-bin-macos-" + arch + ".tar.gz"
    if system == "windows":
        return system + "-" + arch, "-bin-win-cpu-" + arch + ".zip"
    raise RuntimeError("Unsupported operating system: " + system)


def _inside(root, name):
    root = root.resolve()
    output = (root / name).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        raise RuntimeError("Unsafe archive path: " + str(name))
    return output


def unpack(archive, target):
    """Extract safe files without allowing paths outside the cache."""
    target.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                dest = _inside(target, member.filename)
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with package.open(member) as source, dest.open("wb") as out:
                        shutil.copyfileobj(source, out)
        return

    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            dest = _inside(target, member.name)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                source = package.extractfile(member)
                if source is None:
                    raise RuntimeError("Incomplete tar archive entry: " + member.name)
                with source, dest.open("wb") as out:
                    shutil.copyfileobj(source, out)
                dest.chmod(member.mode & 0o777)
            elif member.issym():
                # Relative symlinks are required for some .so/.dylib files.
                if os.path.isabs(member.linkname):
                    raise RuntimeError("Absolute symlink is not allowed: " + member.name)
                link_target = (dest.parent / member.linkname).resolve()
                link_target.relative_to(target.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(member.linkname)
            elif member.islnk():
                # Tar hardlink: the target must remain inside the extraction directory.
                origin = _inside(target, member.linkname)
                if not origin.is_file():
                    raise RuntimeError("Invalid hardlink: " + member.name)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, dest)


def find_cli(directory):
    if not directory.is_dir():
        return None
    preferred = ("llama-cli.exe", "llama-cli", "llama.exe", "llama")
    for name in preferred:
        for candidate in directory.rglob(name):
            if candidate.is_file():
                return candidate
    return None


def load_runtimes():
    """Read a pinned CPU runtime manifest without calling the GitHub API."""
    with RUNTIMES.open("r", encoding="utf-8") as source:
        return json.load(source)


def ensure_runtime(offline=False):
    platform_key, _ = platform_archive_suffix()
    entry = load_runtimes()["llama.cpp"]
    version = entry["version"]
    asset = entry["platforms"].get(platform_key)
    if not asset:
        raise RuntimeError(
            "No prebuilt CPU runtime for {}. See {}".format(
                platform_key, entry["source"]
            )
        )
    install_dir = CACHE / "runtime" / version / platform_key
    binary = find_cli(install_dir)
    if binary:
        return binary
    if offline:
        raise RuntimeError("No local runtime found. Run once without --offline.")

    info("Preparing llama.cpp {} CPU runtime...".format(version))
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-", dir=CACHE) as workspace:
        workspace = Path(workspace)
        filename = asset["filename"]
        if filename != Path(filename).name or not (
            filename.endswith(".zip") or filename.endswith(".tar.gz")
        ):
            raise RuntimeError("Invalid CPU runtime archive name in runtimes.json.")
        archive = workspace / filename
        download(asset["url"], archive, asset["sha256"])
        extracted = workspace / "extracted"
        unpack(archive, extracted)
        binary = find_cli(extracted)
        if not binary:
            raise RuntimeError("The llama.cpp archive does not contain a compatible executable.")
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(extracted), str(install_dir))

    binary = find_cli(install_dir)
    if binary is None:
        raise RuntimeError("Could not install the llama.cpp runtime.")
    if platform.system() != "Windows":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def ensure_artifact(artifact, offline=False):
    """Fetch a verified GGUF file; shared by text models and vision projectors."""
    filename = artifact["filename"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.lower().endswith(".gguf")
    ):
        raise RuntimeError("Invalid GGUF filename in models.json.")
    checksum = artifact["sha256"]
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise RuntimeError("Invalid SHA-256 in models.json.")

    path = CACHE / "models" / filename
    if path.is_file() and file_sha256(path).lower() == checksum.lower():
        info("✓ Cached file: " + filename)
        return path
    if offline:
        raise RuntimeError("No verified file in the cache. Run once without --offline.")
    download(artifact["url"], path, checksum)
    return path


def ensure_model(model, offline=False):
    """Backwards-compatible helper for the main model weights."""
    return ensure_artifact(model, offline=offline)


def load_catalog():
    with CATALOG.open("r", encoding="utf-8") as source:
        models = json.load(source)
    if not isinstance(models, dict):
        raise RuntimeError("models.json must contain a JSON object of models.")
    for model_id, spec in models.items():
        if not isinstance(model_id, str) or not isinstance(spec, dict):
            raise RuntimeError("Invalid entry in models.json.")
        if spec.get("type") not in ("text", "vision"):
            raise RuntimeError("Unsupported model type for " + model_id)
        if spec.get("status", "available") not in ("available", "research"):
            raise RuntimeError("Invalid status for " + model_id)
        if (
            spec.get("status", "available") == "available"
            and spec.get("runtime", "llama.cpp") != "llama.cpp"
        ):
            raise RuntimeError(
                "Unsupported runtime for " + model_id
                + ". The current launcher supports llama.cpp-compatible GGUF files."
            )
        aliases = spec.get("aliases", [])
        if (
            not isinstance(aliases, list)
            or any(not isinstance(alias, str) or not alias for alias in aliases)
        ):
            raise RuntimeError("Invalid aliases for " + model_id)
        if (spec.get("status", "available") == "available"
                and spec.get("type") == "vision" and "mmproj" not in spec):
            raise RuntimeError("Vision model " + model_id + " requires an mmproj artifact.")
    return models


def resolve_model(models, model_id):
    """Resolve a canonical model ID or a declared alias."""
    if model_id in models:
        return model_id, models[model_id]
    matches = [
        (name, spec) for name, spec in models.items()
        if model_id in spec.get("aliases", [])
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("Ambiguous model alias: " + model_id)
    raise RuntimeError(
        "Unknown model: {}. Use --list to see available models.".format(model_id)
    )


def example_command(model_id, spec):
    """Create a runnable example for every model from the catalog."""
    if spec["type"] == "vision":
        return (
            'python run.py {} "Describe the main object, its color, '
            'shape, and setting." --image benchmark_image/example.png'
        ).format(model_id)
    return 'python run.py {} "Tell me something interesting."'.format(model_id)


def list_models(models):
    """List text and vision models separately, without network access."""
    print("Available models (CPU only):")
    for model_type, heading in (("text", "Text models"), ("vision", "Vision models")):
        entries = [
            (model_id, spec) for model_id, spec in models.items()
            if spec["type"] == model_type
        ]
        if not entries:
            continue
        print()
        print(heading + ":")
        for model_id, spec in entries:
            aliases = spec.get("aliases", [])
            alias_text = " | aliases: {}".format(", ".join(aliases)) if aliases else ""
            print("  {} | {} | {}{}".format(
                model_id, spec["parameters"], spec["name"], alias_text
            ))
            if spec.get("status", "available") == "research":
                print("    Status: research only (not runnable with the current CPU runner)")
            else:
                print("    Run: {}".format(example_command(model_id, spec)))
        print()


def build_command(engine, model, weights, prompt, args, projector=None):
    """Build an inference command from catalog metadata, not model-specific code."""
    command = [str(engine)]
    if engine.stem == "llama":
        command.append("cli")  # Unified CLI in newer llama.cpp versions
    command.extend([
        "--model", str(weights),
        "--gpu-layers", "0",
        "--ctx-size", str(args.ctx_size or model.get("context_size", 2048)),
        "--predict", str(args.max_tokens),
        "--show-timings",
    ])
    if args.threads is not None:
        command.extend(["--threads", str(args.threads)])
    if model.get("chat_template"):
        command.extend(["--chat-template", model["chat_template"]])
    if args.image:
        if projector is None:
            raise RuntimeError("A vision model with a projector is required for --image.")
        command.extend([
            "--mmproj", str(projector),
            "--no-mmproj-offload",
            "--image", str(Path(args.image).expanduser().resolve()),
        ])
    if prompt:
        command.extend([
            "--prompt", prompt,
            "--single-turn", "--no-display-prompt",
        ])
    return command



def run_inference(command, cwd, interactive=False):
    """Report native llama.cpp token throughput and measured wall-clock time.

    llama.cpp measures prompt/generation tokens per second for each answer.
    The Python timer covers process startup, model loading, prompt processing,
    and generation; it deliberately excludes the initial file downloads.
    """
    started = time.perf_counter()
    exit_code = subprocess.call(command, cwd=str(cwd))
    elapsed = time.perf_counter() - started
    if exit_code == 0:
        if interactive:
            info(
                "[CPU] Chat session: {:.2f} s total. "
                "Per-response generation throughput is shown above (tokens/s).".format(
                    elapsed
                )
            )
        else:
            info(
                "[CPU] Response time: {:.2f} s "
                "(includes model loading, prompt processing, and generation). "
                "Generation speed: see llama.cpp tokens/s above.".format(elapsed)
            )
    return exit_code


def main():
    parser = argparse.ArgumentParser(
        description="Run tiny open-source AI models on CPU. No Ollama or HF CLI.",
        epilog='Example: python run.py granite-350m "Tell me something"',
    )
    parser.add_argument("model", nargs="?", metavar="MODEL_ID", help="Model ID from models.json")
    parser.add_argument("prompt", nargs="*", metavar="PROMPT", help="Text prompt (omit for interactive chat)")
    parser.add_argument("--models", "--list", dest="list", action="store_true",
                        help="List all models with copy-and-paste test commands")
    parser.add_argument("--offline", action="store_true", help="Never download any files")
    parser.add_argument("--image", metavar="PATH", help="Image path (vision models only)")
    parser.add_argument("--max-tokens", type=int, default=192, help="Maximum output tokens")
    parser.add_argument("--ctx-size", type=int, default=None, help="Context size in tokens")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads")
    parser.add_argument("--model", dest="legacy_model", default=None, help=argparse.SUPPRESS)
    args = parser.parse_intermixed_args()

    if (
        args.max_tokens <= 0
        or (args.ctx_size is not None and args.ctx_size <= 0)
        or (args.threads is not None and args.threads <= 0)
    ):
        parser.error("--max-tokens, --ctx-size, and --threads must be positive.")

    models = load_catalog()
    if args.list:
        list_models(models)
        return 0

    # Keep older commands such as: python run.py --model granite "Hello".
    if args.legacy_model:
        if args.model:
            args.prompt.insert(0, args.model)
        args.model = args.legacy_model

    if not args.model:
        parser.print_help()
        print()
        list_models(models)
        return 0

    try:
        model_id, model = resolve_model(models, args.model)
    except RuntimeError as error:
        parser.error(str(error))

    if model.get("status", "available") == "research":
        parser.error(
            "Model '{}' is research-only, not runnable yet. {}".format(
                model_id, model.get("notes", "No supported runtime is configured.")
            )
        )

    if args.image:
        if model["type"] != "vision":
            parser.error("Model '{}' is text-only and cannot accept --image.".format(model_id))
        image_file = Path(args.image).expanduser()
        if not image_file.is_file():
            parser.error("Image file not found: " + str(image_file))

    engine = ensure_runtime(offline=args.offline)
    weights = ensure_model(model, offline=args.offline)

    projector = None
    if args.image and model["type"] == "vision":
        projector = ensure_artifact(model["mmproj"], offline=args.offline)

    prompt = " ".join(args.prompt)
    command = build_command(engine, model, weights, prompt, args, projector=projector)
    if not prompt:
        info("Local chat is ready. Press Ctrl+C or type /exit to quit.")
    info("▶ Running {} ({}, CPU-only)".format(model["name"], model_id))
    return run_inference(command, engine.parent, interactive=not bool(prompt))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
        info("Error: " + str(error))
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nInterrupted.")
        sys.exit(130)
