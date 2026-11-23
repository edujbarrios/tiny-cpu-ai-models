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
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CATALOG = ROOT / "models.json"
RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
USER_AGENT = "tiny-cpu-ai-models/0.1 (CPU-only, no-hf-cli)"


def info(message):
    print(message, file=sys.stderr, flush=True)


def request(url):
    if not url.startswith("https://"):
        raise RuntimeError("Only HTTPS downloads are allowed.")
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def get_json(url):
    with urllib.request.urlopen(request(url), timeout=45) as response:
        return json.load(response)


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

    if dest.is_file() and file_sha256(dest) == expected_sha256:
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


def select_asset(releases, suffix):
    """Solo paquetes CPU oficiales de llama.cpp; no CUDA, ROCm ni Vulkan."""
    for release in releases:
        for asset in release.get("assets", []):
            name = asset.get("name", "")
            if name.startswith("llama-") and name.endswith(suffix):
                digest = asset.get("digest", "")
                if not digest.startswith("sha256:"):
                    continue
                return asset["browser_download_url"], digest.removeprefix("sha256:")
    raise RuntimeError(
        "No verifiable llama.cpp CPU binary was found for " + suffix
        + ". See https://github.com/ggml-org/llama.cpp/releases"
    )


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


def ensure_runtime(offline=False):
    platform_key, suffix = platform_archive_suffix()
    install_dir = CACHE / "runtime" / platform_key
    binary = find_cli(install_dir)
    if binary:
        return binary
    if offline:
        raise RuntimeError("No local runtime found. Run once without --offline.")

    info("Preparing llama.cpp CPU runtime...")
    url, checksum = select_asset(get_json(RELEASES_URL), suffix)
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="llama-", dir=CACHE) as workspace:
        workspace = Path(workspace)
        extension = ".zip" if suffix.endswith(".zip") else ".tar.gz"
        archive = workspace / ("llama" + extension)
        download(url, archive, checksum)
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


def ensure_model(model, offline=False):
    filename = model["filename"]
    if Path(filename).name != filename or not filename.endswith(".gguf"):
        raise RuntimeError("Invalid GGUF filename in models.json.")
    path = CACHE / "models" / filename
    if path.is_file() and file_sha256(path).lower() == model["sha256"].lower():
        info("✓ Cached model: " + filename)
        return path
    if offline:
        raise RuntimeError("No verified model found in the cache. Run once without --offline.")
    download(model["url"], path, model["sha256"])
    return path


def load_catalog():
    with CATALOG.open("r", encoding="utf-8") as source:
        models = json.load(source)
    if not isinstance(models, dict):
        raise RuntimeError("models.json must contain a JSON object of models.")
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Tiny open models running on CPU. No Ollama or Hugging Face libraries."
    )
    parser.add_argument("prompt", nargs="*", help="Question (leave blank for interactive chat)")
    parser.add_argument("--model", default="granite", help="Model ID from models.json")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--offline", action="store_true", help="Do not download anything")
    parser.add_argument("--max-tokens", type=int, default=192, help="Maximum response tokens")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads")
    args = parser.parse_args()

    if args.max_tokens <= 0 or (args.threads is not None and args.threads <= 0):
        parser.error("--max-tokens and --threads must be positive.")
    models = load_catalog()
    if args.list:
        for key, model in models.items():
            print("{}: {} ({}, {})".format(
                key, model["name"], model["parameters"], model["license"]
            ))
        return 0
    if args.model not in models:
        parser.error("Unknown model: {}. Use --list.".format(args.model))

    model = models[args.model]
    engine = ensure_runtime(offline=args.offline)
    weights = ensure_model(model, offline=args.offline)

    command = [str(engine)]
    if engine.stem == "llama":
        command.append("cli")  # Unified CLI in newer llama.cpp versions
    command.extend([
        "--model", str(weights),
        "--gpu-layers", "0",
        "--ctx-size", "2048",
        "--predict", str(args.max_tokens),
        "--no-show-timings",
    ])
    if args.threads:
        command.extend(["--threads", str(args.threads)])
    if args.prompt:
        command.extend(["--prompt", " ".join(args.prompt), "--single-turn", "--no-display-prompt"])
    else:
        info("Local chat is ready. Press Ctrl+C or type /exit to quit.")

    info("▶ Running {} (CPU-only)".format(model["name"]))
    return subprocess.call(command, cwd=str(engine.parent))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
        info("Error: " + str(error))
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nInterrupted.")
        sys.exit(130)
