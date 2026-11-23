#!/usr/bin/env python3
"""Tiny CPU AI Models: descarga y ejecuta modelos GGUF en CPU sin dependencias pip."""

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
        raise RuntimeError("Solo se permiten descargas HTTPS.")
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
    """Descarga atómica, comprobación SHA-256 y reutilización de caché."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise RuntimeError("Falta un SHA-256 válido para verificar la descarga.")

    if dest.is_file() and file_sha256(dest) == expected_sha256:
        info("✓ Ya descargado: " + dest.name)
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    info("↓ Descargando " + dest.name)
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
                "SHA-256 incorrecto al descargar " + dest.name
                + ". Comprueba la fuente o el manifiesto."
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
        raise RuntimeError("Arquitectura no soportada: " + machine)
    if system == "linux":
        return system + "-" + arch, "-bin-ubuntu-" + arch + ".tar.gz"
    if system == "darwin":
        return "macos-" + arch, "-bin-macos-" + arch + ".tar.gz"
    if system == "windows":
        return system + "-" + arch, "-bin-win-cpu-" + arch + ".zip"
    raise RuntimeError("Sistema operativo no soportado: " + system)


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
        "No se encontró un binario CPU verificable de llama.cpp para " + suffix
        + ". Puedes consultar https://github.com/ggml-org/llama.cpp/releases"
    )


def _inside(root, name):
    root = root.resolve()
    output = (root / name).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        raise RuntimeError("Archivo comprimido inseguro: " + str(name))
    return output


def unpack(archive, target):
    """Extrae archivos válidos sin permitir rutas fuera de la caché."""
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
                    raise RuntimeError("Archivo tar incompleto: " + member.name)
                with source, dest.open("wb") as out:
                    shutil.copyfileobj(source, out)
                dest.chmod(member.mode & 0o777)
            elif member.issym():
                # Los enlaces relativos son necesarios para ciertas .so/.dylib.
                if os.path.isabs(member.linkname):
                    raise RuntimeError("Enlace absoluto no permitido: " + member.name)
                link_target = (dest.parent / member.linkname).resolve()
                link_target.relative_to(target.resolve())
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.symlink_to(member.linkname)
            elif member.islnk():
                # Hardlink del tar; el objetivo debe estar dentro del directorio.
                origin = _inside(target, member.linkname)
                if not origin.is_file():
                    raise RuntimeError("Hardlink inválido: " + member.name)
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
        raise RuntimeError("No hay motor local. Ejecuta una vez sin --offline.")

    info("Preparando motor llama.cpp (CPU)...")
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
            raise RuntimeError("El paquete de llama.cpp no contiene un ejecutable compatible.")
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        if install_dir.exists():
            shutil.rmtree(install_dir)
        shutil.move(str(extracted), str(install_dir))

    binary = find_cli(install_dir)
    if binary is None:
        raise RuntimeError("No se pudo instalar el motor de llama.cpp.")
    if platform.system() != "Windows":
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def ensure_model(model, offline=False):
    filename = model["filename"]
    if Path(filename).name != filename or not filename.endswith(".gguf"):
        raise RuntimeError("Nombre de archivo GGUF incorrecto en models.json.")
    path = CACHE / "models" / filename
    if path.is_file() and file_sha256(path).lower() == model["sha256"].lower():
        info("✓ Modelo en caché: " + filename)
        return path
    if offline:
        raise RuntimeError("No hay modelo verificado en caché. Ejecuta una vez sin --offline.")
    download(model["url"], path, model["sha256"])
    return path


def load_catalog():
    with CATALOG.open("r", encoding="utf-8") as source:
        models = json.load(source)
    if not isinstance(models, dict):
        raise RuntimeError("models.json debe contener un objeto de modelos.")
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Modelos abiertos y diminutos en CPU. Sin Ollama ni librerías de Hugging Face."
    )
    parser.add_argument("prompt", nargs="*", help="Pregunta (vacío: chat interactivo)")
    parser.add_argument("--model", default="granite", help="Identificador de models.json")
    parser.add_argument("--list", action="store_true", help="Mostrar los modelos disponibles")
    parser.add_argument("--offline", action="store_true", help="No descargar nada")
    parser.add_argument("--max-tokens", type=int, default=192, help="Máximo de tokens de respuesta")
    parser.add_argument("--threads", type=int, default=None, help="Hilos de CPU")
    args = parser.parse_args()

    if args.max_tokens <= 0 or (args.threads is not None and args.threads <= 0):
        parser.error("--max-tokens y --threads deben ser positivos.")
    models = load_catalog()
    if args.list:
        for key, model in models.items():
            print("{}: {} ({}, {})".format(
                key, model["name"], model["parameters"], model["license"]
            ))
        return 0
    if args.model not in models:
        parser.error("Modelo desconocido: {}. Usa --list.".format(args.model))

    model = models[args.model]
    engine = ensure_runtime(offline=args.offline)
    weights = ensure_model(model, offline=args.offline)

    command = [str(engine)]
    if engine.stem == "llama":
        command.append("cli")  # CLI unificada de versiones nuevas de llama.cpp
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
        info("Chat local listo. Usa Ctrl+C o /exit para salir.")

    info("▶ Ejecutando {} (solo CPU)".format(model["name"]))
    return subprocess.call(command, cwd=str(engine.parent))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, urllib.error.URLError) as error:
        info("Error: " + str(error))
        sys.exit(1)
    except KeyboardInterrupt:
        info("\nInterrumpido.")
        sys.exit(130)
