# Tiny CPU AI Models

**Small, open-source AI models. Local inference. CPU-only. Minimal setup.**

A growing, reproducible collection of lightweight models you can run with one consistent command:

```bash
python run.py MODEL_ID "Your prompt"
```

> **Project philosophy:** no mandatory Hugging Face Python packages, Ollama, PyTorch, Transformers, Docker, model managers, accounts, or API keys. Just **Python 3.9+**, the model you choose, and your CPU. The launcher downloads the appropriate native runtime and model weights automatically on first use. All inference is local.

## Quick start

```bash
git clone https://github.com/edujbarrios/tiny-cpu-ai-models.git
cd tiny-cpu-ai-models
python run.py granite-350m "Tell me something interesting."
```

That's it. On first use, the script automatically downloads the model and CPU runtime. No `pip install`, Ollama, or Hugging Face installation required.

**Python command:** If `python` is not available, use `python3` (common on Linux/macOS) or `py -3` (Windows).


## Command-line interface

```text
python run.py MODEL_ID "Your prompt" [OPTIONS]
```

```bash
python run.py --models
python run.py granite-350m "Explain how neural networks work."
python run.py granite-350m
python run.py granite-350m "Hello" --max-tokens 128
python run.py granite-350m "Hello" --threads 4
python run.py granite-350m "Hello" --ctx-size 2048
python run.py granite-350m "Hello" --offline
python run.py --help
```

Omit the prompt for interactive chat. `--offline` disables downloads after the model and runtime have been cached.

**One consistent interface for every registered model.** Use `python run.py --models` to list models and see a ready-to-copy test command for each one (`--list` is also supported). The launcher does not hard-code a particular model.

## Available models

```bash
python run.py --models
```

This prints every registered model with a **ready-to-copy test command**. The following models can be downloaded directly through the runner (no Hugging Face installation or Ollama required):

| Model ID | Model | Parameters | Type | Quantization | Download size | License |
| --- | --- | ---: | --- | --- | ---: | --- |
| `granite-350m` | [IBM Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m) | 350M | Text | Q4_K_M | 237 MB | Apache-2.0 |
| `smollm2-135m` | [SmolLM2 135M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | Text | Q4_K_M | 105 MB | Apache-2.0 |
| `smollm2-360m` | [SmolLM2 360M Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | Text | Q4_K_M | 271 MB | Apache-2.0 |
| `smolvlm2-256m` | [SmolVLM2 256M Video Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) | 256M | Vision | Q4_K_M | 131 MB model + 104 MB projector | Apache-2.0 |
| `qwen2.5-0.5b` | [Qwen2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) | 500M | Text | Q4_K_M | 491 MB | Apache-2.0 |

### Tested CPU performance

| Model | CPU | Quantization | Prompt processing | Generation speed | Response time |
| --- | --- | --- | ---: | ---: | ---: |
| `granite-350m` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 281.1 tok/s | 42.4 tok/s | 3.58 s |
| `smollm2-135m` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 528.9 tok/s | 111.6 tok/s | 2.43 s |

### Run each model

```bash
python run.py granite-350m "Tell me something interesting."
python run.py smollm2-135m "Tell me something interesting."
python run.py smollm2-360m "Tell me something interesting."
python run.py qwen2.5-0.5b "Tell me something interesting."
python run.py smolvlm2-256m "Describe this image." --image photo.jpg
```

For the vision example, replace `photo.jpg` with the path to an image on your computer. The visual projector is downloaded automatically when needed. The SmolVLM2 variant was trained for video and images; this launcher currently exposes **single-image** inference, not a video-processing flag.

**Download provenance:** Granite and Qwen use their publishers' GGUF repositories. SmolLM2 GGUFs come from third-party quantizations published by QuantFactory and LM Studio Community. SmolVLM2 GGUF and projector come from ggml-org. The `models.json` manifest includes direct HTTPS links and published SHA-256 checksums for all files. GGUF hosting does **not** mean Hugging Face software is installed.

The entry `granite` remains an alias for `granite-350m`. Registration confirms that the source files and their checksums are available; it does not imply that every combination of CPU/OS and model has been benchmarked.

## Adding another model

**No changes to `run.py` are required for new llama.cpp-compatible GGUF models.** Add a record to [`models.json`](models.json), using a unique model ID, a direct HTTPS download URL, and its verified SHA-256 checksum. Then use:

```bash
python run.py my-model "Tell me something."
```

For example, this is a *template*, not an installed model:

```json
{
  "my-model": {
    "name": "Example open model",
    "type": "text",
    "runtime": "llama.cpp",
    "parameters": "250M",
    "license": "Apache-2.0",
    "source": "Original publisher",
    "filename": "example-Q4_K_M.gguf",
    "url": "https://example.org/example-Q4_K_M.gguf",
    "sha256": "REPLACE_WITH_A_VERIFIED_64_CHARACTER_SHA256_HASH",
    "context_size": 2048
  }
}
```

Insert this as another entry alongside `granite-350m`, not as a replacement for the complete catalog. The launcher will fetch and verify the weights on first use.

### Vision models (VLMs)

The launcher supports **vision models compatible with llama.cpp**, including a separate visual projector. SmolVLM2 is the first registered VLM. Register `"type": "vision"` and an `"mmproj"` object containing its own `filename`, `url` and `sha256`:

```json
{
  "example-vlm": {
    "name": "Example vision model",
    "type": "vision",
    "runtime": "llama.cpp",
    "parameters": "256M",
    "license": "Apache-2.0",
    "filename": "example-vlm.gguf",
    "url": "https://example.org/example-vlm.gguf",
    "sha256": "REPLACE_WITH_A_VERIFIED_64_CHARACTER_SHA256_HASH",
    "mmproj": {
      "filename": "example-mmproj.gguf",
      "url": "https://example.org/example-mmproj.gguf",
      "sha256": "REPLACE_WITH_A_VERIFIED_64_CHARACTER_SHA256_HASH"
    }
  }
}
```

Once **actual compatible files and checksums** are provided, a registered VLM can use:

```bash
python run.py example-vlm "Describe this picture." --image photo.jpg
```

Vision support is an integration point; performance and compatibility should be validated per model and CPU platform. Some models require different runtimes or additional preprocessing, which will need separate adapters. The current version targets llama.cpp-compatible GGUF language models and supported VLM architectures; it is **not** a universal runtime for arbitrary model formats.

More details and contribution guidelines: [CONTRIBUTING.md](CONTRIBUTING.md).

## Direct downloads, without Hugging Face or Ollama installation

- **CPU runtime:** downloads a pinned official [llama.cpp](https://github.com/ggml-org/llama.cpp) CPU build from [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases), verified with SHA-256. Version, per-platform URLs and checksums live in [`runtimes.json`](runtimes.json); the launcher does not call the GitHub API.
- **Granite weights:** downloads the official [IBM Granite GGUF](https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF) directly over HTTPS and verifies SHA-256.
- **No Hugging Face installation:** while the official IBM model files are hosted on Hugging Face, there is **no Hugging Face CLI, Python package, or account requirement**.
- **Local cache:** downloaded files live in `.cache/`; inference runs locally and can be repeated offline with `--offline`.

The code uses only Python's standard library; no `pip install`, no Ollama, and no Docker. Pinning the runtime version and verifying checksums improves reproducibility.

## System requirements

- Python **3.9+**.
- Windows x64/ARM64, macOS x64/ARM64, or Linux x64/ARM64 **where matching CPU binaries are available**.
- No GPU required (`--gpu-layers 0` and no visual projector GPU offload).
- Allow at least 500 MB of storage for Granite and the runtime, plus temporary extraction space.
- RAM use depends on model size, context window, and system; it exceeds the size of the GGUF file.

Linux binaries provided upstream may require compatible system libraries. CPU inference speeds vary; benchmarks should state CPU, model, quantization, and measured tokens per second.

## Testing

```bash
python -m unittest discover -s tests -v
python run.py --models
python run.py granite-350m "Hello" --max-tokens 24
python run.py granite-350m "Hello again" --offline --max-tokens 24
```

GitHub Actions checks Python unit tests on Windows, macOS and Linux, and performs an end-to-end Granite CPU smoke test on Linux.

## Licenses

Original code in this repository is licensed under [MIT](LICENSE). IBM Granite weights are licensed under **Apache-2.0**; llama.cpp is licensed under **MIT**. We download the respective artifacts from their upstream publishers and do not bundle them in this repository.

---

**Mission:** `clone → choose a model → run`. Open, small, local AI should not require an entire AI software stack.
