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
python run.py --list
python run.py granite-350m "Explain how neural networks work."
python run.py granite-350m
python run.py granite-350m "Hello" --max-tokens 128
python run.py granite-350m "Hello" --threads 4
python run.py granite-350m "Hello" --ctx-size 2048
python run.py granite-350m "Hello" --offline
python run.py --help
```

Omit the prompt for interactive chat. `--offline` disables downloads after the model and runtime have been cached.

**One consistent interface for every registered model.** You choose a model using its catalog ID; the launcher does not hard-code Granite.

## CPU performance metrics

Every run prints **tokens per second** directly from llama.cpp:

```text
[ Prompt: ... t/s | Generation: ... t/s ]
```

For a one-shot prompt, the launcher additionally reports:

```text
[CPU] Response time: ... s (includes model loading, prompt processing, and generation).
```

- **Generation (tokens/s):** llama.cpp's measured token decoding throughput. This is the metric to compare model generation speed.
- **Prompt (tokens/s):** prompt-processing throughput, reported separately.
- **Response time (seconds):** wall-clock time from launching the inference process until it finishes, **excluding downloads but including model loading and prompt processing**. It is *not* decode-only time.
- **Interactive chat:** llama.cpp prints token speed for each answer; the launcher displays the total chat-session duration when you exit.

To compare models scientifically, keep the same prompt, context window, quantization, number of CPU threads, and machine. Repeat warmed-up runs because loading and disk caching influence total elapsed time.

## Available models

| Model ID | Model | Task | Parameters | Quantization | Approximate download | Model license |
| --- | --- | --- | --- | --- | --- | --- |
| `granite-350m` | [IBM Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m) | Text generation | 350M | GGUF Q4_K_M | 237 MB | Apache-2.0 |

`granite` is a backwards-compatible alias for `granite-350m`.

Additional small language models and VLMs will be added as their CPU inference paths are verified. **Only Granite is registered at present.** It is a text-only model.

## Community CPU results

These are **observed runs**, not standardized benchmarks or guaranteed performance. Different CPUs, prompts, output lengths, and settings can produce very different results.

| Model | Quantization | Prompt processing | Generation speed | Response time | CPU | Source |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `granite-350m` | Q4_K_M | 281.1 tok/s | **42.4 tok/s** | **3.58 s** | Intel Core i7-1165G7 (Tiger Lake; 4 cores / 8 threads, up to 4.7 GHz) | Community-reported run |

**Measurement notes:** Response time excludes initial model/runtime downloads but includes process startup, model loading, prompt processing, and generation. The original run's CPU model is now documented, but the inference thread count, exact prompt, generated token count, and context settings were not provided; therefore this sample is for reference only and should not be used as a controlled comparison. Contributions with full hardware and methodology details are welcome.

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

The launcher has a manifest-based path for **vision models that are compatible with llama.cpp**, including their separate visual projector. Register `"type": "vision"` and an `"mmproj"` object containing its own `filename`, `url` and `sha256`:

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

Vision support is a **generic integration point, not a claim that a specific VLM has been tested**. Some models require different runtimes or additional preprocessing, which will need separate adapters. The current version targets llama.cpp-compatible GGUF language models and supported VLM architectures; it is **not** a universal runtime for arbitrary model formats.

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
python run.py --list
python run.py granite-350m "Hello" --max-tokens 24
python run.py granite-350m "Hello again" --offline --max-tokens 24
```

GitHub Actions checks Python unit tests on Windows, macOS and Linux, and performs an end-to-end Granite CPU smoke test on Linux.

## Licenses

Original code in this repository is licensed under [MIT](LICENSE). IBM Granite weights are licensed under **Apache-2.0**; llama.cpp is licensed under **MIT**. We download the respective artifacts from their upstream publishers and do not bundle them in this repository.

---

**Mission:** `clone → choose a model → run`. Open, small, local AI should not require an entire AI software stack.
