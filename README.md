# Tiny CPU AI Models

**Tiny, open-source AI models. Local. CPU-only. No complicated setup.**

Run lightweight AI models on your own computer with just a few commands, even without a GPU.

> **Our philosophy:** users should not have to install **Hugging Face libraries, Ollama, PyTorch, Transformers, Docker, or a model manager**. All you need is **Python 3.9+** and an internet connection for the initial download. No accounts, API keys, or access tokens are required. Inference runs locally; your prompts are not sent to an inference server.

## ⚡ Get started in 3 commands

**Linux / macOS (Terminal)**

```bash
git clone https://github.com/edujbarrios/tiny-cpu-ai-models.git
cd tiny-cpu-ai-models
python3 run.py "Explain what a CPU is in one sentence"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/edujbarrios/tiny-cpu-ai-models.git
cd tiny-cpu-ai-models
py -3 run.py "Explain what a CPU is in one sentence"
```

**That's it.** On the first run, the script automatically downloads a small AI model and a prebuilt CPU-only `llama.cpp` executable. Later runs reuse the cached files.

Don't have Git? Use **Code → Download ZIP**, extract the project, and run `python run.py` in that directory.

## 💬 Usage

```bash
python3 run.py                            # Interactive chat with Granite
python3 run.py "What is GGUF?"           # Ask one question and exit
python3 run.py --list                    # List available models
python3 run.py --model granite "Hello"   # Choose a specific model
python3 run.py --offline "Hello"         # No downloads after first setup
python3 run.py --max-tokens 256 "Hello"  # Limit response length
python3 run.py --threads 4 "Hello"       # Use 4 CPU threads
```

On Windows, replace `python3` with `py -3`.

## Available models

| ID | Model | Type | Parameters | Quantization | Download size | License |
| --- | --- | --- | --- | --- | --- | --- |
| `granite` | [IBM Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m) | Text | 350M | GGUF Q4_K_M | ~237 MB | Apache-2.0 |

**We start with Granite.** More small models will follow, including vision-language models (VLMs) such as SmolVLM once an equally straightforward setup has been verified. Granite is a **text-only** model; it cannot interpret images.

## Where do the downloads come from?

Transparency matters:

- **Inference engine:** [llama.cpp](https://github.com/ggml-org/llama.cpp), licensed under MIT. The script downloads an official **prebuilt CPU executable** from [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases). No compilation is needed on supported platforms.
- **Model weights:** downloaded over HTTPS from [IBM's official GGUF repository](https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF), currently hosted on Hugging Face. **You do not need to install Hugging Face, run its CLI, or create an account.** The initial model download **does** rely on Hugging Face hosting; we make that distinction clear.
- **Local cache:** downloads are saved in this project's `.cache/` directory and verified with SHA-256. After setup, use `--offline` to prevent any further downloads. To use a different host for model files, update the direct download URL in `models.json` and retain SHA-256 verification.

No extra Python packages are installed, and no remote inference service is used.

## Requirements and compatibility

**Initial support:** Windows x64/ARM64, macOS Intel/Apple Silicon, and Linux x64/ARM64, **where suitable prebuilt binaries are published in GitHub Releases**. Official Ubuntu builds may depend on compatible system libraries, so not every Linux distribution will work out of the box. Unsupported architectures produce a clear error rather than installing additional tools or compiling in the background.

- **Python:** version 3.9 or newer.
- **GPU:** not required or used (`--gpu-layers 0`).
- **RAM:** depends on the model, operating system, and context length. The GGUF file is about 237 MB; actual RAM usage is higher.
- **Storage:** allow at least 500 MB for downloads and extracted files, plus temporary extraction space.
- **Offline usage:** available after the model and runtime have been downloaded successfully.

Inference speed depends on your CPU. We do not claim specific tokens-per-second figures without reproducible benchmarks.

## Adding more models

Models are listed in [`models.json`](models.json). Each new **text GGUF** model needs a name, direct download URL, filename, and SHA-256 hash. You can then run it using `python3 run.py --model MODEL_ID`.

VLMs require additional integration and testing for visual encoders and projector files; **adding a GGUF filename alone is not enough**.

See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute.

## Licenses

Original code in this repository is licensed under [MIT](LICENSE). IBM Granite model weights are licensed under **Apache-2.0**, and llama.cpp is licensed under **MIT**. They are downloaded from their respective publishers and are not included in this source repository.

---

**Goal:** `clone → run → experiment`. Getting started with tiny local AI should be easier than setting up an AI stack.
