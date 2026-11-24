# Contributing

We welcome genuinely lightweight models that can run **locally on a CPU**.

## Requirements for new models

1. Model architecture and weights must have clear, open licenses. Do not add weights to the repository without permission.
2. Provide a direct **HTTPS** download and a verified SHA-256 checksum. No mandatory accounts, tokens, or additional applications.
3. Text models must use a GGUF format compatible with the public llama.cpp runtime.
4. Users should be able to run models with the same command: `python run.py MODEL_ID "Your question"`.
5. Test basic functionality, approximate RAM usage, and measured performance on an identified CPU. Do not invent benchmark results.
6. Multimodal models require separate validation of the vision encoder, projector files, and inference runtime.

## Adding a text model

Add an entry to `models.json`, for example:

```json
{
  "my-model": {
    "name": "Human-readable model name",
    "runtime": "llama.cpp",
    "type": "text",
    "parameters": "350M",
    "license": "Apache-2.0",
    "source": "Original publisher",
    "filename": "my-model-Q4_K_M.gguf",
    "url": "https://example.org/my-model-Q4_K_M.gguf",
    "sha256": "REPLACE_WITH_THE_REAL_64_CHARACTER_SHA256",
    "size_mb": 250,
    "context_size": 2048
  }
}
```

Do not invent a checksum. Calculate it from the original model file and verify its source.

Add the entry alongside existing models. The launcher discovers it automatically from the JSON catalog. A unique ID is required; optional `aliases` can provide shorter names. For compatible vision GGUF models, specify `"type": "vision"` and provide an `mmproj` artifact object with its own `filename`, `url`, and verified `sha256`. Other runtime formats require implementing a new runtime adapter; do not describe them as supported until tested.

Before opening a pull request, run `python -m unittest discover -s tests`, `python run.py --list`, and a real CPU inference with `python run.py MODEL_ID "Your question"`.

## Benchmark documentation

Record text-model results in [benchmarks/llms.md](benchmarks/llms.md) and vision-model results in [benchmarks/vlms.md](benchmarks/vlms.md). Include exact prompts, model outputs when available, hardware, quantization and measured throughput. Mark incomplete responses and distinguish descriptive detail from correctness. Keep the main README focused on the catalog and usage.

## Minimal-dependency philosophy

- Use the Python standard library for the launcher.
- No mandatory `pip install`, Hugging Face CLI, Ollama, Torch, or Docker.
- Download the native runtime directly from an open, verifiable source.
- Download weights and runtime on first use; support offline inference afterward.
- Clearly disclose where files are hosted and what is downloaded.

Thanks for helping make tiny, open-source AI easy to run.
