# LLM CPU benchmarks

[← Back to the main README](../README.md)

## Test environment

- CPU: Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz).
- CPU-only inference using the repository's llama.cpp launcher.
- Response time includes model loading, prompt processing and generation, excluding initial downloads.
- The historical results below were recorded in the main README. Exact prompts, generated outputs and token counts were not preserved, so the example commands are reproduction starting points, not a claim that every historic run used identical settings.

## Performance

| Model | CPU | Quantization | Prompt processing | Generation speed | Response time |
| --- | --- | --- | ---: | ---: | ---: |
| `granite-350m` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 281.1 tok/s | 42.4 tok/s | 3.58 s |
| `smollm2-135m` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 528.9 tok/s | 111.6 tok/s | 2.43 s |
| `smollm2-360m` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 232.8 tok/s | 38.4 tok/s | 1.50 s |
| `qwen2.5-0.5b` | Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads, up to 4.7 GHz) | Q4_K_M | 180.4 tok/s | 30.9 tok/s | 3.62 s |

## Run the models

```bash
python run.py granite-350m "Tell me something interesting."
python run.py smollm2-135m "Tell me something interesting."
python run.py smollm2-360m "Tell me something interesting."
python run.py qwen2.5-0.5b "Tell me something interesting."
```

## Observations

- **SmolLM2 135M:** highest recorded prompt and generation throughput.
- **SmolLM2 360M:** shortest recorded response time, though output lengths were not controlled in the available records.
- **Granite 350M:** compact alternative with competitive generation throughput.
- **Qwen2.5 0.5B:** a larger multilingual instruction model; response quality has not been formally compared here.

These are speed measurements, not a benchmark of reasoning, RAG accuracy or instruction-following quality.

## Individual model results

| Model | Saved response |
| --- | --- |
| `granite-350m` | Not recorded |
| `smollm2-135m` | Not recorded |
| `smollm2-360m` | Not recorded |
| `qwen2.5-0.5b` | Not recorded |

For future runs, add the exact prompt, full model output, token limit, thread count, context size, CPU and measured performance here. Do not substitute generated examples for missing historical responses.
