# VLM CPU benchmarks

[← Back to the main README](../README.md)

## Test environment

- CPU recorded for the benchmark collection: Intel Core i7-1165G7 (Tiger Lake, 4 cores / 8 threads).
- CPU-only inference. The Qwen3.5 and LFM2-VL logs report llama.cpp build `b12651-f5e6915c2`; they do not independently identify the CPU.
- Timings include loading, prompt processing and generation, excluding initial downloads.

## Visual recognition benchmark

![Blue lemon-shaped object on a stone in a green meadow](../benchmark_image/example.png)

**Reference description (ground truth supplied by the image creator):** A bright blue lemon, deliberately textured to resemble a golf ball, resting on a rock in a sunny green meadow. The unusual color and hybrid appearance are intended to challenge visual object recognition. **The correct identification is a blue lemon, not a golf ball.**

**Benchmark prompt (English):** `Describe the main object, its color, shape, and setting.`

```bash
python run.py smolvlm2-256m "Describe the main object, its color, shape, and setting." --image benchmark_image/example.png
python run.py moondream-2b "Describe the main object, its color, shape, and setting." --image benchmark_image/example.png
python run.py lfm2-vl-450m "Describe the main object, its color, shape, and setting." --image benchmark_image/example.png
python run.py qwen3.5-0.8b "Describe the main object, its color, shape, and setting." --image benchmark_image/example.png --max-tokens 1024
```

All vision models download their weights and visual projectors automatically. Use the same image and exact prompt for comparison.

**Qwen3.5 retest completed:** The original run used the 192-token default and ended during thinking. With a 1024-token budget, the model completed both thinking and its final answer. The table reports the latest completed run; the earlier incomplete run is retained below for reference. Other models use the 192-token default, so the response-time comparison does not control for output length or reasoning budget.

## Performance and qualitative analysis

| Model | Quantization | Correct identification | Qualitative visual reasoning / description | Prompt processing | Generation speed | Response time |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `smolvlm2-256m` | Q4_K_M | ❌ | Basic; mistakes lemon for a ball | 48.6 tok/s | 65.4 tok/s | 20.15 s |
| `moondream-2b` | F16 | ✅ | Accurate, concise identification | 40.6 tok/s | 7.4 tok/s | 34.41 s |
| `lfm2-vl-450m` | Q4_0 | ❌ | **Standout descriptive detail**; recognizes fruit-like shape but invents a pair and plastic material | 157.9 tok/s | 51.4 tok/s | 13.80 s |
| `qwen3.5-0.8b` | Q4_K_M | ❌ | **Standout explicit reasoning**; detailed visual analysis, but identifies a blueberry instead of a lemon | 72.7 tok/s | 21.5 tok/s | 57.88 s |

**Qualitative observation:** Qwen3.5 and LFM2-VL stand out for the detail and organization of their visual analysis in these samples. Qwen3.5 explicitly reasons about the object, texture, color and scene; LFM2-VL produces a fluent and detailed scene description quickly. This is not a validated reasoning ranking: LFM2-VL misidentifies the number and nature of the object, while Qwen3.5's completed retest misidentifies the lemon as a blueberry. Qwen3.5 mentioned a lemon in its earlier incomplete run, but did not confirm it in a final answer. Moondream2 remains the only model with a correct completed identification in these samples.

LFM2-VL has the fastest recorded prompt processing and shortest end-to-end response time. SmolVLM2 has the highest generation throughput. Qwen3.5's generation rate and response time include thinking and final-answer generation. Its larger token budget makes end-to-end latency comparisons less controlled. A single image and limited runs do not establish general visual reasoning ability.

## Individual model outputs

### SmolVLM2 256M Instruct — Q4_K_M

The object is a blue-colored ball with a smooth texture, likely made from a natural material like concrete or plastic. It's placed on a rock, possibly in a grassy field or a rocky slope, with the sky visible in the background. The colors are vibrant and have a serene, natural aesthetic.

- Identification: incorrect.
- Prompt: 48.6 tok/s; generation: 65.4 tok/s; response time: 20.15 s.

### Moondream2 ~2B (2025-04-14) — F16

The main object is a blue lemon sitting on a rock in a grassy field. The lemon is round in shape and has a vibrant blue color, making it a visually striking and unique scene in the image. The setting appears to be a grassy field or pasture, providing a natural backdrop that complements the bright blue hue of the lemon.

- Identification: correct in the completed response.
- Prompt: 40.6 tok/s; generation: 7.4 tok/s; response time: 34.41 s.

### LFM2-VL 450M — Q4_0

The main object is a pair of blue, plastic fruit spheres. They are round and vibrant blue, standing out against the natural setting. The spheres are positioned on a large gray rock in a field of green grass. The background features a clear blue sky with some clouds and distant mountains. The image has a realistic feel, capturing the blue color of the spheres and the surrounding environment in great detail.

- Identification: incorrect (describes two plastic fruit spheres rather than one blue lemon).
- Detailed, coherent description of color and setting; some object details are wrong.
- Prompt: 157.9 tok/s; generation: 51.4 tok/s; response time: 13.80 s.

### Qwen3.5 0.8B — Q4_K_M

#### Completed retest — 1024-token budget

[Start thinking]

The user wants a description of the main object in the image.

**1. Identify the main object:**

The main object is a blue, spherical object sitting on a rock. It looks like a lemming, a blueberry, or a berry (maybe a 'Mammoth' berry). Given the context of nature and thespecific look, it's most likely a blueberry or a similar berry. However, the prompt just asks for the description, so identifying it as "a blueberry or similar berry" is accurate but not strictly necessary. I should focus on the visual characteristics.

**2. Analyze the color:**

The object is a vibrant, deep blue. It has a textured, almost cracked surface, suggesting it might be fresh or just very weathered, but the paint-like finish suggests it was painted. The color is very saturated.

**3. Analyze the shape:**

The object is clearly round. It has a distinct, somewhat lumpy texture, which gives it a natural shape like a berry, not a perfectly smooth sphere.

**4. Analyze the setting:**

* **Foreground:** A grey, textured rock.
* **Background:** A grassy field with green grass and some small yellow flowers.
* **Sky:** A bright blue sky with scattered white clouds.
* **Lighting:** It looks like a sunny day, casting a shadow beneath the blue object.

**5. Synthesize the description:**

* **Object:** A round, blue fruit (likely a blueberry or berry).
* **Color:** Vibrant, deep blue.
* **Shape:** Round with a textured, slightly uneven surface.
* **Setting:** Sitting on a grey rock in a green grassy field under a blue sky.

**Drafting the response:**

Start with the object: A round, blue fruit (likely a blueberry).  
Add color details: Vibrant blue.  
Add shape details: Round with a textured, uneven surface.  
Add setting details: Sitting on a grey rock in a green grassy field. Mention the sky and shadow for context.

Refining the description based on the visual evidence:

- The object is a blueberry (specifically a 'Mammoth' style or just a blueberry).
- It is sitting on a grey rock.
- The background is a green pasture.
- The sky is blue with white clouds.

Let's write it out clearly.  
"The main object is a round, blueberry (likely a 'Mammoth' variety) sitting on a rock."  
"Color: Vibrant, deep blue."  
"Shape: Round with a textured, uneven surface."  
"Setting: A grey rock in a green grassy field under a blue sky with white clouds."

This covers all the required points. I will formulate the final output based on this structure.

[End thinking]

**Final answer:**

The image features a single, round object that appears to be a blueberry (specifically a 'Mammoth' variety).

* **Object:** It is a round, blueberry with a textured, uneven surface that suggests it is fresh or perhaps freshly painted.
* **Color:** It is a vibrant, deep blue that stands out against the green and sky.
* **Setting:** The object is sitting on a grey, mossy rock in a lush green grassy field. Above it, there is a bright blue sky with scattered white clouds, and a shadow is cast directly beneath the berry.

**Evaluation:**

- Identification: incorrect (blueberry instead of blue lemon).
- Response: complete.
- Prompt: 72.7 tok/s; generation: 21.5 tok/s; response time: 57.88 s.

#### Earlier incomplete run — 192-token budget

[Start thinking]

The user wants a description of the main object in the image.

1.  **Identify the object:** It's a sphere. It looks like a lemon. The texture is bumpy and textured, not smooth like a plastic ball.
2.  **Identify the color:** It's a very vibrant, deep blue. It looks almost plasticized or coated. It contrasts sharply with the earthy tones of the rock and grass.
3.  **Identify the shape:** It is perfectly spherical.
4.  **Identify the setting:**
    *   It's sitting on a large, grey rock.
    *   The rock is on a grassy hill.
    *   There is a bright, sunny day with a clear blue sky.
    *   The background is a lush green landscape, slightly out of focus (bokeh), suggesting a landscape park or nature trail.

\*\*Drafting the

- Identification: lemon mentioned in thinking only; no final answer, so completed identification cannot be scored.
- Prompt: 63.5 tok/s; generation: 21.6 tok/s; response time: 36.26 s.

## Recording new runs

Record the exact prompt, model output (including whether it is complete), model quantization, hardware, runtime version, token limit, prompt and generation rates, and elapsed time. Evaluate object identification separately from descriptive detail. Keep prior observations unless they are explicitly superseded by a new run.
