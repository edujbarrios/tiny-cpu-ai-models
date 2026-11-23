# Tiny CPU AI Models

**Modelos de IA diminutos, abiertos y locales. Solo CPU. Sin complicaciones.**

La idea es que cualquier persona pueda probar modelos ligeros con unos pocos comandos, incluso sin GPU.

> **Filosofía:** no obligar a instalar **Hugging Face, Ollama, PyTorch, Transformers, Docker** ni gestores de modelos. El usuario necesita únicamente **Python 3.9+** y conexión a Internet para la primera descarga. No se requiere cuenta, token ni clave API. La inferencia se ejecuta en su ordenador, sin enviar prompts a servidores.

## ⚡ Empieza en 3 comandos

Linux / macOS (terminal):

```bash
git clone https://github.com/edujbarrios/tiny-cpu-ai-models.git
cd tiny-cpu-ai-models
python3 run.py "Explica qué es una CPU en una frase"
```

Windows (PowerShell):

```powershell
git clone https://github.com/edujbarrios/tiny-cpu-ai-models.git
cd tiny-cpu-ai-models
py -3 run.py "Explica qué es una CPU en una frase"
```

**Eso es todo.** El primer arranque descarga automáticamente el modelo y un binario de `llama.cpp` para CPU. Los siguientes reutilizan lo descargado.

Si no tienes Git, también puedes usar **Code → Download ZIP**, descomprimirlo y ejecutar `python run.py` desde la carpeta.

## 💬 Uso

```bash
python3 run.py                         # chat interactivo con Granite
python3 run.py "¿Qué es GGUF?"        # pregunta y termina
python3 run.py --list                 # modelos disponibles
python3 run.py --model granite "Hola" # selección explícita
python3 run.py --offline "Hola"       # sin conexión (tras primera descarga)
python3 run.py --max-tokens 256 "Hola"
python3 run.py --threads 4 "Hola"
```

En Windows puedes sustituir `python3` por `py -3`.

## Modelos

| Identificador | Modelo | Tipo | Parámetros | Cuantización | Descarga | Licencia |
| --- | --- | --- | --- | --- | --- | --- |
| `granite` | [IBM Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m) | Texto | 350M | GGUF Q4_K_M | ~237 MB | Apache-2.0 |

**Primero Granite.** Más adelante iremos incorporando otros modelos pequeños (incluidos VLM como SmolVLM cuando se valide una instalación igual de sencilla). No se presenta Granite como un modelo de visión: **solo procesa texto**.

## ¿De dónde sale lo que descarga?

La transparencia forma parte del proyecto:

- **Motor de inferencia:** [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT). Se descarga su **binario CPU precompilado** de [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases); no tienes que compilar nada en las plataformas compatibles.
- **Pesos del modelo:** descarga HTTPS directa del [repositorio GGUF oficial de IBM](https://huggingface.co/ibm-granite/granite-4.0-350m-GGUF), alojado actualmente en Hugging Face. **No hace falta instalar Hugging Face, usar su CLI ni registrarse**, pero **la descarga inicial sí depende de ese alojamiento**. No ocultamos esa diferencia.
- Los archivos se guardan en `.cache/` dentro de este proyecto, se verifica su SHA-256 y después puedes ejecutarlos **sin Internet** con `--offline`. Para usar otra fuente de descarga de pesos, se puede configurar la URL en `models.json` manteniendo la comprobación SHA-256.

No instalamos librerías Python adicionales y no utilizamos servicios de inferencia externos.

## Compatibilidad

**Primera versión:** Windows x64/ARM64, macOS Intel/Apple Silicon y Linux x64/ARM64 **si existen los binarios adecuados en GitHub Releases**. En Linux, los binarios oficiales de Ubuntu pueden requerir bibliotecas de sistema compatibles y no funcionan necesariamente en todas las distribuciones. Para arquitecturas que no tengan binarios precompilados, el script mostrará un error claro; no descarga instaladores extra ni compila silenciosamente.

- **Python:** 3.9 o posterior.
- **GPU:** no necesaria ni utilizada (`--gpu-layers 0`).
- **RAM:** depende del sistema, del contexto y del modelo; el GGUF pesa ~237 MB, pero la RAM real será mayor.
- **Espacio:** al menos 500 MB libres para el modelo y el motor, con margen adicional para descomprimir.
- **Sin conexión:** después de haber descargado modelo y motor, `--offline` bloquea las descargas.

La velocidad depende de cada CPU. No se prometen tokens/segundo sin medirlos.

## ¿Cómo añadir otro modelo?

La lista está en [`models.json`](models.json). Cada modelo GGUF de **texto** necesita nombre, URL de descarga directa, nombre de archivo y SHA-256. El mismo `run.py` lo podrá lanzar con `--model ID`. Para VLM harán falta, además, pruebas de compatibilidad con el proyector visual: **no basta con añadir el nombre de un GGUF**.

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) si quieres colaborar.

## Licencias

El código propio de este repositorio se distribuye bajo [MIT](LICENSE). El modelo IBM Granite se distribuye bajo **Apache-2.0** y el motor llama.cpp bajo **MIT**; se descargan desde sus respectivos autores, no se incluyen en el código fuente de este repositorio.

---

**Objetivo:** `clonar → ejecutar → probar`. Que lo difícil sea experimentar con la IA, no instalarla.
