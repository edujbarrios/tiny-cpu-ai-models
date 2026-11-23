# Contribuir

Buscamos modelos realmente ligeros que puedan usarse **localmente y en CPU**.

## Criterios de entrada

1. Modelo y pesos con licencia abierta y clara; no añadir pesos al repositorio sin permiso.
2. Descarga directa **HTTPS** y SHA-256 comprobado; sin logins, tokens ni aplicaciones obligatorias.
3. Formato GGUF compatible con la versión pública de llama.cpp (para texto).
4. Instrucciones que funcionen con el mismo comando `python run.py --model IDENTIFICADOR "Pregunta"`.
5. Comprobar funcionamiento, RAM aproximada y velocidad real en una CPU identificada. No inventar métricas.
6. Para modelos multimodales hay que validar procesador visual, archivos auxiliares y ejecución; requieren una integración específica.

## Añadir un modelo de texto

Edita `models.json` con una entrada como:

```json
{
  "mi-modelo": {
    "name": "Nombre legible",
    "type": "text",
    "parameters": "350M",
    "license": "Apache-2.0",
    "source": "Nombre del editor",
    "filename": "mi-modelo-Q4_K_M.gguf",
    "url": "https://ejemplo.org/mi-modelo-Q4_K_M.gguf",
    "sha256": "SHA256_DE_64_CARACTERES",
    "size_mb": 250
  }
}
```

No sustituyas `sha256` por un hash inventado: obtenlo del archivo original y verifica el origen.

Prueba localmente `python -m unittest discover -s tests`, `python run.py --list` y una inferencia antes de proponer el cambio.

## Filosofía de dependencias

- Biblioteca estándar de Python para el lanzador.
- Sin `pip install`, Hugging Face CLI, Ollama, Torch o Docker.
- El motor nativo se descarga directamente de un origen abierto y verificable.
- Pesos y motor se descargan solo en el primer uso; inferencia offline.
- Explicar claramente los servidores desde los que se descargan los archivos.
