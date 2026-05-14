# Whisper Overlay

Small Windows overlay for microphone recording and Whisper transcription with:

- local `faster-whisper`
- automatic GPU/CPU runtime detection
- optional fallback to OpenAI-compatible transcription endpoint
- quick text snippets, copy helpers, and hotkey control

## Included Files

- `whisper_overlay.py`
- `requirements.txt`
- `setup_overlay_env.ps1`
- `run_overlay.bat`
- `.gitignore`

## Not Included

These are intentionally excluded from the GitHub upload:

- `.venv/`
- `models/`
- `logs/`
- `__pycache__/`
- user config file `~/.whisper_support.json`

## Requirements

- Windows
- Python 3.12
- NVIDIA GPU is optional
- for GPU mode, the setup script installs the same Python-side CUDA stack used by the local project environment

## Setup

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_overlay_env.ps1
```

This creates `.venv`, installs the Python dependencies, installs the CUDA-enabled `torch` wheel, and probes whether the Whisper GPU runtime is ready.

## Run

Use:

```bat
run_overlay.bat
```

## Local Models

The app supports local CTranslate2 Whisper models inside a `models/` folder. If no local model is found, `faster-whisper` can use a model name such as `large-v3` and download/cache it through the normal Hugging Face flow.

## Notes

- Runtime settings are stored in the user-local file `~/.whisper_support.json`.
- The default API endpoint is `http://localhost:8000/v1/audio/transcriptions`.
- If GPU libraries are unavailable, the overlay falls back to CPU instead of crashing.

## Publishing

For GitHub, upload only the contents of this folder. Do not upload your local `.venv`, cached models, logs, or home-directory config file.
