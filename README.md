# Whisper Overlay

RU | [EN](#english)

Лёгкий Windows overlay для записи с микрофона и быстрой расшифровки речи через Whisper. Проект умеет работать локально через `faster-whisper`, автоматически использовать GPU при готовом runtime и безопасно откатываться на CPU, если CUDA недоступна.

## Возможности

- Локальная расшифровка через `faster-whisper`
- Автоопределение `GPU/CPU` без падений
- Поддержка CUDA-стека внутри `.venv`
- Live partial transcription во время записи
- Финальная расшифровка после остановки записи
- Горячая клавиша для старта и остановки
- Быстрые фразы / snippets
- Автокопирование результата
- Опциональный OpenAI-compatible endpoint fallback
- Простой запуск через `.bat`

## Как это выглядит

Overlay показывает:

- текущую модель Whisper
- текущий runtime (`CUDA float16` или `CPU int8`)
- live-статус записи
- промежуточный текст во время записи
- финальный текст после обработки

## Почему проект удобный

- Не требует отдельной системной установки CUDA/cuDNN, если Python GPU-стек уже установлен в `.venv`
- Не падает, если GPU-runtime недоступен: просто переключается на CPU
- Хранит пользовательские настройки отдельно от репозитория
- Подходит для диктовки, заметок, саппорта, быстрых ответов и голосового ввода

## Что входит в репозиторий

- `whisper_overlay.py` — основное приложение overlay
- `requirements.txt` — Python-зависимости
- `setup_overlay_env.ps1` — сборка локального окружения
- `run_overlay.bat` — быстрый запуск
- `.gitignore`

## Требования

- Windows
- Python 3.12
- NVIDIA GPU — опционально
- Для GPU-режима setup-скрипт ставит совместимый `torch + CUDA` runtime в `.venv`

## Установка

Открой PowerShell в папке проекта и выполни:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_overlay_env.ps1
```

Скрипт:

- создаёт `.venv`
- ставит Python-зависимости
- ставит CUDA-enabled `torch`
- проверяет, готов ли Whisper runtime для GPU

## Запуск

```bat
run_overlay.bat
```

## GPU и CPU

Если GPU runtime доступен, приложение использует `CUDA`.

Если нужные библиотеки недоступны или GPU-стек не готов, приложение автоматически переключается на `CPU`, не ломая workflow.

## Локальные модели

Если рядом есть папка `models/` с CTranslate2-моделью Whisper, overlay использует её.

Если локальной модели нет, можно работать с именем модели вроде `large-v3` и использовать стандартный механизм загрузки/кеша `faster-whisper`.

## Настройки

Пользовательские настройки сохраняются в:

```text
~/.whisper_support.json
```

Там хранятся:

- модель
- язык
- hotkey
- прозрачность
- snippet-фразы
- настройки backend/runtime

## Endpoint fallback

По умолчанию в конфиге есть endpoint:

```text
http://localhost:8000/v1/audio/transcriptions
```

Если нужен только локальный Whisper, можно оставить локальный backend и не использовать endpoint вообще
---

## English

Lightweight Windows overlay for microphone recording and fast Whisper transcription. The app can run locally with `faster-whisper`, automatically use GPU when the runtime is ready, and safely fall back to CPU when CUDA is unavailable.

## Features

- Local transcription with `faster-whisper`
- Automatic `GPU/CPU` runtime detection
- CUDA runtime support inside `.venv`
- Live partial transcription while recording
- Final transcription after stopping recording
- Hotkey-based start/stop
- Quick text snippets
- Auto-copy support
- Optional OpenAI-compatible endpoint fallback
- Simple `.bat` launcher

## What You See

The overlay shows:

- current Whisper model
- current runtime (`CUDA float16` or `CPU int8`)
- live recording status
- partial text while recording
- final transcription result

## Why It Is Handy

- No separate system-wide CUDA/cuDNN installation is required if the Python GPU stack is already present inside `.venv`
- No hard crash when GPU runtime is unavailable: it falls back to CPU
- User settings stay outside the repository
- Useful for dictation, notes, support work, quick replies, and general voice input

## Repository Contents

- `whisper_overlay.py` — main overlay application
- `requirements.txt` — Python dependencies
- `setup_overlay_env.ps1` — local environment setup
- `run_overlay.bat` — quick launcher
- `.gitignore`

## Requirements

- Windows
- Python 3.12
- NVIDIA GPU is optional
- For GPU mode, the setup script installs a compatible `torch + CUDA` runtime inside `.venv`

## Setup

Open PowerShell in the project folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_overlay_env.ps1
```

The script:

- creates `.venv`
- installs Python dependencies
- installs CUDA-enabled `torch`
- probes whether the Whisper GPU runtime is ready

## Run

```bat
run_overlay.bat
```

## GPU and CPU Behavior

If the GPU runtime is available, the app uses `CUDA`.

If the required libraries are missing or the GPU stack is not ready, the app automatically switches to `CPU` instead of breaking.

## Local Models

If a local `models/` folder contains a CTranslate2 Whisper model, the overlay will use it.

If no local model is available, you can still use a model name such as `large-v3` and rely on the standard `faster-whisper` download/cache flow.

## Settings

User settings are stored in:

```text
~/.whisper_support.json
```

This file can contain:

- model selection
- language
- hotkey
- opacity
- snippets
- backend/runtime settings

## Endpoint Fallback

The default config contains this endpoint:

```text
http://localhost:8000/v1/audio/transcriptions
```

If you want a fully local workflow, just keep using the local Whisper backend and ignore the endpoint
