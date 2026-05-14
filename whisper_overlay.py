#!/usr/bin/env python3
"""
WhisperSupport Overlay
Use `setup_overlay_env.ps1` to build the local GPU-ready environment.
"""

import tkinter as tk
from tkinter import simpledialog
import threading
import json, os, tempfile, time, importlib
import sys
import ctypes
import logging
import traceback
import faulthandler
import wave
from logging.handlers import RotatingFileHandler

CFG_PATH = os.path.join(os.path.expanduser("~"), ".whisper_support.json")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(APP_DIR, "logs")
LOG_PATH = os.path.join(LOG_DIR, f"whisper_overlay-{os.getpid()}.log")
CRASH_LOG_PATH = os.path.join(LOG_DIR, f"whisper_overlay-crash-{os.getpid()}.log")

KNOWN_MODELS = [
    "tiny", "tiny.en", "base", "base.en",
    "small", "small.en", "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "whisper-1",
]

DEFAULT_CFG = {
    "backend":    "auto",
    "endpoint":   "http://localhost:8000/v1/audio/transcriptions",
    "language":   "ru",
    "model":      "whisper-1",
    "device":     "auto",
    "compute_type": "auto",
    "samplerate": 16000,
    "auto_copy":  True,
    "opacity":    0.93,
    "hotkey":     "ctrl+shift+r",
    "chunk_sec":  3,
    "partial_context_sec": 8,
    "snippets": [
        "Здравствуйте! Чем могу помочь?",
        "Подождите, пожалуйста, секунду.",
        "Уточните детали, пожалуйста.",
        "Спасибо за обращение!",
        "Вопрос решён, всего доброго!",
        "Приношу извинения за неудобства.",
        "Передам информацию коллегам.",
        "Проверю и вернусь к вам.",
    ],
}

MODEL_ALIASES = {
    "whisper-1": "large-v3",
}

LOGGER = logging.getLogger("whisper_overlay")
_FAULT_LOG_FILE = None
_DLL_DIR_HANDLES = []


def _setup_logging():
    if LOGGER.handlers:
        return LOGGER

    os.makedirs(LOG_DIR, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False

    handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(threadName)s %(message)s")
    )
    LOGGER.addHandler(handler)
    LOGGER.info("logging started pid=%s", os.getpid())
    return LOGGER


def _enable_fault_logging():
    global _FAULT_LOG_FILE
    if _FAULT_LOG_FILE is not None:
        return
    _FAULT_LOG_FILE = open(CRASH_LOG_PATH, "a", encoding="utf-8", buffering=1)
    _FAULT_LOG_FILE.write(
        f"\n=== session start pid={os.getpid()} time={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    )
    faulthandler.enable(file=_FAULT_LOG_FILE, all_threads=True)


def _install_exception_hooks():
    def log_unhandled(exc_type, exc_value, exc_tb):
        LOGGER.critical(
            "unhandled exception\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )

    def log_thread_exception(args):
        LOGGER.critical(
            "unhandled thread exception thread=%s\n%s",
            getattr(args.thread, "name", "unknown"),
            "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)),
        )

    sys.excepthook = log_unhandled
    threading.excepthook = log_thread_exception


_setup_logging()
_enable_fault_logging()
_install_exception_hooks()


def _infer_model_name_from_path(path):
    normalized = path.lower().replace("\\", "/").replace("_", "-")
    for name in sorted(KNOWN_MODELS, key=len, reverse=True):
        if name.lower().replace("_", "-") in normalized:
            return name
    return os.path.basename(path)


def _list_local_models():
    models_root = os.path.join(APP_DIR, "models")
    found = []
    if not os.path.isdir(models_root):
        return found

    for root, _dirs, files in os.walk(models_root):
        if "model.bin" not in files or "tokenizer.json" not in files:
            continue
        found.append({
            "name": _infer_model_name_from_path(root),
            "path": root,
        })
    return found


def _register_windows_gpu_dll_dirs():
    if os.name != "nt":
        return []

    candidates = []
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    candidates.extend([
        os.path.join(site_packages, "torch", "lib"),
        os.path.join(site_packages, "ctranslate2"),
    ])

    for module_name in ("torch", "ctranslate2"):
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception:
            spec = None
        if spec and spec.origin:
            pkg_dir = os.path.dirname(spec.origin)
            candidates.append(pkg_dir)
            if module_name == "torch":
                candidates.append(os.path.join(pkg_dir, "lib"))

    seen = set()
    added = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(candidate)
        if candidate in seen or not os.path.isdir(candidate):
            continue
        seen.add(candidate)
        try:
            handle = os.add_dll_directory(candidate)
        except (AttributeError, FileNotFoundError, OSError):
            continue
        _DLL_DIR_HANDLES.append(handle)
        added.append(candidate)
    return added


def _required_cuda_library_groups():
    if os.name == "nt":
        return [
            [
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_cnn64_9.dll",
            ],
            [
                "cublas64_12.dll",
                "cublasLt64_12.dll",
                "cudnn_ops_infer64_8.dll",
                "cudnn_cnn_infer64_8.dll",
            ],
        ]
    if sys.platform == "darwin":
        return []
    return [
        [
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_cnn.so.9",
        ],
        [
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcudnn_ops_infer.so.8",
            "libcudnn_cnn_infer.so.8",
        ],
    ]


def _probe_cuda_runtime():
    dll_dirs = _register_windows_gpu_dll_dirs()
    try:
        import ctranslate2
    except Exception as err:
        return {
            "device_count": 0,
            "cuda_visible": False,
            "cuda_ready": False,
            "missing_libraries": [],
            "ready_group": [],
            "dll_dirs": dll_dirs,
            "error": str(err),
        }

    try:
        device_count = int(ctranslate2.get_cuda_device_count())
    except Exception as err:
        return {
            "device_count": 0,
            "cuda_visible": False,
            "cuda_ready": False,
            "missing_libraries": [],
            "ready_group": [],
            "dll_dirs": dll_dirs,
            "error": str(err),
        }

    missing = []
    ready_group = []
    if device_count > 0:
        loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
        for group in _required_cuda_library_groups():
            group_missing = []
            for lib_name in group:
                try:
                    loader(lib_name)
                except OSError as err:
                    group_missing.append(f"{lib_name}: {err}")
            if not group_missing:
                ready_group = group
                missing = []
                break
            if not missing:
                missing = group_missing

    return {
        "device_count": device_count,
        "cuda_visible": device_count > 0,
        "cuda_ready": device_count > 0 and bool(ready_group),
        "missing_libraries": missing,
        "ready_group": ready_group,
        "dll_dirs": dll_dirs,
        "error": "",
    }


def resolve_whisper_runtime(preferred_device="auto", preferred_compute_type="auto"):
    device = (preferred_device or "auto").lower()
    compute_type = (preferred_compute_type or "auto").lower()
    cuda_info = _probe_cuda_runtime()

    if device == "auto":
        resolved_device = "cuda" if cuda_info["cuda_ready"] else "cpu"
    elif device == "cuda" and not cuda_info["cuda_ready"]:
        resolved_device = "cpu"
    else:
        resolved_device = device

    if compute_type == "auto":
        resolved_compute_type = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute_type = compute_type

    return resolved_device, resolved_compute_type, cuda_info

def load_cfg():
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
                cfg = {**DEFAULT_CFG, **raw}
                # Migrate the legacy local-backend default to autodetect so
                # existing installs start using CUDA automatically when present.
                changed = False
                if (
                    cfg.get("device") == "cpu"
                    and cfg.get("compute_type") == "int8"
                    and cfg.get("backend", "auto") == "auto"
                ):
                    cfg["device"] = "auto"
                    cfg["compute_type"] = "auto"
                    changed = True
                if "partial_context_sec" not in raw:
                    changed = True
                local_models = _list_local_models()
                if local_models:
                    selected = {cfg.get("model"), MODEL_ALIASES.get(cfg.get("model", ""))}
                    available = {item["name"] for item in local_models}
                    if not (selected & available):
                        cfg["model"] = local_models[0]["name"]
                        changed = True
                if changed:
                    with open(CFG_PATH, "w", encoding="utf-8") as wf:
                        json.dump(cfg, wf, ensure_ascii=False, indent=2)
                return cfg
        except Exception:
            pass
    return dict(DEFAULT_CFG)

def save_cfg(cfg):
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        public_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        json.dump(public_cfg, f, ensure_ascii=False, indent=2)

C = {
    "bg":      "#141414", "bg2":    "#1e1e1e",
    "bg3":     "#2a2a2a", "border": "#333333",
    "text":    "#e8e8e8", "muted":  "#888888",
    "accent":  "#4f9eff", "rec":    "#e84040",
    "success": "#3fb950", "warn":   "#e3a135",
    "partial": "#aaaaaa", "btn":    "#252525",
}


class WhisperOverlay:
    def __init__(self):
        self.cfg = load_cfg()
        self.recording      = False
        self.audio_data     = []           # all chunks accumulated
        self._audio_lock    = threading.Lock()
        self.stream         = None
        self._hotkey_ok     = False
        self._stats_after   = None
        self._proc          = None
        self._rt_thread     = None         # realtime transcription thread
        self._last_partial  = ""           # last partial text shown
        self._hotkey_ref    = None
        self._local_model   = None
        self._local_model_source = None
        self._local_model_lock = threading.Lock()
        self._preload_thread = None
        self._preload_started = False
        self._preload_failed = False
        self._transcribe_lock = threading.Lock()
        self._partial_busy  = False
        self._partial_error = False
        self._runtime_notice = None
        self._last_backend  = None
        self._endpoint_down = False
        self.collapsed      = False
        self._drag_x = self._drag_y = 0

        self.root = tk.Tk()
        self.root.report_callback_exception = self._report_tk_exception
        self.root.title("Whisper")
        self.root.configure(bg=C["bg"])
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", self.cfg["opacity"])
        self.root.resizable(True, False)

        self._build_ui()
        self._try_register_hotkey()
        self._init_mouse()
        self._check_deps()
        self._init_stats()
        self._start_model_preload()

        self.root.bind("<FocusOut>", lambda e: self.root.attributes("-topmost", True))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._log(
            "overlay initialized",
            model=self.cfg.get("model"),
            backend=self.cfg.get("backend"),
            endpoint=self.cfg.get("endpoint"),
            device=self.cfg.get("device"),
            compute_type=self.cfg.get("compute_type"),
        )

    def _report_tk_exception(self, exc_type, exc_value, exc_tb):
        self._log_exception("tk callback", exc_value, exc_tb)
        self._set_status(f"tk error: {exc_value}", True)

    def _log(self, message, **fields):
        if fields:
            parts = [message]
            for key, value in fields.items():
                parts.append(f"{key}={value}")
            LOGGER.info(" | ".join(parts))
        else:
            LOGGER.info(message)

    def _log_exception(self, context, err=None, tb=None):
        if err is None:
            LOGGER.exception("%s failed", context)
            return
        if tb is None:
            tb = err.__traceback__
        LOGGER.error(
            "%s failed: %s\n%s",
            context,
            err,
            "".join(traceback.format_exception(type(err), err, tb)),
        )

    def _get_perf_snapshot(self):
        try:
            import psutil

            proc = self._proc or psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            cpu_pct = proc.cpu_percent(interval=None)
            return f"ram={mem_mb:.0f}MB cpu={cpu_pct:.0f}%"
        except Exception:
            return "ram=? cpu=?"

    def _get_audio_summary(self):
        try:
            with self._audio_lock:
                count = len(self.audio_data)
                frames = sum(len(chunk) for chunk in self.audio_data)
            sec = frames / float(self.cfg["samplerate"]) if frames else 0.0
            return count, frames, sec
        except Exception:
            return 0, 0, 0.0

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.minsize(320, 0)
        self._build_titlebar()
        self.content = tk.Frame(self.root, bg=C["bg"])
        self.content.pack(fill="both", expand=True)
        self._build_model_row()
        self._build_record_row()
        self._build_result_area()
        self._build_action_row()
        self._build_snippets()

    def _build_titlebar(self):
        bar = tk.Frame(self.root, bg=C["bg2"], height=28)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="Whisper", bg=C["bg2"], fg=C["muted"],
                 font=("Consolas", 10)).pack(side="left", padx=8)

        self.status_lbl = tk.Label(bar, text="готов", bg=C["bg2"],
                                   fg=C["muted"], font=("Consolas", 9))
        self.status_lbl.pack(side="left", padx=2)

        self.backend_lbl = tk.Label(bar, text="", bg=C["bg2"],
                                    fg=C["accent"], font=("Consolas", 8, "bold"))
        self.backend_lbl.pack(side="left", padx=6)

        self.stats_lbl = tk.Label(bar, text="", bg=C["bg2"],
                                  fg=C["muted"], font=("Consolas", 8))
        self.stats_lbl.pack(side="left", padx=6)

        for txt, cmd in [("x", self._on_close),
                         ("-", self._toggle_collapse),
                         ("o", self._open_settings)]:
            b = tk.Button(bar, text=txt, bg=C["bg2"], fg=C["muted"],
                          bd=0, padx=7, pady=0, font=("Consolas", 11),
                          cursor="hand2", activebackground=C["bg3"],
                          activeforeground=C["text"], command=cmd)
            b.pack(side="right")

        bar.bind("<Button-1>",  self._drag_start)
        bar.bind("<B1-Motion>", self._drag_motion)

    def _build_model_row(self):
        row = tk.Frame(self.content, bg=C["bg"])
        row.pack(fill="x", padx=10, pady=(8, 0))

        tk.Label(row, text="модель:", bg=C["bg"], fg=C["muted"],
                 font=("Consolas", 9)).pack(side="left")

        self.model_var = tk.StringVar(value=self.cfg["model"])
        vals = list(KNOWN_MODELS)
        if self.cfg["model"] not in vals:
            vals.insert(0, self.cfg["model"])
        om = tk.OptionMenu(row, self.model_var, *vals,
                           command=self._on_model_change)
        om.configure(bg=C["bg3"], fg=C["text"], activebackground=C["bg2"],
                     activeforeground=C["text"], relief="flat",
                     highlightthickness=0, bd=0,
                     font=("Consolas", 10), cursor="hand2")
        om["menu"].configure(bg=C["bg2"], fg=C["text"],
                              activebackground=C["accent"],
                              activeforeground="white", relief="flat")
        om.pack(side="left", padx=4)

        tk.Label(row, text="  язык:", bg=C["bg"], fg=C["muted"],
                 font=("Consolas", 9)).pack(side="left")

        self.lang_var = tk.StringVar(
            value=self.cfg["language"] if self.cfg["language"] else "авто")
        lm = tk.OptionMenu(row, self.lang_var,
                           *["авто", "ru", "en", "uk", "de", "fr", "ja", "zh"],
                           command=self._on_lang_change)
        lm.configure(bg=C["bg3"], fg=C["text"], activebackground=C["bg2"],
                     activeforeground=C["text"], relief="flat",
                     highlightthickness=0, bd=0,
                     font=("Consolas", 10), cursor="hand2")
        lm["menu"].configure(bg=C["bg2"], fg=C["text"],
                              activebackground=C["accent"],
                              activeforeground="white", relief="flat")
        lm.pack(side="left", padx=4)

    def _build_record_row(self):
        row = tk.Frame(self.content, bg=C["bg"], pady=8)
        row.pack(fill="x", padx=10)

        self.rec_btn = tk.Button(
            row, text="  REC", width=10,
            bg=C["btn"], fg=C["text"],
            activebackground=C["rec"], activeforeground="white",
            relief="flat", bd=0, pady=6,
            font=("Consolas", 11, "bold"), cursor="hand2",
            command=self.toggle_record,
        )
        self.rec_btn.pack(side="left")
        self._hover(self.rec_btn, C["bg3"], C["btn"])

        self.wave_lbl = tk.Label(row, text="", bg=C["bg"],
                                  fg=C["rec"], font=("Consolas", 14))
        self.wave_lbl.pack(side="left", padx=8)

        # Индикатор «live»
        self.live_lbl = tk.Label(row, text="", bg=C["bg"],
                                  fg=C["partial"], font=("Consolas", 9))
        self.live_lbl.pack(side="left")

    def _build_result_area(self):
        frame = tk.Frame(self.content, bg=C["border"])
        frame.pack(fill="x", padx=10, pady=(0, 6))

        self.result_txt = tk.Text(
            frame, height=4, wrap="word",
            bg=C["bg2"], fg=C["text"],
            insertbackground=C["text"],
            relief="flat", bd=1,
            font=("Consolas", 11),
            padx=8, pady=6,
            selectbackground=C["accent"],
        )
        self.result_txt.pack(fill="x")

        # Тег для частичного (серого) текста
        self.result_txt.tag_configure("partial", foreground=C["partial"])

    def _build_action_row(self):
        row = tk.Frame(self.content, bg=C["bg"], pady=4)
        row.pack(fill="x", padx=10)

        for txt, cmd in [
            ("copy",  self._copy),
            ("clean", self._clean),
            ("Aa",    self._capitalize),
            (".",     self._add_period),
            ("clr",   self._clear),
        ]:
            b = tk.Button(row, text=txt, bg=C["btn"], fg=C["text"],
                          relief="flat", bd=0, padx=8, pady=4,
                          font=("Consolas", 10), cursor="hand2",
                          activebackground=C["bg3"], command=cmd)
            b.pack(side="left", padx=2)
            self._hover(b, C["bg3"], C["btn"])

        self.auto_copy_var = tk.BooleanVar(value=self.cfg["auto_copy"])
        tk.Checkbutton(row, text="авто", variable=self.auto_copy_var,
                       bg=C["bg"], fg=C["muted"], selectcolor=C["bg2"],
                       activebackground=C["bg"], font=("Consolas", 9),
                       command=self._toggle_auto_copy).pack(side="right")

    def _build_snippets(self):
        self.snip_outer = tk.Frame(self.content, bg=C["bg"])
        self.snip_outer.pack(fill="x", padx=10, pady=(2, 10))
        self._render_snippets()

    def _render_snippets(self):
        for w in self.snip_outer.winfo_children():
            w.destroy()

        hdr = tk.Frame(self.snip_outer, bg=C["bg"])
        hdr.pack(fill="x")
        tk.Label(hdr, text="быстрые фразы", bg=C["bg"], fg=C["muted"],
                 font=("Consolas", 8)).pack(side="left", pady=(0, 3))
        tk.Button(hdr, text="[edit]", bg=C["bg"], fg=C["muted"],
                  relief="flat", bd=0, font=("Consolas", 8), cursor="hand2",
                  activebackground=C["bg"], activeforeground=C["accent"],
                  command=self._open_snippet_editor).pack(side="right")

        wrap = tk.Frame(self.snip_outer, bg=C["bg"])
        wrap.pack(fill="x")

        for s in self.cfg["snippets"]:
            short = s[:32] + "…" if len(s) > 32 else s
            b = tk.Button(wrap, text=short,
                          bg=C["bg3"], fg=C["muted"],
                          relief="flat", bd=0, padx=6, pady=3,
                          font=("Consolas", 9), cursor="hand2",
                          activebackground=C["bg2"], activeforeground=C["text"],
                          anchor="w", justify="left",
                          command=lambda t=s: self._use_snippet(t))
            b.pack(fill="x", pady=1)
            self._hover(b, C["bg2"], C["bg3"])

        tk.Button(wrap, text="+ добавить",
                  bg=C["bg"], fg=C["muted"],
                  relief="flat", bd=0, padx=6, pady=2,
                  font=("Consolas", 9), cursor="hand2",
                  command=self._add_snippet).pack(fill="x", pady=(3, 0))

    # ── Редактор сниппетов ────────────────────────────────────────────────────

    def _open_snippet_editor(self):
        win = tk.Toplevel(self.root)
        win.title("Редактор фраз")
        win.configure(bg=C["bg"])
        win.attributes("-topmost", True)
        win.geometry("430x400")

        tk.Label(win, text="Быстрые фразы", bg=C["bg"], fg=C["text"],
                 font=("Consolas", 11, "bold")).pack(pady=(10, 2))
        tk.Label(win, text="2x клик — редактировать    Del — удалить",
                 bg=C["bg"], fg=C["muted"], font=("Consolas", 8)).pack()

        lf = tk.Frame(win, bg=C["bg"])
        lf.pack(fill="both", expand=True, padx=12, pady=8)

        sb = tk.Scrollbar(lf, bg=C["bg3"], troughcolor=C["bg2"],
                          relief="flat", bd=0)
        sb.pack(side="right", fill="y")

        lb = tk.Listbox(lf, bg=C["bg2"], fg=C["text"],
                        selectbackground=C["accent"], selectforeground="white",
                        relief="flat", bd=0, font=("Consolas", 10),
                        activestyle="none", yscrollcommand=sb.set)
        lb.pack(fill="both", expand=True)
        sb.config(command=lb.yview)

        for s in self.cfg["snippets"]:
            lb.insert("end", s)

        def edit(e=None):
            sel = lb.curselection()
            if not sel:
                return
            idx = sel[0]
            new = simpledialog.askstring("Редактировать", "Текст:",
                                         initialvalue=lb.get(idx), parent=win)
            if new and new.strip():
                lb.delete(idx)
                lb.insert(idx, new.strip())
                lb.selection_set(idx)

        def delete(e=None):
            sel = lb.curselection()
            if sel:
                lb.delete(sel[0])

        def move(delta):
            sel = lb.curselection()
            if not sel:
                return
            idx, new_idx = sel[0], sel[0] + delta
            if new_idx < 0 or new_idx >= lb.size():
                return
            val = lb.get(idx)
            lb.delete(idx)
            lb.insert(new_idx, val)
            lb.selection_set(new_idx)

        lb.bind("<Double-Button-1>", edit)
        lb.bind("<Delete>", delete)

        ctrl = tk.Frame(win, bg=C["bg"])
        ctrl.pack(fill="x", padx=12, pady=(0, 8))

        for txt, fn in [("up", lambda: move(-1)), ("down", lambda: move(1)),
                        ("+ new", lambda: lb.insert("end",
                            simpledialog.askstring("Добавить", "Текст:", parent=win) or "")),
                        ("edit", edit), ("del", delete)]:
            tk.Button(ctrl, text=txt, bg=C["btn"], fg=C["text"],
                      relief="flat", bd=0, padx=10, pady=4,
                      font=("Consolas", 10), cursor="hand2",
                      command=fn).pack(side="left", padx=2)

        def save_close():
            self.cfg["snippets"] = [s for s in lb.get(0, "end") if s.strip()]
            save_cfg(self.cfg)
            win.destroy()
            self._render_snippets()

        tk.Button(ctrl, text="save", bg=C["accent"], fg="white",
                  relief="flat", bd=0, padx=14, pady=4,
                  font=("Consolas", 10), cursor="hand2",
                  command=save_close).pack(side="right")

    # ── Статистика ────────────────────────────────────────────────────────────

    def _init_stats(self):
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
            self._update_stats()
            self._log("stats enabled", perf=self._get_perf_snapshot())
        except ImportError:
            self.stats_lbl.configure(text="[pip install psutil]")

    def _update_stats(self):
        try:
            import psutil
            mem = self._proc.memory_info().rss / 1024 / 1024
            cpu = self._proc.cpu_percent(interval=None)
            color = C["success"] if mem < 100 else C["warn"] if mem < 300 else C["rec"]
            self.stats_lbl.configure(
                text=f"RAM {mem:.0f}M  CPU {cpu:.0f}%", fg=color)
        except Exception:
            pass
        self._stats_after = self.root.after(2000, self._update_stats)

    # ── Мышь M4 ───────────────────────────────────────────────────────────────

    def _init_mouse(self):
        try:
            from pynput import mouse

            def on_click(x, y, button, pressed):
                # Button.x1 = M4 (back), Button.x2 = M5 (forward)
                if button == mouse.Button.x1 and pressed:
                    self.root.after(0, self.toggle_record)

            listener = mouse.Listener(on_click=on_click)
            listener.daemon = True
            listener.start()
            self._set_status("M4 готов")
        except ImportError:
            self._set_status("нет pynput (pip install pynput)")
        except Exception as err:
            self._set_status(f"мышь: {err}", True)

    # ── Запись ────────────────────────────────────────────────────────────────

    def toggle_record(self):
        if self.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        try:
            import sounddevice as sd
        except ImportError:
            self._set_status("нет sounddevice!", True)
            return

        self.recording = True
        self.audio_data = []
        self._last_partial = ""
        self._partial_busy = False
        self._log(
            "recording started",
            samplerate=self.cfg.get("samplerate"),
            chunk_sec=self.cfg.get("chunk_sec"),
            partial_context_sec=self.cfg.get("partial_context_sec"),
            model=self.cfg.get("model"),
            backend=self._choose_backend(),
            perf=self._get_perf_snapshot(),
        )
        self.rec_btn.configure(bg=C["rec"], fg="white", text="  STOP")
        self._set_status("запись…")
        self._animate_wave()

        sr = self.cfg["samplerate"]

        def cb(indata, frames, t, status):
            if status:
                self._log("audio callback status", status=status)
            if self.recording:
                with self._audio_lock:
                    self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(samplerate=sr, channels=1,
                                     dtype="float32", callback=cb)
        self.stream.start()

        # Поток реалтайм-расшифровки
        self._rt_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        self._rt_thread.start()

    def _stop_record(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        count, frames, sec = self._get_audio_summary()
        self._log(
            "recording stopped",
            chunks=count,
            frames=frames,
            seconds=f"{sec:.2f}",
            perf=self._get_perf_snapshot(),
        )
        self.rec_btn.configure(bg=C["btn"], fg=C["text"], text="  REC")
        self.wave_lbl.configure(text="")
        self.live_lbl.configure(text="")
        self._set_status("финальная расшифровка…")
        threading.Thread(target=self._transcribe_full_after_stop, daemon=True).start()

    # ── Реалтайм ─────────────────────────────────────────────────────────────

    def _realtime_loop(self):
        """Каждые chunk_sec секунд расшифровывает только свежий хвост аудио."""
        chunk_sec = int(self.cfg.get("chunk_sec", 3))
        context_sec = max(chunk_sec, int(self.cfg.get("partial_context_sec", 8)))
        self._log("realtime loop started", chunk_sec=chunk_sec, context_sec=context_sec)
        while self.recording:
            time.sleep(chunk_sec)
            if not self.recording:
                break
            if self._choose_backend() == "local" and not self._is_local_model_ready():
                self._start_model_preload()
                self._log("partial skipped, model not ready yet", perf=self._get_perf_snapshot())
                continue
            if self._partial_busy:
                continue
            with self._audio_lock:
                if not self.audio_data:
                    continue
                snapshot = self._build_recent_snapshot(context_sec)
            self._partial_busy = True
            try:
                frames = sum(len(chunk) for chunk in snapshot)
                self._log(
                    "partial transcription start",
                    seconds=f"{frames / float(self.cfg['samplerate']):.2f}",
                    perf=self._get_perf_snapshot(),
                )
                self._transcribe_snapshot(snapshot, partial=True)
                self._partial_error = False
            except Exception:
                self._partial_error = True
            finally:
                self._partial_busy = False
        self._log("realtime loop stopped")

    def _transcribe_snapshot(self, snapshot, partial=False):
        try:
            if not snapshot:
                return
            frames = sum(len(chunk) for chunk in snapshot)
            start = time.perf_counter()
            with self._transcribe_lock:
                tmp = self._write_snapshot_wav(snapshot)
                try:
                    self._log(
                        "transcription request",
                        mode="partial" if partial else "final",
                        seconds=f"{frames / float(self.cfg['samplerate']):.2f}",
                        tmp=tmp,
                        perf=self._get_perf_snapshot(),
                    )
                    text, backend = self._transcribe_file(tmp)
                finally:
                    if tmp and os.path.exists(tmp):
                        os.unlink(tmp)
            self._log(
                "transcription done",
                mode="partial" if partial else "final",
                backend=backend,
                seconds=f"{frames / float(self.cfg['samplerate']):.2f}",
                elapsed=f"{time.perf_counter() - start:.2f}s",
                text_len=len(text),
                perf=self._get_perf_snapshot(),
            )

            if partial:
                self.root.after(0, lambda t=text, b=backend: self._show_partial(t, b))
            else:
                self.root.after(0, lambda t=text, b=backend: self._show_result(t, b))

        except Exception as err:
            self._log_exception("transcribe snapshot", err)
            if partial:
                self.root.after(0, lambda e=err: self._set_status(f"live: {e}", True))
            if not partial:
                self.root.after(0, lambda e=err: self._set_status(f"ошибка: {e}", True))

    def _transcribe_full(self):
        with self._audio_lock:
            snapshot = self.audio_data
            self.audio_data = []
        frames = sum(len(chunk) for chunk in snapshot)
        self._log(
            "final transcription start",
            seconds=f"{frames / float(self.cfg['samplerate']):.2f}",
            perf=self._get_perf_snapshot(),
        )
        self._transcribe_snapshot(snapshot, partial=False)

    def _transcribe_full_after_stop(self):
        # Let the realtime loop exit and finish any in-flight partial pass first.
        rt = self._rt_thread
        if rt and rt.is_alive():
            self._log("waiting realtime thread", timeout=max(1, int(self.cfg.get("chunk_sec", 3)) + 1))
            rt.join(timeout=max(1, int(self.cfg.get("chunk_sec", 3)) + 1))
        while self._partial_busy:
            self._log("waiting partial pass", perf=self._get_perf_snapshot())
            time.sleep(0.05)
        if self._choose_backend() == "local" and not self._is_local_model_ready():
            self._start_model_preload()
            pt = self._preload_thread
            if pt and pt.is_alive():
                self._log("waiting model preload before final", perf=self._get_perf_snapshot())
                pt.join()
        self._transcribe_full()

    def _build_recent_snapshot(self, context_sec):
        sr = int(self.cfg["samplerate"])
        target_frames = max(sr, int(sr * context_sec))
        taken = 0
        recent = []
        for chunk in reversed(self.audio_data):
            recent.append(chunk)
            taken += len(chunk)
            if taken >= target_frames:
                break
        recent.reverse()
        return recent

    def _write_snapshot_wav(self, snapshot):
        import numpy as np

        sr = int(self.cfg["samplerate"])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name

        with wave.open(tmp, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            for chunk in snapshot:
                chunk_i16 = np.clip(chunk, -1.0, 1.0)
                chunk_i16 = (chunk_i16 * 32767).astype(np.int16, copy=False)
                wf.writeframes(chunk_i16.tobytes())

        self._log("wav snapshot written", path=tmp, size=os.path.getsize(tmp))
        return tmp

    # ── Отображение ───────────────────────────────────────────────────────────

    def _show_partial(self, text, backend=None):
        """Обновляет текст во время записи (серым, с индикатором)."""
        if not self.recording:
            return
        self._last_partial = text
        self._update_backend_label(backend)
        self.result_txt.configure(state="normal")
        self.result_txt.delete("1.0", "end")
        self.result_txt.insert("1.0", text, "partial")
        label = f"● live {backend}" if backend else "● live"
        self.live_lbl.configure(text=label)

    def _show_result(self, text, backend=None):
        """Финальный результат — белый текст."""
        self._update_backend_label(backend)
        self.result_txt.configure(state="normal")
        self.result_txt.delete("1.0", "end")
        self.result_txt.insert("1.0", text)
        self.live_lbl.configure(text="")
        self._set_status(f"готово ✓ {backend}" if backend else "готово ✓")
        if self.auto_copy_var.get():
            self._copy(silent=True)

    def _animate_wave(self):
        frames = ["▁▂▃▄▅", "▂▃▄▅▄", "▃▄▅▄▃", "▄▅▄▃▂", "▅▄▃▂▁"]
        self._wi = 0
        def tick():
            if self.recording:
                self.wave_lbl.configure(text=frames[self._wi % len(frames)])
                self._wi += 1
                self.root.after(150, tick)
        tick()

    # ── Текстовые операции ────────────────────────────────────────────────────

    def _copy(self, silent=False):
        t = self.result_txt.get("1.0", "end").strip()
        if not t:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(t)
        if not silent:
            self._set_status("скопировано!")

    def _clean(self):
        import re
        t = self.result_txt.get("1.0", "end")
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\s([.,!?:;])", r"\1", t).strip()
        self.result_txt.delete("1.0", "end")
        self.result_txt.insert("1.0", t)

    def _capitalize(self):
        t = self.result_txt.get("1.0", "end").strip()
        if t:
            self.result_txt.delete("1.0", "end")
            self.result_txt.insert("1.0", t[0].upper() + t[1:])

    def _add_period(self):
        t = self.result_txt.get("1.0", "end").rstrip("\n").rstrip()
        if t and t[-1] not in ".!?":
            self.result_txt.delete("1.0", "end")
            self.result_txt.insert("1.0", t + ".")

    def _clear(self):
        self.result_txt.delete("1.0", "end")
        self._last_partial = ""
        self._set_status("готов")

    def _use_snippet(self, text):
        cur = self.result_txt.get("1.0", "end").strip()
        full = (cur + " " + text).strip() if cur else text
        self.result_txt.delete("1.0", "end")
        self.result_txt.insert("1.0", full)
        self.root.clipboard_clear()
        self.root.clipboard_append(full)
        self._set_status("вставлено + скопировано!")

    def _add_snippet(self):
        val = simpledialog.askstring("Добавить", "Текст фразы:", parent=self.root)
        if val and val.strip():
            self.cfg["snippets"].append(val.strip())
            save_cfg(self.cfg)
            self._render_snippets()

    # ── Настройки ─────────────────────────────────────────────────────────────

    def _open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Настройки")
        win.configure(bg=C["bg"])
        win.attributes("-topmost", True)
        win.resizable(False, False)
        win.geometry("400x310")

        tk.Label(win, text="Настройки", bg=C["bg"], fg=C["text"],
                 font=("Consolas", 12, "bold")).pack(pady=10)

        fields = []
        for label, key in [("Endpoint",     "endpoint"),
                           ("Горячая кл",  "hotkey"),
                           ("Прозрачность","opacity"),
                           ("Чанк (сек)",  "chunk_sec"),
                           ("Live окно",   "partial_context_sec"),
                           ("Device",      "device"),
                           ("Compute",     "compute_type")]:
            row = tk.Frame(win, bg=C["bg"])
            row.pack(fill="x", padx=16, pady=4)
            tk.Label(row, text=label, width=14, anchor="w",
                     bg=C["bg"], fg=C["muted"],
                     font=("Consolas", 10)).pack(side="left")
            var = tk.StringVar(value=str(self.cfg[key]))
            tk.Entry(row, textvariable=var, bg=C["bg2"], fg=C["text"],
                     insertbackground=C["text"], relief="flat", bd=1,
                     font=("Consolas", 10)).pack(side="left", fill="x", expand=True)
            fields.append((var, key))

        def apply():
            for var, key in fields:
                v = var.get().strip()
                if key == "opacity":
                    try:
                        v = float(v)
                        self.root.attributes("-alpha", v)
                    except ValueError:
                        v = 0.93
                elif key == "chunk_sec":
                    try:
                        v = max(1, int(v))
                    except ValueError:
                        v = 3
                elif key == "partial_context_sec":
                    try:
                        v = max(2, int(v))
                    except ValueError:
                        v = 8
                elif key in {"device", "compute_type"}:
                    v = v.lower() or "auto"
                self.cfg[key] = v
            self._release_local_model()
            save_cfg(self.cfg)
            self._try_register_hotkey()
            self._update_backend_label()
            self._start_model_preload()
            win.destroy()
            self._set_status("сохранено!")

        tk.Button(win, text="Сохранить", command=apply,
                  bg=C["accent"], fg="white", relief="flat", bd=0,
                  padx=16, pady=6, font=("Consolas", 10),
                  cursor="hand2").pack(pady=10)

    # ── Служебное ─────────────────────────────────────────────────────────────

    def _on_model_change(self, val):
        self.cfg["model"] = val
        self._release_local_model()
        save_cfg(self.cfg)
        self._update_backend_label()
        self._start_model_preload()
        self._set_status(f"модель: {val}")

    def _on_lang_change(self, val):
        self.cfg["language"] = "" if val == "авто" else val
        save_cfg(self.cfg)

    def _set_status(self, text, error=False):
        ok_words = ("✓", "сохран", "скопир", "вставл", "готов", "M4")
        fg = (C["rec"] if error
              else C["success"] if any(w in text for w in ok_words)
              else C["muted"])
        self.status_lbl.configure(text=text, fg=fg)

    def _toggle_auto_copy(self):
        self.cfg["auto_copy"] = self.auto_copy_var.get()
        save_cfg(self.cfg)

    def _toggle_collapse(self):
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.content.pack_forget()
        else:
            self.content.pack(fill="both", expand=True)

    def _drag_start(self, e):
        self._drag_x = e.x_root - self.root.winfo_x()
        self._drag_y = e.y_root - self.root.winfo_y()

    def _drag_motion(self, e):
        self.root.geometry(f"+{e.x_root - self._drag_x}+{e.y_root - self._drag_y}")

    def _hover(self, btn, on, off):
        btn.bind("<Enter>", lambda e: btn.configure(bg=on))
        btn.bind("<Leave>", lambda e: btn.configure(bg=off))

    def _try_register_hotkey(self):
        try:
            import keyboard
            if self._hotkey_ref is not None:
                try:
                    keyboard.remove_hotkey(self._hotkey_ref)
                except Exception:
                    pass
            self._hotkey_ref = keyboard.add_hotkey(
                self.cfg["hotkey"],
                lambda: self.root.after(0, self.toggle_record),
            )
            self._hotkey_ok = True
        except ImportError:
            pass
        except Exception as err:
            self._set_status(f"хоткей: {err}", True)

    def _check_deps(self):
        missing = [lib for lib in ["sounddevice", "numpy", "scipy", "requests", "pynput"]
                   if not importlib.util.find_spec(lib)]
        if missing:
            self._set_status(f"pip install {' '.join(missing)}", True)
            return

        self._update_backend_label()
        if self._should_prefer_local():
            device, compute_type = self._resolve_runtime_device()
            self._set_status(f"локальная модель: {device}/{compute_type}")

    def _transcribe_file(self, wav_path):
        backend = self._choose_backend()
        errors = []
        self._log("backend selected", backend=backend, wav=wav_path, perf=self._get_perf_snapshot())

        if backend == "local":
            try:
                text = self._transcribe_with_local(wav_path)
                self._last_backend = "local"
                return text, "local"
            except Exception as err:
                errors.append(("local", err))
                if self.cfg.get("backend", "auto") != "auto":
                    raise

        try:
            text = self._transcribe_with_endpoint(wav_path)
            self._endpoint_down = False
            self._last_backend = "api"
            return text, "api"
        except Exception as err:
            errors.append(("api", err))
            self._endpoint_down = True
            if self.cfg.get("backend", "auto") == "auto":
                text = self._transcribe_with_local(wav_path)
                self._last_backend = "local"
                return text, "local"

        names = ", ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(names)

    def _transcribe_with_endpoint(self, wav_path):
        import requests

        lang = self.cfg["language"]
        self._log("endpoint request", endpoint=self.cfg.get("endpoint"), model=self.cfg.get("model"))
        with open(wav_path, "rb") as f:
            data = {"model": self.cfg["model"]}
            if lang and lang != "авто":
                data["language"] = lang
            r = requests.post(
                self.cfg["endpoint"],
                files={"file": ("audio.wav", f, "audio/wav")},
                data=data,
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("text", "").strip()

    def _transcribe_with_local(self, wav_path):
        model = self._get_local_model()
        language = self.cfg["language"] or None
        if language == "авто":
            language = None
        device, compute_type = self._current_runtime_device()
        self._log(
            "local transcription start",
            model=self.cfg.get("model"),
            resolved_model=self._local_model_source,
            device=device,
            compute_type=compute_type,
            language=language or "auto",
        )
        segments, _info = model.transcribe(
            wav_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _choose_backend(self):
        backend = self.cfg.get("backend", "auto")
        if backend in {"local", "endpoint"}:
            return "api" if backend == "endpoint" else "local"

        if self._should_prefer_local():
            return "local"
        return "api"

    def _should_prefer_local(self):
        if self._endpoint_down and self._can_use_local():
            return True
        endpoint = self.cfg.get("endpoint", "").strip()
        if endpoint == DEFAULT_CFG["endpoint"] and self._can_use_local():
            return True
        return False

    def _can_use_local(self):
        if not importlib.util.find_spec("faster_whisper"):
            return False
        try:
            self._resolve_local_model_source()
            return True
        except Exception:
            return False

    def _resolve_local_model_source(self):
        configured = self.cfg.get("local_model_path", "").strip()
        if configured:
            path = os.path.abspath(os.path.expanduser(configured))
            if os.path.isfile(os.path.join(path, "model.bin")):
                return path
            raise FileNotFoundError(f"не найдена модель: {path}")

        model_name = MODEL_ALIASES.get(self.cfg["model"], self.cfg["model"])
        models_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        if os.path.isdir(models_root):
            found = self._find_model_dir(models_root, model_name)
            if found:
                return found
        return model_name

    def _find_model_dir(self, root_dir, model_name):
        hint = model_name.lower().replace("_", "-")
        for root, _dirs, files in os.walk(root_dir):
            if "model.bin" not in files:
                continue
            if "tokenizer.json" not in files:
                continue
            normalized = root.lower().replace("_", "-")
            if hint in normalized:
                return root
        return None

    def _get_local_model(self):
        source = self._resolve_local_model_source()
        device, compute_type = self._resolve_runtime_device()
        with self._local_model_lock:
            if (
                self._local_model is not None
                and self._local_model_source == source
                and self.cfg.get("_runtime_device") == device
                and self.cfg.get("_runtime_compute_type") == compute_type
            ):
                return self._local_model

            _register_windows_gpu_dll_dirs()
            from faster_whisper import WhisperModel

            self._log(
                "loading local model",
                source=source,
                device=device,
                compute_type=compute_type,
                perf=self._get_perf_snapshot(),
            )
            try:
                self._local_model = WhisperModel(
                    source,
                    device=device,
                    compute_type=compute_type,
                )
            except RuntimeError as err:
                fallback = self._get_runtime_fallback(err, device, compute_type)
                if not fallback:
                    raise
                device, compute_type = fallback
                self._log(
                    "retry loading local model on fallback runtime",
                    source=source,
                    device=device,
                    compute_type=compute_type,
                )
                self._local_model = WhisperModel(
                    source,
                    device=device,
                    compute_type=compute_type,
                )
            self._local_model_source = source
            self.cfg["_runtime_device"] = device
            self.cfg["_runtime_compute_type"] = compute_type
            self._log("local model loaded", source=source, device=device, compute_type=compute_type)
            return self._local_model

    def _resolve_runtime_device(self):
        device, compute_type, cuda_info = resolve_whisper_runtime(
            self.cfg.get("device"),
            self.cfg.get("compute_type"),
        )
        self._log_runtime_notice(device, compute_type, cuda_info)
        return device, compute_type

    def _current_runtime_device(self):
        device = self.cfg.get("_runtime_device")
        compute_type = self.cfg.get("_runtime_compute_type")
        if device and compute_type:
            return device, compute_type
        return self._resolve_runtime_device()

    def _log_runtime_notice(self, device, compute_type, cuda_info):
        if device != "cpu" or not cuda_info.get("cuda_visible"):
            return
        missing = cuda_info.get("missing_libraries") or []
        if not missing and not cuda_info.get("error"):
            return

        details = ", ".join(missing) if missing else cuda_info.get("error", "unknown error")
        notice = f"cuda unavailable, fallback to cpu/{compute_type}: {details}"
        if notice == self._runtime_notice:
            return
        self._runtime_notice = notice
        self._log(
            "runtime fallback",
            requested_device=self.cfg.get("device"),
            requested_compute_type=self.cfg.get("compute_type"),
            resolved_device=device,
            resolved_compute_type=compute_type,
            details=details,
        )

    def _get_runtime_fallback(self, err, device, compute_type):
        if device != "cuda":
            return None

        lowered = str(err).lower()
        cuda_markers = ("cudnn", "cublas", "cuda", "cufft", "curand", "driver")
        if not any(marker in lowered for marker in cuda_markers):
            return None

        fallback_compute_type = "int8"
        if compute_type not in {"auto", "float16", "int8_float16"}:
            fallback_compute_type = compute_type

        self._log(
            "local model runtime failed, falling back to cpu",
            original_device=device,
            original_compute_type=compute_type,
            fallback_compute_type=fallback_compute_type,
            error=err,
        )
        return "cpu", fallback_compute_type

    def _release_local_model(self):
        with self._local_model_lock:
            self._local_model = None
            self._local_model_source = None
            self.cfg.pop("_runtime_device", None)
            self.cfg.pop("_runtime_compute_type", None)
        self._preload_started = False
        self._preload_failed = False
        self._preload_thread = None
        self._runtime_notice = None

    def _is_local_model_ready(self):
        with self._local_model_lock:
            return self._local_model is not None

    def _start_model_preload(self):
        if self._choose_backend() != "local":
            return
        if self._preload_started and self._preload_thread and self._preload_thread.is_alive():
            return
        if self._is_local_model_ready():
            return

        self._preload_started = True
        self._preload_failed = False
        self._preload_thread = threading.Thread(
            target=self._preload_local_model,
            name="model-preload",
            daemon=True,
        )
        self._preload_thread.start()

    def _preload_local_model(self):
        try:
            self._log("model preload start", perf=self._get_perf_snapshot())
            self._get_local_model()
            self._log("model preload done", perf=self._get_perf_snapshot())
            self.root.after(0, lambda: self._set_status("модель готова ✓"))
        except Exception as err:
            self._preload_failed = True
            self._log_exception("model preload", err)
            self.root.after(0, lambda e=err: self._set_status(f"model preload: {e}", True))

    def _update_backend_label(self, backend=None):
        text = "API"
        color = C["muted"]

        if backend == "api":
            text = "API"
            color = C["warn"]
        elif backend == "local":
            device, compute_type = self._current_runtime_device()
            text = f"{device.upper()} {compute_type}"
            color = C["success"] if device == "cuda" else C["accent"]
        else:
            chosen = self._choose_backend()
            if chosen == "local":
                device, compute_type = self._current_runtime_device()
                text = f"{device.upper()} {compute_type}"
                color = C["success"] if device == "cuda" else C["accent"]

        self.backend_lbl.configure(text=text, fg=color)

    def _on_close(self):
        self._log("overlay closing", perf=self._get_perf_snapshot())
        if self._stats_after:
            self.root.after_cancel(self._stats_after)
        self.recording = False
        save_cfg(self.cfg)
        self.root.destroy()

    def run(self):
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"320x460+{sw - 340}+30")
        self.root.mainloop()


if __name__ == "__main__":
    app = WhisperOverlay()
    app.run()
