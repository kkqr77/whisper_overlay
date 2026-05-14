param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $root $VenvPath
$python = Join-Path $venvDir "Scripts\python.exe"
$pip = Join-Path $venvDir "Scripts\pip.exe"
$torchIndexUrl = "https://download.pytorch.org/whl/cu128"
$torchVersion = "2.11.0+cu128"

function Invoke-Checked {
    param(
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $python)) {
    Write-Host "Creating virtual environment at $venvDir..."
    Invoke-Checked { python -m venv $venvDir }
}

Write-Host "Upgrading pip in $venvDir..."
Invoke-Checked { & $python -m pip install --upgrade pip }

Write-Host "Installing overlay requirements..."
Invoke-Checked { & $pip install -r (Join-Path $root "requirements.txt") }

$installedTorchVersion = ""
try {
    $installedTorchVersion = (& $python -c "import importlib.util as u; import sys; spec=u.find_spec('torch'); print(__import__('torch').__version__ if spec else '')").Trim()
} catch {
    $installedTorchVersion = ""
}

if ($installedTorchVersion -ne $torchVersion) {
    Write-Host "Installing CUDA-enabled torch runtime..."
    Invoke-Checked { & $pip install --index-url $torchIndexUrl "torch==$torchVersion" }
} else {
    Write-Host "CUDA-enabled torch runtime already installed: $installedTorchVersion"
}

Write-Host "Probing Whisper runtime..."
$probeScript = @'
import ctypes
import json
import os
import sys
from pathlib import Path

site_packages = Path(sys.prefix) / "Lib" / "site-packages"
for dll_dir in (
    site_packages / "torch" / "lib",
    site_packages / "ctranslate2",
):
    if os.name == "nt" and dll_dir.is_dir():
        os.add_dll_directory(str(dll_dir))

import ctranslate2

loader = ctypes.WinDLL if os.name == "nt" else ctypes.CDLL
details = {}
groups = [
    ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"],
    ["cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops_infer64_8.dll", "cudnn_cnn_infer64_8.dll"],
]

ready_group = None
for group in groups:
    group_ok = True
    for lib in group:
        try:
            loader(lib)
            details[lib] = "ok"
        except OSError as err:
            details[lib] = str(err)
            group_ok = False
    if group_ok:
        ready_group = group
        break

print(json.dumps({
    "python": sys.executable,
    "ctranslate2": ctranslate2.__version__,
    "cuda_devices": ctranslate2.get_cuda_device_count(),
    "cudnn_ready": ready_group is not None,
    "ready_group": ready_group,
    "cudnn": details,
}, ensure_ascii=False, indent=2))
'@

$probeScript | & $python -
if ($LASTEXITCODE -ne 0) {
    throw "Whisper runtime probe failed with exit code $LASTEXITCODE"
}

Write-Host "Environment setup complete."
