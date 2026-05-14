@echo off
setlocal
set ROOT=%~dp0
set VENV=%ROOT%.venv
set BASE_PYTHON_HOME=

if not exist "%VENV%\Scripts\pythonw.exe" (
  echo Virtual environment not found. Run setup_overlay_env.ps1 first.
  exit /b 1
)

for /f "tokens=1,* delims==" %%A in ('findstr /b /c:"home = " "%VENV%\pyvenv.cfg"') do (
  set BASE_PYTHON_HOME=%%B
)

if defined BASE_PYTHON_HOME (
  set BASE_PYTHON_HOME=%BASE_PYTHON_HOME:~1%
  set TCL_LIBRARY=%BASE_PYTHON_HOME%\tcl\tcl8.6
  set TK_LIBRARY=%BASE_PYTHON_HOME%\tcl\tk8.6
)

if exist "%VENV%\Lib\site-packages\torch\lib" set PATH=%VENV%\Lib\site-packages\torch\lib;%PATH%
if exist "%VENV%\Lib\site-packages\ctranslate2" set PATH=%VENV%\Lib\site-packages\ctranslate2;%PATH%

if not exist "%TCL_LIBRARY%\init.tcl" set TCL_LIBRARY=
if not exist "%TK_LIBRARY%\tk.tcl" set TK_LIBRARY=

start "" "%VENV%\Scripts\pythonw.exe" "%ROOT%whisper_overlay.py"
