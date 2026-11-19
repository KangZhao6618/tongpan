@echo off
setlocal
set "TONGPAN_ROOT=%~dp0"
set "TONGPAN_PY=%TONGPAN_ROOT%runtime\ocr"
if not exist "%TONGPAN_PY%\pythonw.exe" (
    echo Missing bundled runtime: "%TONGPAN_PY%\pythonw.exe"
    pause
    exit /b 1
)
set "PATH=%TONGPAN_PY%;%TONGPAN_PY%\DLLs;%TONGPAN_PY%\Library\bin;%SystemRoot%\System32;%SystemRoot%"
start "" "%TONGPAN_PY%\pythonw.exe" -I -S -B -X utf8 "%TONGPAN_ROOT%operations\dev_launcher.py" %*
