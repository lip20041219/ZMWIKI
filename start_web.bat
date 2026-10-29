@echo off
setlocal
rem One-click Windows launcher for the ZMWiki browser workbench.
rem Requires Python 3.10+ on PATH. The web server is standard-library only,
rem so running from the source tree needs no install step.

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is required.
  echo Install it from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Python 3.10 or newer is required; found an older interpreter.
  echo Install it from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
set "WORKSPACE=%CD%\my_wiki.json"

if exist "%WORKSPACE%" (
  echo Using workspace: %WORKSPACE%
  python -m zmwiki.web.server --workspace "%WORKSPACE%" --port 8000
) else (
  echo Creating workspace: %WORKSPACE%
  python -m zmwiki.web.server --workspace "%WORKSPACE%" --port 8000 --create
)

if errorlevel 1 pause
endlocal
