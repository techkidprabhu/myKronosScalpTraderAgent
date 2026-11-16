@echo off
rem Fine-tune Kronos on this computer's NVIDIA GPU. Sets Python up the first
rem time (in %USERPROFILE%\.kronos-trainer, shared by every kit you export),
rem then trains. Double-click it, or from a command prompt:
rem    train.bat            train.bat --steps 6000 --batch 16
setlocal
cd /d "%~dp0"
if "%KRONOS_TRAINER_HOME%"=="" set "KRONOS_TRAINER_HOME=%USERPROFILE%\.kronos-trainer"
if "%TORCH_INDEX%"=="" set "TORCH_INDEX=https://download.pytorch.org/whl/cu130"
set "VENV=%KRONOS_TRAINER_HOME%\venv"
if exist "%VENV%\Scripts\python.exe" goto run

where py >nul 2>nul && (set "PYLAUNCH=py -3") || (set "PYLAUNCH=python")
%PYLAUNCH% --version >nul 2>nul
if errorlevel 1 (
  echo Python is not installed. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
  echo ^(tick "Add python.exe to PATH" in the installer^), then start train.bat again.
  goto end
)
echo First run: setting up Python packages in %VENV% ^(a few GB, several minutes^)...
if not exist "%KRONOS_TRAINER_HOME%" mkdir "%KRONOS_TRAINER_HOME%"
%PYLAUNCH% -m venv "%VENV%" || goto failed
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV%\Scripts\python.exe" -m pip install torch --index-url %TORCH_INDEX% || goto failed

:run
"%VENV%\Scripts\python.exe" -m pip install -q -r requirements.txt || goto failed
"%VENV%\Scripts\python.exe" train_kronos.py %*
goto end

:failed
echo.
echo Setup did not finish. Check the internet connection and start train.bat again.
echo If it keeps failing, delete %VENV% and try once more.

:end
echo.
pause
endlocal
