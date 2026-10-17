@echo off
rem Double-click to open catchtool in your browser. The first run sets itself up (a few minutes).
cd /d "%~dp0"
set "CONDA_BAT="
for %%P in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
  if exist "%%~P\condabin\conda.bat" if not defined CONDA_BAT set "CONDA_BAT=%%~P\condabin\conda.bat"
)
if not defined CONDA_BAT (
  where conda >nul 2>nul && set "CONDA_BAT=conda"
)
if not defined CONDA_BAT (
  echo Conda was not found. Install Miniconda from https://docs.conda.io/en/latest/miniconda.html and run this again.
  pause
  exit /b 1
)
call "%CONDA_BAT%" env list | findstr /b /c:"catchtool " >nul
if errorlevel 1 (
  echo First run: installing catchtool and its libraries. Please wait...
  call "%CONDA_BAT%" create -y -n catchtool python=3.12 -c conda-forge rasterio geopandas numba matplotlib
  if errorlevel 1 ( echo Setup failed. & pause & exit /b 1 )
)
call "%CONDA_BAT%" activate catchtool
pip install -q -e .
catchtool-gui
pause
