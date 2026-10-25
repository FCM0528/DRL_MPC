@echo off
setlocal
cd /d "%~dp0"
echo [1/3] Configure Visual Studio 2022 x64...
cmake -S . -B build_vs2022 -G "Visual Studio 17 2022" -A x64
if errorlevel 1 goto :error
echo [2/3] Build Release...
cmake --build build_vs2022 --config Release
if errorlevel 1 goto :error
echo [3/3] Run using configs\default.ini...
"build_vs2022\Release\MPC_DoubleLaneChange.exe" --config "configs\default.ini"
if errorlevel 1 goto :error
echo Done. See output path printed above.
pause
exit /b 0
:error
echo Build or run failed. Ensure Visual Studio C++ and CMake are installed.
pause
exit /b 1
