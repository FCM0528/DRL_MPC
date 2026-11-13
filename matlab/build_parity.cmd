@echo off
setlocal
cd /d "%~dp0"
call "D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cd build
cl /nologo /std:c++17 /EHsc /MD /O2 /utf-8 /DNOMINMAX /I"..\..\src" ..\tests\parity_original.cpp ..\..\src\mpc.cpp ..\..\src\vehicle.cpp ..\..\src\reference.cpp ..\..\src\rl.cpp ..\..\src\config.cpp ..\..\src\carsim_plant.cpp /Fe:parity_original.exe
exit /b %errorlevel%
