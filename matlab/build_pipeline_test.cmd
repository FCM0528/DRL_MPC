@echo off
setlocal
cd /d "%~dp0"
call "D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cd build
cl /nologo /std:c++17 /EHsc /MD /O2 /utf-8 /DNOMINMAX /DPARITY_PIPELINE /I"..\controllers\native" ..\tests\parity_original.cpp ..\controllers\native\mpc_pipeline.cpp ..\controllers\native\mpc.cpp ..\controllers\native\vehicle.cpp ..\controllers\native\reference.cpp ..\controllers\native\rl.cpp ..\controllers\native\config.cpp ..\controllers\native\carsim_plant.cpp /Fe:parity_pipeline.exe
exit /b %errorlevel%
