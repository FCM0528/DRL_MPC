@echo off
setlocal
set "MPC_MEX_NAME=mpc_mex"
if not "%~1"=="" set "MPC_MEX_NAME=%~1"
cd /d "%~dp0"
if not exist build mkdir build
call "D:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44
if errorlevel 1 exit /b 1
cd build
cl /nologo /std:c++17 /EHsc /MD /O2 /utf-8 /LD /DMATLAB_MEX_FILE /DMW_NEEDS_VERSION_H /DNOMINMAX /I"D:\Program Files\MATLAB\R2025a\extern\include" /I"..\controllers\native" ..\controllers\native\mpc_mex.cpp ..\controllers\native\mpc_pipeline.cpp ..\controllers\native\mpc.cpp ..\controllers\native\vehicle.cpp ..\controllers\native\reference.cpp ..\controllers\native\rl.cpp ..\controllers\native\config.cpp ..\controllers\native\carsim_plant.cpp /link /OUT:%MPC_MEX_NAME%.mexw64 /LIBPATH:"D:\Program Files\MATLAB\R2025a\extern\lib\win64\microsoft" libmex.lib libmx.lib libmat.lib
exit /b %errorlevel%


