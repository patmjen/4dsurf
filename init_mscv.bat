@echo off

where /q vswhere.exe
if ERRORLEVEL 1 (
    echo INFO: vswhere.exe is not in PATH -- assuming it is in "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer"
    set VSWHERE="%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
) else (
    set VSWHERE=vswhere.exe
)

echo INFO: Finding Visual Studio installation path
for /f "usebackq tokens=*" %%i in (`%VSWHERE% -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VS_INSTALL_DIR=%%i
)
echo INFO: Found Visual Studio in %VS_INSTALL_DIR%

echo INFO: Finding vcvarsall.bat
if exist "%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat" (
    set VCVARSALL="%VS_INSTALL_DIR%\VC\Auxiliary\Build\vcvarsall.bat"
) else (
    echo ERROR: Could not find vcvarsall.bat in Visual Studio installation path
    exit /b 1
)
echo INFO: Found vcvarsall.bat as %VCVARSALL%

echo INFO: Set Visual Studio Toolset to 64-bit
call %VCVARSALL% x64
