@echo off
:: Check if we're using 'python'  or 'py'
:beginScript
py -3 --version >nul 2>nul && set "pyLaunch=py" || set "pyLaunch=python"
%pyLaunch% --version >nul 2>nul && goto:checkVersion || goto:errorNoPython

:: check python version
:checkVersion
if "%pyLaunch%" == "py" (
    echo Checking version using py launcher...
    %pyLaunch% -3 "%~dp0\scripts\check_version.py" >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
) else (
    echo Checking version using python PATH variable...
    %pyLaunch% "%~dp0\scripts\check_version.py" >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
)

:: Correct python version is installed, we may proceed
:pythonInstalled
echo Checking virtual environment...
if exist "%~dp0\niclassifyenv\Scripts\activate.bat" (
    "%~dp0\niclassifyenv\Scripts\python.exe" "%~dp0\scripts\testvenv.py" >nul 2>nul && goto:launchProgram || goto:corruptVenv

) else (
    wscript "%~dp0\scripts\message.vbs" "Executing First time setup. Please do not close the Console Window."
    echo Executing First time setup. Please do not close the Console Window...
    goto:makeVenv
)
goto:eof

:: in the event that the venv appears to be corrupt
:corruptVenv
echo Virtual environment appears corrupted, attempting to fix...
goto:makeVenv

:: venv isn't set up, we must run initial setup
:makeVenv
echo Setting up virtual environment...
if exist "%~dp0\niclassifyenv" rmdir /S /q "%~dp0\niclassifyenv"
if "%pyLaunch%" == "py" (
    %pyLaunch% -3 -m venv "%~dp0\niclassifyenv"
) else (
    %pyLaunch% -m venv "%~dp0\niclassifyenv"
)
call "%~dp0\niclassifyenv\Scripts\python.exe" -m pip install --upgrade pip
call "%~dp0\niclassifyenv\Scripts\python.exe" "%~dp0\setup.py" install
goto:launchProgram

:: we can launch the program
:launchProgram
echo Starting Program...
start "" "%~dp0\niclassifyenv\Scripts\pythonw.exe" "%~dp0\niclassify\niclassify.pyw"
goto:eof

:: notify user python is not installed
:errorNoPython
wscript "%~dp0\scripts\message.vbs" "Python is not installed. Please install Python 3.7.x - 3.8.x and then click 'ok'."
echo Python is not installed. Please install Python 3.7.x - 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:beginScript
goto:eof

:: notify user wrong version of python is installed
:errorWrongPython
wscript "%~dp0\scripts\message.vbs" "The wrong version of Python is installed. Please ensure you have a version of Python 3 between 3.7.x and 3.8.x and try again."
echo The wrong version of Python is installed. Please ensure you have a version of Python between 3.7.x and 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:eof
