@echo off
:: Check if we're using 'python'  or 'py'
:: If neither work, Python's not installed, ask the user to install
:beginScript
py -3 --version >nul 2>nul && set "pyLaunch=py" || set "pyLaunch=python"
%pyLaunch% --version >nul 2>nul && goto:checkVersion || goto:errorNoPython

:: Check python version
:: Scripts\check_version.py exits -1 if the version isn't acceptable, or 0 if it is.
:: If the version's wrong, inform the user
:checkVersion
if "%pyLaunch%" == "py" (
    echo Checking version using py launcher...
    %pyLaunch% -3 "%~dp0\scripts\check_version.py" >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
) else (
    echo Checking version using python PATH variable...
    %pyLaunch% "%~dp0\scripts\check_version.py" >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
)

:: Correct python version is installed, we may proceed
:: Check if the venv is already installed and has integrity, setup if not
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

:: In the event that the venv appears to be corrupt, inform the user and re-install venv
:corruptVenv
wscript "%~dp0\scripts\message.vbs" "The virtual environment appears to be corrupt. The program will now attempt to fix it."
echo Virtual environment appears corrupted, attempting to fix...
goto:makeVenv

:: Delete previous venv (if it exists) and setup new venv
:: Allows setup output so the user knows that it's working on it
:makeVenv
echo Setting up virtual environment...
if exist "%~dp0\niclassifyenv" rmdir /S /q "%~dp0\niclassifyenv"
if "%pyLaunch%" == "py" (
    %pyLaunch% -3 -m venv "%~dp0\niclassifyenv"
) else (
    %pyLaunch% -m venv "%~dp0\niclassifyenv"
)
:: make sure pip is upgraded to avoid problems with certain packages
call "%~dp0\niclassifyenv\Scripts\python.exe" -m pip install --upgrade pip
call "%~dp0\niclassifyenv\Scripts\python.exe" "%~dp0\setup.py" install
goto:launchProgram

:: Launch the program and exit the console window
:launchProgram
echo Starting Program...
start "" "%~dp0\niclassifyenv\Scripts\pythonw.exe" "%~dp0\niclassify\niclassify.pyw"
goto:eof

:: Notify user python is not installed and ask them to install it
:: Open browser to the preferred version and restart script after they install
:errorNoPython
wscript "%~dp0\scripts\message.vbs" "Python is not installed. Please install Python 3.7.x - 3.8.x and then click 'ok'."
echo Python is not installed. Please install Python 3.7.x - 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:beginScript
goto:eof

:: Notify the user that the wrong version of python is installed and exit.
:: Assumes that if user has python installed, but wrong version, they would prefer to fix it on their own
:errorWrongPython
wscript "%~dp0\scripts\message.vbs" "The wrong version of Python is installed. Please ensure you have a version of Python 3 between 3.7.x and 3.8.x and try again."
echo The wrong version of Python is installed. Please ensure you have a version of Python between 3.7.x and 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:eof
