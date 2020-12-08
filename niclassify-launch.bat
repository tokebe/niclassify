@echo off
:: Check if we're using 'python'  or 'py'
:beginScript
py -3 --version >nul 2>nul && set "pyLaunch=py" || set "pyLaunch=python"
%pyLaunch% --version >nul 2>nul && goto:checkVersion || goto:errorNoPython

:: check python version
:checkVersion
if "%pyLaunch%" == "py" (
    echo Checking version using py launcher...
    %pyLaunch% -3 scripts\check_version.py >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
) else (
    echo Checking version using python PATH variable...
    %pyLaunch% scripts\check_version.py >nul 2>nul && goto:pythonInstalled || goto:errorWrongPython
)

:: Correct python version is installed, we may proceed
:pythonInstalled
if exist "niclassifyenv\scripts\activate.bat" (
    goto:launchProgram
) else (
    goto:makeVenv
)
goto:eof

:: venv isn't set up, we must run initial setup
:makeVenv
wscript "scripts\message.vbs" "Executing First time setup. Please do not close the Console Window."
echo Executing First time setup. Please do not close the Console Window...
if "%pyLaunch%" == "py" (
    %pyLaunch% -3 -m venv niclassifyenv
) else (
    %pyLaunch% -m venv niclassifyenv
)
call niclassifyenv\Scripts\python.exe -m pip install --upgrade pip
call niclassifyenv\Scripts\python.exe setup.py install
goto:launchProgram

:: we can launch the program
:launchProgram
echo Starting Program...
start "" "niclassifyenv\Scripts\pythonw.exe" "niclassify\niclassify.pyw"
goto:eof

:: notify user python is not installed
:errorNoPython
wscript "scripts\message.vbs" "Python is not installed. Please install Python 3.7.x - 3.8.x and then click 'ok'."
echo Python is not installed. Please install Python 3.7.x - 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:beginScript
goto:eof

:: notify user wrong version of python is installed
:errorWrongPython
wscript "scripts\message.vbs" "The wrong version of Python is installed. Please ensure you have a version of Python 3 between 3.7.x and 3.8.x and try again."
echo The wrong version of Python is installed. Please ensure you have a version of Python between 3.7.x and 3.8.x. https://www.python.org/downloads/release/python-386/
start "" https://www.python.org/downloads/release/python-386/
goto:eof
