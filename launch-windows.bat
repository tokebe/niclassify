@echo off
if "%1"=="am_admin" (goto:RPackageInstall)
cd %~dp0
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
    "%~dp0\niclassifyenv\Scripts\python.exe" "%~dp0\scripts\testvenv.py" >nul 2>nul && goto:RInstalled || goto:corruptVenv
) else (
    wscript "%~dp0\scripts\message.vbs" "Executing First time setup. Please do not close the Console Window."
    echo Executing First time setup. Please do not close the Console Window...
    goto:makeVenv
)
goto:eof

:: Prompt user to select R location and check if Rscript.exe exists
:GetRLoc
wscript "%~dp0\scripts\message.vbs" "R could not be located. After ensuring it is installed, click OK. A folder prompt will open. Please locate the R folder with the latest version (e.g. 'R-4.0.3')."
call "%~dp0\scripts\fchooser.bat"
goto:RInstalled

:: Check if the defined location for R exists/if there is a defined location
:: Prompt user to install and locate R if not (current 3.6.3)
:RInstalled
echo Checking status of R installation...
if exist "%~dp0\niclassify\core\utilities\config\rloc.txt" (
    :: read in saved folder path
    :: need usebackq and delims= to allow spaces in target file path, and not delim on spaces, respectively
    :: (windows batch and its documentation makes me sad)
    for /f "usebackq delims=" %%x in ("%~dp0\niclassify\core\utilities\config\rloc.txt") do set RFolderLocation=%%x&goto:next1
    :next1
    if exist "%RFolderLocation%\bin\Rscript.exe" (
        goto:RPackageCheck
    ) else (
        start "" https://cloud.r-project.org/
        goto:GetRLoc
    )
) else (
    goto:GetRLoc
)

:: Check if required packages are installed
:RPackageCheck
echo Checking required R packages...
"%RFolderLocation%\bin\Rscript.exe" "%~dp0\scripts\check_r_reqs.R" >nul 2>nul && goto:launchProgram || goto:RPackageInstall

:RPackageInstall
echo Installing required R packages. This may take some time...
if "%RFolderLocation%"=="" (
    for /f "usebackq delims=" %%x in ("%~dp0\niclassify\core\utilities\config\rloc.txt") do set RFolderLocation=%%x&goto:next2
    )
:next2
if not "%1"=="am_admin" (powershell start -verb runas '%0' am_admin & exit /b)
"%RFolderLocation%\bin\Rscript.exe" "%~dp0\scripts\install_r_reqs.R" && goto:launchProgram || goto:RPackageFailure

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

goto:RInstalled

:: Launch the program and exit the console window
:launchProgram
echo Starting Program...
start "" "%~dp0\niclassifyenv\Scripts\pythonw.exe" "%~dp0\main.py"
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

:: Notify user that something went wrong with R package installation and give up
:RPackageFailure
wscript "%~dp0\scripts\message.vbs" "Something went wrong with R package installation. Please review the console output for debug purposes. You may have to complete package installation manually. If this problem persists, please register an issue or contact the project developer."
