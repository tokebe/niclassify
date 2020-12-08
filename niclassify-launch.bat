::check if python is installed
>nul 2>nul python --version goto:pythonInstalled || goto:errorNoPython

:pythonInstalled
if exist "niclassifyenv\scripts\activate.bat" (
    goto:launchProgram
) else (
    goto:makeVenv
)
goto:eof

:makeVenv
wscript "scripts\message.vbs" "Executing First time setup. Please do not close the Console Window."
call "scripts\easy-install.bat"
goto launchProgram

:launchProgram
start "" "niclassifyenv\Scripts\pythonw.exe" "niclassify\niclassify.pyw"
goto:eof

:errorNoPython
wscript "scripts\message.vbs" "Python is not installed. Please install Python 3.7.x - 3.8.x https://www.python.org/downloads/release/python-379/"
pause
