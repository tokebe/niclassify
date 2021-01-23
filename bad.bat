@echo off
setlocal
:: call scripts\fchooser.bat
set /p rloc=< "%~dp0\niclassify\core\utilities\config\rloc.txt"
echo %rloc%
echo "%rloc%\bin\Rscript.exe"
::"%rloc%\bin\Rscript.exe" "scripts\check_r_reqs.R"
endlocal
