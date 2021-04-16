#!/bin/bash

# get script dir for file reference
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

installationFailure() {
 echo "Something went wrong installing packages, please check the output for debug."
 exit 1
}

rInstallationFailure() {
 echo "Something went wrong installing R or its packages, please check the output for debug."
 exit 1
}

venvSetupFailure() {
    echo "Something went wrong with Python venv setup. Please review the console output for debug purposes, and copy any error messages to paste when registering an issue on the repositoy page."
    exit 1
}

setup_venv() {
    if [ -e "$DIR/niclassifyenv" ];
    then
        sudo rm -rf "$DIR/niclassifyenv"
    fi
    python3 -m venv "$DIR/niclassifyenv"
    "$DIR/niclassifyenv/bin/python3" -m pip install --upgrade pip && : || venvSetupFailure
    "$DIR/niclassifyenv/bin/python3" "$DIR/setup.py" install && : || venvSetupFailure
}

# check if xcode is installed
if ! xcode-select -p 1>/dev/null 2>&1
then
    osascript -e 'display alert "XCode Installation" message "XCode will now be installed. Please confirm any prompts that show up, and start the script again when complete."'
    xcode-select --install
fi

# check again if xcode is installed
if ! xcode-select -p 1>/dev/null 2>&1
then
    installationFailure
fi

# check if python3 is installed and prompt user to install
echo "Checking status of Python and related dependencies..."
if [ ! -e "$DIR/scripts/.python3installed" ];
then
    osascript -e 'display alert "Python 3 installation" message "Before clicking ok, please install Python 3.9.x from the official Python website. If you have already done so, you may safetly ignore this notification." '
    touch "$DIR/scripts/.python3installed"
fi

# check if python version is acceptable and inform user if not
if ! python3 "$DIR/scripts/check_version.py"
then
    echo "The Python version is not supported. Please install Python 3.7.x - 3.9.x"
    exit 1
fi

# check if the venv is already installed and has integrity, and install if not
echo "Checking Virtual Environment..."
if [ ! -f "$DIR/niclassifyenv/bin/python3" ];
then
    echo "Executing First time setup. Please do not interrupt the program."
    setup_venv
fi

if ! "$DIR/niclassifyenv/bin/python3" "$DIR/scripts/testvenv.py"
then
    echo "Virtual environment appears corrupted, attempting to fix..."
    setup_venv
fi

# check if R is installed
if ! R --version  &>/dev/null
then
    open -n "https://cloud.r-project.org/"
    osascript -e 'display alert "R Not Installed" message "R is not installed. Please install R and then click ok"'
fi

# check if R and required R packages work, if not install prereqs, R, r packages
if ! Rscript "$DIR/scripts/check_r_reqs.R" &>/dev/null
then
    echo "required R packages not installed. This installation may take some time..."
    sleep 0.5
    sudo Rscript scripts/install_r_reqs.R && : || rInstallationFailure
    if ! Rscript "$DIR/scripts/check_r_reqs.R"
    then
        echo "Something has gone wrong with automated R/package installation. Please see above output for debug purposes."
        exit 1
    else
        echo "Installation success."
    fi
fi

echo "Starting Program..."
echo "=====[ V NICLASSIFY DEBUG OUTPUT BELOW V ]====="
"$DIR/niclassifyenv/bin/python3" "$DIR/main.py"
exit 0
