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

# generic installation function, tests most common package managers
generic_install() {
    for iarg in "$@"
    do
        if [ $(dpkg-query -W -f='${Status}' $iarg 2>/dev/null | grep -c "ok installed") -eq 1 ]; then continue; fi
        echo "Attempting to install $iarg..."
        if [ -x "$(command -v apt)" ]; then sudo apt install "$iarg" && return 0
        elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install "$iarg" && return 0
        elif [ -x "$(command -v yum)" ]; then sudo yum install "$iarg" && return 0
        elif [ -x "$(command -v dnf)" ]; then sudo dnf install "$iarg" && return 0
        elif [ -x "$(command -v pacman)" ]; then sudo pacman -S "$iarg" && return 0
        elif [ -x "$(command -v zypper)" ]; then sudo zypper install "$iarg" && return 0
        else echo "PACKAGE INSTALLATION FAILED: Package manager not found. Please manually install packages.">&2 && return 1;
        fi
    done
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

# check if python3 is installed and prompt user to install
echo "Checking status of Python and related dependencies..."
if [ ! command -v python3 &>/dev/null ];
then
    while true; do
        read -p 'requirement python3 not installed. Install now (y/n)? ' yn
        case $yn in
            [Yy]* ) generic_install python3 python3-venv python3-tk && : || installationFailure; break;;
            [Nn]* ) exit;;
            * ) echo 'Please answer y or n.';;
        esac
    done
else
	generic_install python3-venv python3-tk && : || installationFailure
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

# check if R and required R packages work, if not install prereqs, R, r packages
if ! Rscript "$DIR/scripts/check_r_reqs.R" &>/dev/null
then
    echo "R or required packages not installed. This installation may take some time..."
    sleep 0.5
    generic_install libcurl4-openssl-dev libssl-dev libxml2-dev dirmngr gnupg apt-transport-https ca-certificates software-properties-common libcurl4-gnutls-dev libudunits2-dev libgdal-dev gdal-bin libproj-dev proj-data proj-bin libgeos-dev && : || rInstallationFailure
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9 && : || rInstallationFailure
    sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/' && : || rInstallationFailure
    sudo apt install r-base-dev && : || rInstallationFailure
    sudo rm -Rf /usr/local/lib/R/site-library && : || rInstallationFailure
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
