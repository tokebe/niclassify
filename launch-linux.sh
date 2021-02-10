#!/bin/bash

# generic installation function, tests most common package managers
generic_install () {
    echo "Attempting to install $1..."
    if [ -x "$(command -v apt)" ]; then sudo apt install $1
    elif [ -x "$(command -v apt-get)" ]; then sudo apt-get install $1
    elif [ -x "$(command -v yum)" ]; then sudo yum install $1
    elif [ -x "$(command -v dnf)" ]; then sudo dnf install $1
    elif [ -x "$(command -v pacman)" ]; then sudo pacman -S $1
    elif [ -x "$(command -v zypper)" ]; then sudo zypper install $1
    else echo "PACKAGE INSTALLATION FAILED: Package manager not found. Please manually install $1.">&2;
    fi
}

# check if python3 is installed and prompt user to install
if [ ! -x "$(command -v python3 &> /dev/null)" ];
then
    while true; do
        read -p 'requirement python3 not installed. Install now (y/n)? ' yn
        case $yn in
            [Yy]* ) generic_install python3; break;;
            [Nn]* ) exit;;
            * ) echo 'Please answer y or n.';;
        esac
    done
fi

# check if r is installed and prompt user to install
if [ ! -x "$(command -v Rscript)" ]
# something like this:
# sudo apt install install libcurl4-openssl-dev libssl-dev libxml2-dev dirmngr gnupg apt-transport-https ca-certificates software-properties-common libcurl4-gnutls-dev libudunits2-dev libgdal-dev gdal-bin libproj-dev proj-data proj-bin libgeos-dev
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
# sudo apt install r-base-dev
# sudo rm -Rf /usr/local/lib/R/site-library
# sudo Rscript scripts/install_r_reqs.R
