# NIClassify

## What is NIClassify?

NIClassify is a combined toolkit for classifying species (usually, as in the name, as Native or Introduced), based on the principles laid out in [Categorization of species as likely native or likely non-native using DNA barcodes without a complete reference library.](https://doi.org/10.1002/eap.1914) (Andersen JC, et al.)

The main point of this project has been to provide a straightforward GUI to use this toolkit.

(For a more in-depth explanation, see [The User's Manual](docs/user-manual.md))

## Installation

1. Install [Python](https://www.python.org/downloads/release/python-386/) 3.7.x - 3.8.x (3.9 is not yet supported).
2. Install [the R base](https://cloud.r-project.org/) (Any version _should_ work).
3. Download the latest [Release](https://github.com/tokebe/niclassify/releases) (or clone/dowload the repository, if you want to see the newest ~~bugs~~ code).
4. Run the appropriate `launch-___.bat` for your system and follow any messages you may get.

## Usage

Run the appropriate `launch-___.bat` for your system after installation.

## User Manual

This project comes with a user manual, which you may access [here](docs/user-manual.md) or by clicking 'help' in the main window of the program.

## Advanced Usage/Installation

You can find a list of package requirements in the `requirements.txt` file, and may run the program by running `main.py` instead of relying on a launcher script, if you wish. Please see the warning below.

## WARNING

This project has been structured as a Python package for ease of development, however it will not fully function if installed as a package using `setup.py install`. `niclassify/core/bin` is not currently copied during setuptools installation as it contains many, many files and would take an unacceptably long time to run. As such, the program must be run from within the project folder.

This may be resolved in the future, however for the time being the project relies on R Portable for a number of functions, so it will take time to fix.

## Further Development

Most submodules and functions of NIClassify are documented with comments and docstrings. A technical manual may be added to the docs in a future update, which will detail the inner workings of the project further.

Please, by all means, register issues for any problems you encounter, make pull requests for anything you want to try implementing, etc, etc. Until a message is left below stating otherwise, this project is being actively maintained.
