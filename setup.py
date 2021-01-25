import subprocess
import os
from os import path
import sys
from setuptools import find_packages, setup

setup(
    name='niclassify',
    version='0.3.0a',
    description='Automated insect data retrieval and classification tool',
    author='Jackson Callaghan',
    author_email='jcs.callaghan@gmail.com',
    url='https://github.com/jackson-callaghan',
    packages=find_packages(),
    package_data={
        "": ["*.json", "*.ico", "*.jpg", "*.pdf", "*.R"],
    },
    setup_requires=[
        "numpy==1.19.3",
    ],
    install_requires=[
        "atomicwrites==1.4.0",
        "attrs==20.3.0",
        "biopython==1.78",
        "certifi==2020.12.5",
        "chardet==3.0.4",
        "clldutils==3.6.0",
        "colorama==0.4.4",
        "colorlog==4.6.2",
        "csvw==1.8.1",
        "cycler==0.10.0",
        "docutils==0.16",
        "ete3==3.1.2",
        "flake8==3.8.4",
        "idna==2.10",
        "iniconfig==1.1.1",
        "isodate==0.6.0",
        "joblib==0.11",
        "kiwisolver==1.3.1",
        "lxml==4.6.2",
        "matplotlib==3.3.3",
        "mccabe==0.6.1",
        "newick==1.0.0",
        "numpy==1.19.3",
        "packaging==20.7",
        "pandas==1.2.0rc0",
        "pillow==8.0.1",
        "pluggy==0.13.1",
        "prompt-toolkit==1.0.14",
        "py==1.9.0",
        "pycodestyle==2.6.0",
        "pyflakes==2.2.0",
        "pygments==2.7.3",
        "pyinquirer==1.0.3",
        "pyparsing==3.0.0b1",
        "pyqt5==5.15.2",
        "pyqt5-sip==12.8.1",
        "pytest==6.1.2",
        "python-dateutil==2.8.1",
        "python-nexus==2.0.1",
        "pytz==2020.4",
        "regex==2020.11.13",
        "requests==2.25.0",
        "rfc3986==1.4.0",
        "scikit-learn==0.24.0rc1",
        "scipy==1.5.4",
        "seaborn==0.11.0",
        "six==1.15.0",
        "tabulate==0.8.7",
        "termcolor==1.1.0",
        "threadpoolctl==2.1.0",
        "toml==0.10.2",
        "uritemplate==3.0.1",
        "urllib3==1.26.2",
        "userpaths==0.1.3",
        "wcwidth==0.2.5",
        "xlrd==1.2.0",
    ],
    # cmdclass={
    #     'install': InstallbPTP,
    # },
    entry_points={
        'console_scripts': [
            'niclassify = niclassify.__main__:main'
        ]
    }
)

if sys.argv[1] == "install":
    python_path = sys.executable
    subprocess.run(
        'cd "{}" && "{}" setup.py install'.format(
            path.join(path.dirname(path.abspath(__file__)),
                      "niclassify/bin/PTP-master"),
            python_path,
        ),
        cwd=path.join(path.dirname(
            path.abspath((__file__))), "niclassify/bin/PTP-master"),
        shell=True
    )
