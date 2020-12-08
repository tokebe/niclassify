import subprocess
from os import path
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools import Command

# install bPTP - cwd changes directory for install
# (bPTP install must happen from inside its directory)


# class BuildEXE(Command):
#     """Build an executable version"""
#     description = "build GUI executable"

#     user_options = [
#     ]

#     def initialize_options(self):
#         None

#     def finalize_options(self):
#         None

#     def run(self):
#         subprocess.run(
#             ".\\scripts\\make_exe.bat",
#             shell=True
#         )

# class MakeInnoInstaller(Command):
#     """Build an executable version"""
#     description = "build GUI executable"

#     user_options = [
#     ]

#     def initialize_options(self):
#         None

#     def finalize_options(self):
#         None

#     def run(self):
#         subprocess.run(
#             '"%programfiles(x86)%\\Inno Setup 6\\ISCC.exe" .\\scripts\\installer.iss',
#             shell=True
#         )


setup(
    name='niclassify',  # TODO rename this project at some point please
    version='0.2.0a1',
    description='Automated insect data retrieval and classification tool',
    author='Jackson Callaghan',
    author_email='jcs.callaghan@gmail.com',
    url='https://github.com/jackson-callaghan',
    packages=find_packages(include=['niclassify']),
    install_requires=[
        "joblib==0.11",
        "numpy==1.19.3",
        "biopython",
        "docutils",
        "ete3",
        "flake8",
        "lxml",
        "pyinquirer",
        "pyqt5",
        "pytest",
        "python-nexus",
        "requests",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xlrd",
        "pandas",
        "scipy",
        "userpaths"
    ],
    # cmdclass={
    #     'install': InstallbPTP,
    # },
    # entry_points={
    #     'console_scripts': [
    #         'niclassify-gui = gui:main'
    #     ]
    # }
    # TODO implement entry_points
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
