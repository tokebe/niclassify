import subprocess
from os import path
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install

# install bPTP - cwd changes directory for install
# (bPTP install must happen from inside its directory)


class InstallbPTP(install):
    def run(self):
        install.run(self)
        python_path = sys.executable
        subprocess.run(
            '"{}" "{}" install'.format(
                python_path,
                path.join(path.dirname(path.abspath(__file__)),
                          "bin/PTP-master/setup.py")
            ),
            cwd=path.join(path.dirname(
                path.abspath((__file__))), "bin/PTP-master"),
            shell=True
        )


setup(
    name='niclassify',  # TODO rename this project at some point please
    version='0.2.0a1',
    description='Automated insect data retrieval and classification tool',
    author='Jackson Callaghan',
    author_email='jcs.callaghan@gmail.com',
    url='https://github.com/jackson-callaghan',
    packages=find_packages(include=['niclassify']),
    install_requires=[
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
        "seaborn",
        "xlrd",
        "pandas",
        "joblib",
        "matplotlib",
        "numpy",
        "scipy",
    ],
    cmdclass={'install': InstallbPTP},
    # entry_points={
    #     'console_scripts': [
    #         'niclassify-gui = gui:main'
    #     ]
    # }
    # TODO implement entry_points
)
