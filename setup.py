from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call
import os
import sys

if sys.version_info[0] < 3:
    with open('README.rst') as f:
        long_description = f.read()
    #
else:
    with open('README.rst', encoding='utf-8') as f:
        long_description = f.read()
    #
#

class PostDevelopCommand(develop):
    def run(self):
        check_call("sudo pip uninstall PyEOC".split())
        develop.run(self)
    #
#

class PostInstallCommand(install):
    def run(self):
        install.run(self)
    #
#

setup(
    name='PyEOC',
    version='1.0',
    description='Electro-Optic Coefficients Calculation (PyEOC)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Pr. Sidi Hamady',
    url='https://github.com/sidihamady/PyEOC',
    install_requires=['numpy','scipy','matplotlib'],
    download_url='https://github.com/sidihamady/PyEOC.git',
    py_modules=["eocCore", "tmmCore", "Test"],
    data_files=[
        ('.', ['poynting.png']),
        ('.', ['screenshot.png']),
        ('.', ['iconmain.png']),
        ('.', ['iconmain.gif']),
        ('.', ['SBN_Reflectivity_Dyn_TE.txt']),
        ('.', ['SBN_Reflectivity_Dyn_TM.txt']),
        ('.', ['SBN_Reflectivity_TE.txt']),
        ('.', ['SBN_Reflectivity_TM.txt']),
        ('.', ['AUTHORS']),
        ('.', ['COPYRIGHT']),
        ('.', ['LICENSE']),
        ('.', ['README.md']),
        ('.', ['LICENSE']),
    ],
    package_dir={'':'.'},
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
