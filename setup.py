from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class InstallMorfeusz(install):
    def run(self):
        # Run the standard install process
        install.run(self)
        # Run the custom installation command
        subprocess.check_call(['python', '-m', 'pip', 'install', 'pl_spacy_model_morfeusz_big-0.1.0.tar.gz'])

setup(
    name='your_package_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "tensorflow==2.2.0",
        "spacy=2.2.4",
        "keras==2.3.1",
        "protobuf==3.19",
        "fire"
    ],
    cmdclass={
        'install': InstallMorfeusz,
    },
)