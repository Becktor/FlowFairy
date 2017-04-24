
from setuptools import setup, find_packages

exclude_pkgs = ['flowfairy.bin']

setup(
    name="flowfairy",
    version='0.0.1.dev1',
    license='...',
    packages=find_packages(exclude=exclude_pkgs),
    entry_points = {'console_scripts': [
        'fairy=flowfairy.core.management:execute'
    ]},
    author='Jonas',
    author_email='jonas@appdo.dk',
    url='https://github.com/WhatDo/flowfairy'
)
