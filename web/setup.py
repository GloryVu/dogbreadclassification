from setuptools import setup, find_packages
setup(
    name='dogbreadclassification',
    version='1.0',
    author='Vu Vinh',
    description='training test classification model',
    long_description='A much longer explanation of the project and helpful resources',
    url='https://github.com/GloryVu/dogbreadclassification.git',
    keywords='development, setup, setuptools',
    python_requires='>=3.7, <4',
    packages=find_packages(include=['dogbreadclassification', 'dogbreadclassification.*']),)