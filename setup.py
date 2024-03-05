# Run this script from the origin folder as:
#   > "python setup.py clean" in order to clean previous builds
#   > "python setup.py test" in order to execute all the unittests
#   > "python setup.py sdist" in order to build the library
#
# The package can then be published with:
#   > twine upload dist/*

from setuptools import find_packages, setup

# set up the library metadata and make the build
with open('README.md', 'r') as readme:
    setup(
        name='benchmarks',
        version='0.1.0',
        maintainer='Luca Giuliani',
        maintainer_email='luca.giuliani13@unibo.it',
        author='University of Bologna - DISI',
        description='An Interactive Benchmark Library',
        long_description=readme.read(),
        long_description_content_type='text/markdown',
        packages=find_packages(include=['benchmarks*']),
        python_requires='~=3.10',
        install_requires=[
            'Dickens~=2.1.1',
            'dill~=0.3.7',
            'matplotlib~=3.8.2',
            'numpy~=1.26.2',
            'pandas~=2.1.3',
            'scikit-learn~=1.3.2'
        ],
        test_suite='test'
    )
