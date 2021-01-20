from setuptools import find_packages, setup

setup(
    name='clipppy',
    description='Command Line Interface for Probabilistic Programming in Python',
    version='0.1',
    author='Kosio Karchev',
    author_email='kosiokarchev@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'Click',
        'ruamel.yaml',
        'pyro-ppl',
        'more-itertools',
        'frozendict'
    ],
    entry_points={
        'console_scripts': [
            'clipppy = clipppy.cli:cli'
        ]
    }
)
