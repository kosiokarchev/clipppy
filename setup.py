from setuptools import setup, find_packages

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
        'pyyaml',
        'pyro-ppl',
        'more-itertools'
    ],
    entry_points={
        'console_scripts': [
            'clipppy = clipppy.cli:cli'
        ]
    }
)
