from setuptools import setup

setup(
    name='PyFRep',
    version='0.1',
    description='Library for FRep based modeling',
    author='Pierre-Alain Fayolle, Evgenii Maltsev',
    packages=['FRep'],
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy',
        'polyscope',
        'torch'
    ],
    license='MIT',
    classifiers=(
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    )
)
