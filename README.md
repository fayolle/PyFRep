# F-Rep based library for Python 
Simple Python API for F-Rep based geometric modeling. 

## Requirements 
The dependencies are: 
- NumPy
- Scikit image
- SciPy
- Polyscope
See also the file 'requirements.txt'

## Installation 
Create a Python virtualenv and install the dependencies with pip
```bash
python -m venv venv
./venv/Scripts/Activate
pip install -r requirements.txt
```

## Example 
See some of the examples in the directory '/examples'. For example
```bash
python examples/gyroid_example_render.py
```
should create a window and show an example of a periodic micro-structure. 
