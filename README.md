# QOSY: Quantum Operators from SYmmetries

Qosy is a python package for constructing quantum operators with desired symmetry properties, which can include continuous or discrete symmetries. 

## Features

Qosy provides a simple set of tools to
- specify a basis of Hermitian operators
- specify desired continuous symmetries represented by integrals of motion
- specify desired discrete symmetries represented by transformation rules
- generate all the Hermitian operators in the basis of operators with these symmetries

The tools can be used as either (1) a "forward method" to automatically generate the symmetries of a Hamiltonian or as (2) an "inverse method" to automatically construct Hamiltonians with desired symmetries.

## Getting Started

### Prerequisites

Qosy requires the following software installed on your platform:
- [Python](https://www.python.org/)
- [Numpy](https://www.numpy.org/)
- [Scipy](https://www.scipy.org/)
- If you want the visualization features: [Matplotlib](https://www.matplotlib.org/)
- If you want to use the tutorial ipython notebooks: [IPython](https://www.ipython.org/)
- If you want to build the documentation: [Sphinx](http://www.sphinx-doc.org/)
- If you want to run the tests: [pytest](https://pytest.org)

### Installing

To copy the development version of the code to your machine, type
```
git clone https://github.com/ClarkResearchGroup/qosy.git
```
To install, type
```
cd qosy
python setup.py install --user
```
or add the qosy folder to your PYTHONPATH environment variable.

## Documentation

To generate the documentation on your machine, type

```
cd qosy/docs
make html
```

To view it, type

```
firefox _build/html/index.html
```

<!--- TODO: Create a link to the documentation on github. -->

## Testing

To test Qosy after installation (recommended), type
```
cd qosy/tests
pytest
```

## Tutorials

<!--- TODO: Create a few tutorials. At least two: one of forward method, one of inverse method. -->

## Authors

- Eli Chertkov

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) for details.

## References

Based on work presented in

E. Chertkov, B. Villalonga, and B. K. Clark, “Engineering Topological Models with a General-Purpose Symmetry-to-Hamiltonian Approach,” [arXiv:1910.10165](http://arxiv.org/abs/1910.10165) (2019).
