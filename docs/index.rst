.. qosy documentation master file, created by
   sphinx-quickstart on Fri Nov 30 15:05:36 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QOSY: Quantum Operators from SYmmetry
=====================================

Qosy is a scientific python library that provides algorithms for constructing quantum operators with desired symmetries. Qosy can be used, for example, as an "inverse method" to construct interacting Hamiltonians from desired continuous or discrete symmetries. It can also be used as a "forward method" to find the symmetries of a Hamiltonian.

Provides:
  1. Objects that can represent vector spaces and bases of many-body quantum operators.
  2. Numerical methods for finding many-body quantum operators with desired continuous and discrete symmetries.
  
How to use the documentation
----------------------------
The docstring examples assume that ``qosy`` has been imported
  >>> import qosy

In python, you can access the `docstring` documentation 
of a particular method with
  >>> help(qosy.opstring)

Description of available modules
--------------------------------
:py:mod:`qosy.operatorstring`
  Defines an OperatorString object.
:py:mod:`qosy.basis`
  Defines a Basis and Operator object, provides methods for creating useful bases of operators.
:py:mod:`qosy.conversion`
  Provides methods for converting between different types of OperatorStrings.
:py:mod:`qosy.transformation`
  Defines a Transformation object, provides methods for creating common discrete transformations.
:py:mod:`qosy.algebra`
  Provides methods for computing commutators and anticommutators of operators.
:py:mod:`qosy.core`
  Provides methods for constructing operators with desired continuous and discrete symmetries.
:py:mod:`qosy.lattice`
  Defines UnitCell and Lattice objects useful for analyzing crystalline systems.
:py:mod:`qosy.visualization`
  Provides tools for plotting and printing lattices and operators defined on lattices.
:py:mod:`qosy.tools`
  Provides convenience functions for ``qosy``.

Comments
--------
Most functions and objects can be directly used 
as if they were in the ``qosy`` module:
  >>> XY = qosy.opstring('X 1 Y 2') # instead of qosy.operatorstring.opstring
  >>> T  = qosy.time_reversal()     # instead of qosy.transformation.time_reversal

Operator objects are convenience objects with useful
methods and simple syntax, but they are less efficient
to manipulate than numpy/scipy arrays. When possible,
use only Bases and numpy/scipy arrays, rather than
Operators for calculations.   

-------------------

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

