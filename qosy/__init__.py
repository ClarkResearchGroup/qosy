#!/usr/bin/env python
"""
Qosy: Quantum Operators from SYmmetry
=====================================

Qosy is a scientific python library that provides algorithms for constructing quantum operators with desired symmetries. Qosy can be used, for example, as an "inverse method" to construct interacting Hamiltonians from desired continuous or discrete symmetries. It can also be used as a "forward method" to find the symmetries of a Hamiltonian.

Provides
  1. Objects that can represent vector spaces and bases of many-body quantum operators.
  2. Numerical methods for finding many-body quantum operators with desired continuous and discrete symmetries.
  
How to use the documentation
----------------------------
The docstring examples assume that `qosy` has been imported
  >>> import qosy

In python, you can access the `docstring` documentation 
of a particular method with
  >>> help(qosy.opstring)

Description of available modules
--------------------------------
operatorstring
  Defines an OperatorString object.
basis
  Defines a Basis and Operator object, provides methods for creating useful bases of operators.
conversion
  Provides methods for converting between different types of OperatorStrings.
transformation
  Defines a Transformation object, provides methods for creating common discrete transformations.
algebra
  Provides methods for computing commutator and anticommutators.
core
  Provides methods for constructing operators with desired continuous and discrete symmetries.
lattice
  Defines a Lattice object useful for crystal systems.
tools
  Provides convenience functions for Qosy.

Comments
--------
Most functions and objects can be directly used 
as if they were in the `qosy` module:
  >>> XY = qosy.opstring('X 1 Y 2') # instead of qosy.operatorstring.opstring
  >>> T  = qosy.time_reversal()     # instead of qosy.transformation.time_reversal

Operator objects are convenience objects with useful
methods and simple syntax, but they are less efficient
to manipulate than numpy/scipy arrays. When possible,
use only Bases and numpy/scipy arrays, rather than
Operators for calculations.

"""

# Loads global variables.
from .config import *

# Loads the classes and functions defined in each module.
from .lattice        import UnitCell, Lattice, plot, show
from .operatorstring import OperatorString, opstring
from .basis          import Basis, Operator, cluster_basis
from .conversion     import convert, conversion_matrix
from .algebra        import *
from .transformation import Transformation, symmetry_matrix, time_reversal, particle_hole, charge_conjugation, label_permutation

# Loads the utility functions.
from .tools          import *

# Loads the core functions of qosy.
from .core           import *
