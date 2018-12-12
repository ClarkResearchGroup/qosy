#!/usr/bin/env python
from .context import qosy as qy
from .helper import _random_op_string
import numpy as np

def test_product_pauli():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['X', 'Y', 'Z'], [1, 2, 4], 'Pauli')
    os2 = qy.OperatorString(['X', 'X', 'X'], [1, 2, 6], 'Pauli')

    (coeff_prod12, os_prod12) = qy.product(os1, os2)

    expected_os_prod12    = qy.OperatorString(['Z', 'Z', 'X'], [2, 4, 6], 'Pauli')
    expected_coeff_prod12 = -1j

    assert(np.isclose(coeff_prod12, expected_coeff_prod12) and os_prod12 == expected_os_prod12)

    # Check that random Pauli strings square to identity.
    expected_os_prod34    = qy.OperatorString([], [], 'Pauli')
    expected_coeff_prod34 = 1

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Pauli')
        os4 = os3
        
        (coeff_prod34, os_prod34) = qy.product(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)

def test_commutator_pauli():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['X', 'Y', 'Z'], [1, 2, 4], 'Pauli')
    os2 = qy.OperatorString(['X', 'X', 'X'], [1, 2, 6], 'Pauli')

    (coeff_com12, os_com12) = qy.commutator(os1, os2)

    expected_os_com12    = qy.OperatorString(['Z', 'Z', 'X'], [2, 4, 6], 'Pauli')
    expected_coeff_com12 = -2j

    assert(np.isclose(coeff_com12, expected_coeff_com12) and os_com12 == expected_os_com12)
    
    # Check that random Pauli strings commute with themselves.
    expected_os_prod34    = qy.OperatorString([], [], 'Pauli')
    expected_coeff_prod34 = 0

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Pauli')
        os4 = os3
        
        (coeff_prod34, os_prod34) = qy.commutator(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)

def test_anticommutator_pauli():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['X', 'Y', 'Z'], [1, 2, 4], 'Pauli')
    os2 = qy.OperatorString(['X', 'X', 'X'], [1, 2, 6], 'Pauli')

    (coeff_com12, os_com12) = qy.anticommutator(os1, os2)

    expected_os_com12    = qy.OperatorString(['Z', 'Z', 'X'], [2, 4, 6], 'Pauli')
    expected_coeff_com12 = 0

    assert(np.isclose(coeff_com12, expected_coeff_com12) and os_com12 == expected_os_com12)
    
    # Check that {O,O} = 2 O*O = 2 I for random Pauli strings.
    expected_os_prod34    = qy.OperatorString([], [], 'Pauli')
    expected_coeff_prod34 = 2

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Pauli')
        os4 = os3
        
        (coeff_prod34, os_prod34) = qy.anticommutator(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)

def test_product_majorana():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['A', 'B', 'D'], [1, 2, 4], 'Majorana')
    os2 = qy.OperatorString(['A', 'A', 'A'], [1, 2, 6], 'Majorana')

    (coeff_prod12, os_prod12) = qy.product(os1, os2)

    expected_os_prod12    = qy.OperatorString(['D', 'D', 'A'], [2, 4, 6], 'Majorana')
    expected_coeff_prod12 = -1j

    assert(np.isclose(coeff_prod12, expected_coeff_prod12) and os_prod12 == expected_os_prod12)

    # Check that random Majorana strings square to identity.
    expected_os_prod34    = qy.OperatorString([], [], 'Majorana')
    expected_coeff_prod34 = 1

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Majorana')
        os4 = os3
        
        (coeff_prod34, os_prod34) = qy.product(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)

def test_commutator_majorana():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['A', 'B', 'D'], [1, 2, 4], 'Majorana')
    os2 = qy.OperatorString(['A', 'A', 'A'], [1, 2, 6], 'Majorana')

    (coeff_com12, os_com12) = qy.commutator(os1, os2)

    expected_os_com12    = qy.OperatorString(['D', 'D', 'A'], [2, 4, 6], 'Majorana')
    expected_coeff_com12 = -2j

    assert(np.isclose(coeff_com12, expected_coeff_com12) and os_com12 == expected_os_com12)
    
    # Check that random Majorana strings commute with themselves.
    expected_os_prod34    = qy.OperatorString([], [], 'Majorana')
    expected_coeff_prod34 = 0

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Majorana')
        os4 = os3
        
        (coeff_prod34, os_prod34) = qy.commutator(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)

def test_anticommutator_majorana():
    # Check a simple example by hand.
    os1 = qy.OperatorString(['A', 'B', 'D'], [1, 2, 4], 'Majorana')
    os2 = qy.OperatorString(['A', 'A', 'A'], [1, 2, 6], 'Majorana')

    (coeff_com12, os_com12) = qy.anticommutator(os1, os2)

    expected_os_com12    = qy.OperatorString(['D', 'D', 'A'], [2, 4, 6], 'Majorana')
    expected_coeff_com12 = 0

    assert(np.isclose(coeff_com12, expected_coeff_com12) and os_com12 == expected_os_com12)
    
    # Check that {O,O} = 2 O*O = 2 I for random Majorana strings.
    expected_os_prod34    = qy.OperatorString([], [], 'Majorana')
    expected_coeff_prod34 = 2

    np.random.seed(42)
    num_random_operators = 20
    max_num_orbitals     = 6
    possible_orbital_labels = np.arange(6, dtype=int)
    for ind_random in range(num_random_operators):
        os3 = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Majorana')
        os4 = os3

        (coeff_prod34, os_prod34) = qy.anticommutator(os3, os4)
        
        assert(np.isclose(coeff_prod34, expected_coeff_prod34) and os_prod34 == expected_os_prod34)


def test_structure_constants_simple():
    op_strings_basisA = \
                        [qy.OperatorString(['X'], [1], 'Pauli'),
                         qy.OperatorString(['Y'], [1], 'Pauli'),
                         qy.OperatorString(['Z'], [1], 'Pauli')
                        ]

    op_strings_basisB = \
                        [qy.OperatorString(['X', 'X'], [1, 2], 'Pauli')
                        ]

    op_strings_expected_basisC = \
                        [qy.OperatorString(['Z', 'X'], [1, 2], 'Pauli'),
                         qy.OperatorString(['Y', 'X'], [1, 2], 'Pauli')
                        ]
    
    basisA = qy.Basis(op_strings_basisA)
    basisB = qy.Basis(op_strings_basisB)
    expected_basisC = qy.Basis(op_strings_expected_basisC)

    expected_structure_constants = np.array([[0,-2j,0],[0,0,2j]],dtype=complex)
    
    (structure_constants_list, basisC) = qy.algebra.structure_constants(basisA, basisB, return_extended_basis=True)

    assert(np.allclose(structure_constants_list[0].toarray(), expected_structure_constants))

    assert(set(basisC.op_strings) == set(expected_basisC.op_strings))

    
def test_operator_operations():
    
    # Check the product, commutator, and anticommutator of Operators
    # by hand for a simple example.

    # Product example:
    # (X_1 X_2 + Y_1 Y_2)*(X_1 + Y_1) = X_2 - i Z_1 Y_2 + i Z_1 X_2 + Y_2
    coeffsA     = np.ones(2)
    op_stringsA = [qy.opstring('X 1 X 2'), qy.opstring('Y 1 Y 2')]
    
    operatorA = qy.Operator(coeffsA, op_stringsA)
    
    coeffsB     = np.ones(2)
    op_stringsB = [qy.opstring('X 1'), qy.opstring('Y 1')]
    
    operatorB = qy.Operator(coeffsB, op_stringsB)
 
    expected_prodAB = qy.Operator(np.array([1.0, -1j, 1j, 1.0]), \
                                  [qy.opstring('X 2'),     qy.opstring('Z 1 Y 2'), \
                                   qy.opstring('Z 1 X 2'), qy.opstring('Y 2')])
    
    prodAB = qy.product(operatorA, operatorB)

    assert((prodAB - expected_prodAB).norm() < 1e-12)

    # Commutator example:
    # [X_1 X_2 + Y_1 Y_2, X_1 + Y_1] = -2i Z_1 Y_2 + 2i Z_1 X_2
    expected_comAB = qy.Operator(np.array([-2j, 2j]), \
                                  [qy.opstring('Z 1 Y 2'), qy.opstring('Z 1 X 2')])
    
    comAB = qy.commutator(operatorA, operatorB)

    assert((comAB - expected_comAB).norm() < 1e-12)

    # Anticommutator example:
    # {X_1 X_2 + Y_1 Y_2, X_1 + Y_1} = 2 X_2 + 2 Y_2
    expected_anticomAB = qy.Operator(np.array([2, 2]), \
                                  [qy.opstring('X 2'), qy.opstring('Y 2')])
    
    anticomAB = qy.anticommutator(operatorA, operatorB)

    assert((anticomAB - expected_anticomAB).norm() < 1e-12)


    
