#!/usr/bin/env python
from .context import qosy as qy
import numpy as np
import scipy.sparse as ss

def test_explore_simple():
    # Simple by-hand tests of the
    # "_explore" helper function
    # used in selected_ci to keep track of
    # precomputed structure constants.
    
    # Initial basis is a single Z
    basis = qy.Basis()
    basis += qy.opstring('Z 0')
    
    # Operator is a Heisenberg bond
    H = qy.Operator([1.0, 1.0, 1.0], [qy.opstring('X 0 X 1'), qy.opstring('Y 0 Y 1'), qy.opstring('Z 0 Z 1')])

    # Initially, nothing has been explored.
    explored_basis            = qy.Basis()
    explored_extended_basis   = qy.Basis()
    explored_s_constants_data = dict()

    # First exploration starting from basis
    (com_matrix, extended_basis) = qy.algorithms._explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)

    # After one exploration, the bases should agree.
    assert(set(basis.op_strings) == set(explored_basis.op_strings))
    assert(set(extended_basis.op_strings) == set(explored_extended_basis.op_strings))

    # And the commutant matrices for the structure constants should agree.
    com_matrix2 = qy.algorithms._com_matrix(explored_s_constants_data, H)
    assert(np.allclose(com_matrix.toarray(), com_matrix2.toarray()))
    print('com_mat1 = \n{}\ncom_mat2 = \n{}'.format(com_matrix.toarray(), com_matrix2.toarray()))
    
    # Second exploration starting from the extended basis
    (com_matrix2, extended_basis2) = qy.algorithms._explore(extended_basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)

    expected_extended_basis2 = qy.Basis([qy.opstring('Z 0'), qy.opstring('Z 1')])
    assert(set(extended_basis2.op_strings) == set(expected_extended_basis2.op_strings))

    assert(set(explored_basis.op_strings) == set((basis + extended_basis).op_strings))
    assert(set(explored_extended_basis.op_strings) == set((extended_basis + extended_basis2).op_strings))
    
    # Third exploration, again starting from basis.
    # This one's results should completely agree with the first one's.
    (com_matrix3, extended_basis3) = qy.algorithms._explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
    
    assert(set(extended_basis.op_strings) == set(extended_basis3.op_strings))
    
    # *Note that the extended bases can be permutations of one another.*
    inds_1to3 = np.array([extended_basis3.index(os1) for os1 in extended_basis], dtype=int)

    assert(np.allclose(com_matrix.toarray(), com_matrix3.toarray()))
    
def test_explore_hard():
    # Test the _explore helper function
    # on a more realistic example.
    # In this test, I perform many _explore
    # calculations and check that it agrees
    # with the usual structure constants calculation.

    # Number of sites in chain
    L = 6
    
    # Initial basis is a single Z
    basis = qy.Basis()
    basis += qy.opstring('Z {}'.format(L//2))
    
    # Operator is a disordered Heisenberg chain
    # with magnetic fields in Z.
    coeffs = np.concatenate((0.25*np.ones(3*(L-1)), 0.5*np.ones(L)))
    op_strings = [qy.opstring('{} {} {} {}'.format(op, site, op, site+1)) for site in range(L-1) for op in ['X', 'Y', 'Z']] + [qy.opstring('Z {}'.format(site)) for site in range(L)]
    H = qy.Operator(coeffs, op_strings)
    
    # Initially, nothing has been explored.
    explored_basis            = qy.Basis()
    explored_extended_basis   = qy.Basis()
    explored_s_constants_data = dict()

    # Explore a few times to gather structure constants data
    num_explorations = 3
    for ind_e in range(num_explorations):
        (com_matrix, extended_basis) = qy.algorithms._explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        basis += extended_basis

    # Pick a new basis
    basis = qy.Basis([qy.opstring('X {}'.format(L//2)), qy.opstring('Y {}'.format(L//2)), qy.opstring('Z {}'.format(L//2))])

    # Compute the commutant matrix in two ways:
    # 1: using _explore
    (com_matrix1, extended_basis1) = qy.algorithms._explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
    # 2: using structure_constants
    (s_constants2, extended_basis2) = qy.structure_constants(basis, H._basis, return_extended_basis=True, return_data_tuple=True)

    # *Note that the extended bases can be permutations of one another.*
    inds_1to2 = np.array([extended_basis2.index(os1) for os1 in extended_basis1], dtype=int)

    assert(set(extended_basis1.op_strings) == set(extended_basis2.op_strings))

    com_matrix2 = qy.algorithms._com_matrix(s_constants2, H)
    com_matrix2 = com_matrix2[inds_1to2, :] # Reorder the extended basis
    
    assert(np.allclose(com_matrix1.toarray(), com_matrix2.toarray()))
