#!/usr/bin/env python
"""
This module provides functions for diagonalizing certain
Hermitian operators.

Note
----
Currently, this module only supports diagonalizing 
non-interacting fermionic Hamiltonians.
"""

import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from .operatorstring import OperatorString
from .basis import Basis, Operator
from .transformation import Transformation
from .conversion import convert

def _apply_opstring(op_string, conf, num_orbitals):
    # Helper function to "_to_matrix_opstring"
    # that takes a spin configuration (represented
    # as an integer) and returns the new spin
    # configuration resulting from applying the
    # OperatorString (as well as any acquired
    # coefficient).

    if op_string.op_type != 'Pauli':
        raise NotImplementedError('The application of an OperatorString of op_type {} to a configuration is not supported.'.format(op_string.op_type))
    
    new_conf  = conf
    new_coeff = op_string.prefactor # Should be 1 for Pauli strings
    for (operator, label) in op_string:
        mask = (1 << (num_orbitals-1-label))
        if operator == 'X':
            # Flip the bit at the site
            new_conf = new_conf ^ mask
        elif operator == 'Y':
            # Check whether the bit corresponding to site is 1
            if new_conf & mask != 0:
                new_coeff *= -1j
            else:
                new_coeff *= 1j
            # Flip the bit at the site
            new_conf = new_conf ^ mask
        elif operator == 'Z':
            # Check whether the bit corresponding to site is 1
            if new_conf & mask != 0:
                new_coeff *= -1.0

    return (new_coeff, new_conf)

def _to_matrix_opstring(op_string, num_orbitals, return_tuple=False):
    # Converts an OperatorString to a sparse matrix.
    if op_string.op_type == 'Pauli':
        row_inds = []
        col_inds = []
        data     = []
        
        for conf in range(2**num_orbitals):
            (coeff, new_conf) = _apply_opstring(op_string, conf, num_orbitals)

            row_inds.append(new_conf)
            col_inds.append(conf)
            data.append(coeff)

        if return_tuple:
            return (row_inds, col_inds, data)
        else:
            return ss.csc_matrix((data, (row_inds, col_inds)), shape=(2**num_orbitals, 2**num_orbitals), dtype=complex)
    else:
        raise NotImplementedError('Not finished yet.')

# TODO: document
def to_matrix(operator, num_orbitals):
    if isinstance(operator, OperatorString):
        return _to_matrix_opstring(operator, num_orbitals)
    elif isinstance(operator, Operator):
        row_inds = []
        col_inds = []
        data     = []
        
        for (coeff, op_string) in operator:
            (os_row_inds, os_col_inds, os_data) =_to_matrix_opstring(op_string, num_orbitals, return_tuple=True)
            
            new_os_data = [coeff * datum for datum in os_data]
            row_inds += os_row_inds
            col_inds += os_col_inds
            data     += new_os_data
            
        result = ss.csc_matrix((data, (row_inds, col_inds)), shape=(2**num_orbitals, 2**num_orbitals), dtype=complex)
        return result
    else:
        raise ValueError('Cannot convert an operator of type {} to a matrix.'.format(type(operator)))    

# TODO: document, test
def to_vector(matrix, basis, num_orbitals):
    basis_matrices = [_to_matrix_opstring(op_string, num_orbitals) for op_string in basis]
    
    overlaps = np.array([(matrix.H).dot(mat).trace() for mat in basis_matrices], dtype=complex)
    
    return overlaps
    
# TODO: document, test
def to_operator(matrix, basis, num_orbitals):
    coeffs = to_vector(matrix, basis)

    op = qy.Operator(coeffs, basis.op_strings, op_type=basis.op_strings[0].op_type)

    return op

# TODO: document, test
# Note: only for spin-1/2 vectors. Not taking into account
# signs due to reordering fermionic operators.
def apply_transformation(transformation, vector, num_orbitals):
    permutation = transformation.info
    if not (permutation is not None and len(permutation) == num_orbitals):
        raise ValueError('Transformation is not a valid permutation. Only permutations supported.')
    
    permutation = np.array(permutation, dtype=int)
    
    new_vector = np.zeros(2**num_orbitals, dtype=complex)
    
    for conf in range(2**num_orbitals):
        new_conf = 0
        for label in range(num_orbitals):
            new_label = permutation[label]
            mask_old_label = (1 << (num_orbitals-1-label))
            mask_new_label = (1 << (num_orbitals-1-new_label))
            
            new_conf += (conf & mask_old_label != 0) * mask_new_label
            
        new_vector[new_conf] = vector[conf]
    
    return new_vector
            
# TODO: document, test
def diagonalize(operator, num_orbitals, mode='Hermitian', num_vecs=None):
    matrix = to_matrix(operator, num_orbitals)

    if mode == 'Hermitian':
        if num_vecs is None:
            (evals, evecs) = nla.eigh(matrix.toarray())
        else:
            (evals, evecs) = ssla.eigsh(matrix, k=num_vecs, which='SA')

        inds_sort = np.argsort(evals)
        evals = evals[inds_sort]
        evecs = evecs[:, inds_sort]
    else:
        if num_vecs is None:
            (evals, evecs) = nla.eig(matrix.toarray())
        else:
            (evals, evecs) = ssla.eigs(matrix, k=num_vecs)
    
    return (evals, evecs)
    
# TODO: document, test
def diagonalize_quadratic(operator):

    op = operator
    if operator.op_type == 'Fermion':
        op = convert(operator, 'Majorana') 
    elif operator.op_type != 'Majorana':
        raise ValueError('Diagonalization of an Operator of type {} is not supported.'.format(operator.op_type))

    # Some relevant notes: https://physics.stackexchange.com/questions/383659/is-a-quadratic-majorana-hamiltonian-exactly-solvable
    
    data = []
    row_inds = []
    col_inds = []
    for (coeff, op_string) in op:
        if len(op_string.orbital_operators) == 1 and op_string.orbital_operators[0] == 'D':
            # d_j = i a_j b_j
            # My convention is that the index of a_j is 2j
            #                   and the index of b_j is 2j+1
            orb_label = op_string.orbital_labels[0]
            i = 2*orb_label
            j = 2*orb_label+1
        elif len(op_string.orbital_operators) == 2 and op_string.orbital_operators[0] in ['A','B'] and op_string.orbital_operators[1] in ['A', 'B']:
            orb_label = op_string.orbital_labels[0]
            if op_string.orbital_operators[0] == 'A':
                i = 2*orb_label
            else:
                i = 2*orb_label+1

            orb_label = op_string.orbital_labels[1]
            if op_string.orbital_operators[1] == 'A':
                j = 2*orb_label
            else:
                j = 2*orb_label+1
        else:
            raise ValueError('Unsupported diagonalization. OperatorString in Operator is not quadratic:\n {}'.format(op_string))

        # A_{ij}
        row_inds.append(i)
        col_inds.append(j)
        data.append(0.5*coeff)

        # A_{ji} = -A_{ij}
        row_inds.append(j)
        col_inds.append(i)
        data.append(-0.5*coeff)

    # Number of orbitals (goes up to largest orbital encountered).
    N = np.max(row_inds) // 2

    A = ss.csc_matrix((data, (row_inds, col_inds)), shape=(2*N,2*N), dtype=float)
    #matrix = ss.bmat([[None, A], [-A, None]]).toarray()
    #(evals, evecs) = nla.eigh(matrix)

    # Calculate the 2x2 block (Schur) decomposition of A.
    (T, Z) = ssla.schur(A.toarray(), output='real')

    # N (non-negative) eigenvalues
    eigvals = np.array([T[2*n,2*n+1] for n in range(N)])

    # 2N x 2N, the Majorana operators are organized as
    # a_1, b_1, a_2, b_2, ...
    eigvecs_Majorana = Z
    
    # 2N x 2N, the Fermion operators are organized as
    # c_1, c_2, c_3, ..., c_1^\dagger, c_2^\dagger
    eigvecs_Fermion = np.zeros((2*N,2*N), dtype=complex)
    for m in range(N):
        for n in range(N):
            # The \bar{c_m} vectors
            # \bar{c}_m = \sum_n a_n c_n + ...
            eigvecs_Fermion[m,n] = 0.5 * (    Z[2*m,2*n]   + 1j*Z[2*m,2*n+1] \
                                          -1j*Z[2*m+1,2*n] +    Z[2*m+1,2*n+1])

            # \bar{c}_m = ... + \sum_n b_n c_n^\dagger
            eigvecs_Fermion[m,2*n+1] = 0.5 * (    Z[2*m,2*n]   + 1j*Z[2*m,2*n+1] \
                                              +1j*Z[2*m+1,2*n]    - Z[2*m+1,2*n+1])
            
            # The \bar{c_m}^\dagger vectors
            # \bar{c}_m^\dagger = \sum_n a_n c_n + ...
            eigvecs_Fermion[2*m+1,n] = 0.5 * (    Z[2*m,2*n]   - 1j*Z[2*m,2*n+1] \
                                              -1j*Z[2*m+1,2*n] -    Z[2*m+1,2*n+1])

            # \bar{c}_m^\dagger = ... + \sum_n b_n c_n^\dagger
            eigvecs_Fermion[2*m+1,2*n+1] = 0.5 * (    Z[2*m,2*n]   + 1j*Z[2*m,2*n+1] \
                                                  -1j*Z[2*m+1,2*n]    + Z[2*m+1,2*n+1])
            
    # The ground state energy
    constant_shift = -2.0*np.sum(eigvals)
    
    return (constant_shift, eigvals, eigvecs_Fermion, eigvecs_Majorana)
