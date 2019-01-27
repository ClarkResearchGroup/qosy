#!/usr/bin/env python
"""
This module includes the Selected CI algorithm
for greedily finding sparse operators that
commute with a given operator.
"""

import warnings
import copy
import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from .tools import intersection, gram_schmidt, sparsify
from .basis import Basis, Operator
from .algebra import commutator, commutant_matrix, structure_constants
    
def _truncate_basis(basis, vector, threshold=1e-6, max_basis_size=100):
    # Truncates a Basis according to the coefficients
    # of the vector in the Basis.

    abs_coeffs = np.abs(vector)
    inds = np.argsort(np.abs(abs_coeffs))[::-1]
    inds = [ind for ind in inds if abs_coeffs[ind] > threshold]
    if len(inds) >= max_basis_size:
        inds = inds[:max_basis_size]

    new_op_strings = [basis[ind] for ind in inds]

    result = Basis(new_op_strings)

    return result

def _cdagc(s_constants, operator):
    # Computes C_O^\dagger C_O, where C_O is the
    # commutant matrix of operator O,
    # from the structure constants and O.
    
    dim_extended_basis = int(s_constants[0].shape[0])
    dim_basis          = int(s_constants[0].shape[1])

    commutant_matrix = ss.csc_matrix((dim_extended_basis, dim_basis), dtype=complex)
    
    for ind_os in range(len(operator._basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os]

    CDagC = (commutant_matrix.H).dot(commutant_matrix)
        
    return CDagC

def _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data):
    # Explore the space of OperatorStrings, starting from
    # the given Basis. Update the explored_basis, explored_extended_basis,
    # and explored_s_constants variables as you go.

    # First, find the part of basis that is unexplored.
    unexplored_basis = Basis([os for os in basis if os not in explored_basis])

    # Then explore that part.
    (unexplored_s_constants_data, unexplored_extended_basis) = structure_constants(unexplored_basis, H._basis, return_extended_basis=True, return_data_tuple=True)

    # Update the explored Bases.
    explored_basis          += unexplored_basis
    explored_extended_basis += unexplored_extended_basis

    # Update the explored structure constants data.
    for ind_os in range(len(H)):
        # The new information found from exploring.
        row_inds_unexplored = [explored_extended_basis.index(unexplored_extended_basis.op_strings[ind_eb_os]) for ind_eb_os in unexplored_s_constants_data[ind_os][0]]
        col_inds_unexplored = [explored_basis.index(unexplored_basis.op_strings[ind_b_os]) for ind_b_os in unexplored_s_constants_data[ind_os][1]]
        data_unexplored     = unexplored_s_constants_data[ind_os][2]

        # The old information from previous exploration.
        old_row_inds = explored_s_constants_data[ind_os][0]
        old_col_inds = explored_s_constants_data[ind_os][1]
        old_data     = explored_s_constants_data[ind_os][2]

        # The update
        explored_s_constants_data[ind_os] = (old_row_inds + row_inds_unexplored, old_col_inds + col_inds_unexplored, old_data + data_unexplored)
    
    # From the collected information, find the
    # extended_basis corresponding to basis.
    inds_basis_to_x = [explored_basis.index(os) for os in basis]
    inds_x_to_basis = dict()
    for ind_b in range(len(basis)):
        inds_x_to_basis[inds_basis_to_x[ind_b]] = ind_b 
    
    extended_basis = Basis()
    inds_extended_basis_to_x = []
    for ind_os in range(len(H)):
        (inds_x_eb, inds_x_b, _) = explored_s_constants_data[ind_os]
        for (ind_x_eb, ind_x_b) in zip(inds_x_eb, inds_x_b):
            if ind_x_b in inds_x_to_basis and explored_extended_basis[ind_x_eb] not in extended_basis:
                extended_basis += explored_extended_basis[ind_x_eb]
                inds_extended_basis_to_x.append(ind_x_eb)
    inds_x_to_extended_basis = dict()
    for ind_eb in range(len(extended_basis)):
        inds_x_to_extended_basis[inds_extended_basis_to_x[ind_eb]] = ind_eb 
        
    # From the information collected from the
    # explored bases, construct the s_constants
    # that correspond to basis and extended basis.
    s_constants = []
    for ind_os in range(len(H)):
        (inds_explored_eb, inds_explored_b, explored_data) = explored_s_constants_data[ind_os]
        row_inds = []
        col_inds = []
        data     = []
        for (ind_explored_eb, ind_explored_b, explored_datum) in zip(inds_explored_eb, inds_explored_b, explored_data):
            if ind_explored_b in inds_x_to_basis and ind_explored_eb in inds_x_to_extended_basis:
                row_ind = inds_x_to_extended_basis[ind_explored_eb]
                col_ind = inds_x_to_basis[ind_explored_b]
                row_inds.append(row_ind)
                col_inds.append(col_ind)
                data.append(explored_datum)
                
        s_constants.append(ss.csr_matrix((data, (row_inds, col_inds)), shape=(len(extended_basis), len(basis)), dtype=complex))
    
    return (s_constants, extended_basis)

# TODO: document
def selected_ci1(initial_operator, H, num_steps, threshold=1e-6, max_basis_size=100):
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf
    
    for step in range(num_steps):       
        # Expand basis twice.
        (s_constants, temp_extended_basis) = structure_constants(basis, H._basis, return_extended_basis=True)
        basis += temp_extended_basis
        (s_constants, temp_extended_basis) = structure_constants(basis, H._basis, return_extended_basis=True)
        basis += temp_extended_basis
        
        print('Expanded basis: {}'.format(len(basis)))

        # Find best operator in basis
        com_matrix = commutant_matrix(basis, H)
        CDagC = (com_matrix.H).dot(com_matrix)

        if len(basis) < 20:
            (evals, evecs) = nla.eigh(CDagC.toarray())
        else:
            (evals, evecs) = ssla.eigsh(CDagC, k=4, which='SM')
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]
            
        operator = Operator(evecs[:,0], copy.deepcopy(basis.op_strings))
        operators.append(copy.deepcopy(operator))
        com_norms.append(evals[0])
        
        if np.abs(evals[0]) < best_com_norm:
            best_operator = copy.deepcopy(operator)
            best_com_norm = evals[0]
        
        # Truncate basis
        basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size)
        print('Truncated basis: {}'.format(len(basis)))
        
    return (best_operator, best_com_norm, operators, com_norms)

# TODO: document
def selected_ci2(initial_operator, H, num_steps, threshold=1e-6, max_basis_size=100, explored_basis=None, explored_extended_basis=None, explored_s_constants_data=None):

    # Keep track of all OperatorStrings considered
    # during the calculation.
    if explored_basis is None:
        explored_basis = Basis()
    if explored_extended_basis is None:
        explored_extended_basis = Basis()
    if explored_s_constants_data is None:
        explored_s_constants_data = [([],[],[])]*len(H)
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf
    
    for step in range(num_steps):       
        # Expand basis twice.
        (_, temp_extended_basis) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        basis += temp_extended_basis
        (_, temp_extended_basis) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        basis += temp_extended_basis
        
        print('Expanded basis: {}'.format(len(basis)))

        # Find best operator in basis
        (s_constants, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        CDagC = _cdagc(s_constants, H)
        
        if len(basis) < 20:
            (evals, evecs) = nla.eigh(CDagC.toarray())
        else:
            (evals, evecs) = ssla.eigsh(CDagC, k=4, which='SM')
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]
        
        operator = Operator(evecs[:,0], copy.deepcopy(basis.op_strings))
        operators.append(copy.deepcopy(operator))
        com_norms.append(evals[0])

        if np.abs(evals[0]) < best_com_norm:
            best_operator = copy.deepcopy(operator)
            best_com_norm = evals[0]
        
        # Truncate basis
        basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size)
        print('Truncated basis: {}'.format(len(basis)))
        
    return (best_operator, best_com_norm, operators, com_norms)
