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
from .operatorstring import opstring
from .basis import Basis, Operator
from .algebra import commutator, commutant_matrix, structure_constants, product, anticommutator
from .transformation import symmetry_matrix
    
def _truncate_basis(basis, vector, threshold=1e-6, max_basis_size=100, filter_fun=None):
    # Truncates a Basis according to the coefficients
    # of the vector in the Basis.
    if filter_fun is not None:
        vec = filter_fun(vector, basis)
    else:
        vec = np.copy(vector)
    
    abs_coeffs = np.abs(vec)
    inds = np.argsort(np.abs(abs_coeffs))[::-1]
    inds = [ind for ind in inds if abs_coeffs[ind] > threshold]
    if len(inds) >= max_basis_size:
        inds = inds[:max_basis_size]

    new_op_strings = [basis[ind] for ind in inds]

    result = Basis(new_op_strings)

    return result

def _truncate_operator(operator, threshold=1e-6, max_basis_size=100, filter_fun=None):
    truncated_basis = _truncate_basis(operator._basis, operator.coeffs, threshold=threshold, max_basis_size=max_basis_size, filter_fun=filter_fun)

    coeffs = [operator.coeffs[operator._basis.index(os)] for os in truncated_basis]
    result = Operator(coeffs, truncated_basis.op_strings)
    return result

def _l_matrix(s_constants, operator):
    # Computes C_O, where C_O is the
    # commutant matrix of operator O,
    # from the structure constants and O.

    row_inds = []
    col_inds = []
    data     = []
    for (coeff, os) in operator:
        (ext_basis_inds, basis_inds, s_data) = s_constants[os]
        for (row_ind, col_ind, datum) in zip(ext_basis_inds, basis_inds, s_data):
            row_inds.append(row_ind)
            col_inds.append(col_ind)
            data.append(coeff*datum)

    liouvillian_matrix = ss.csc_matrix((data, (row_inds, col_inds)), dtype=complex)
        
    return liouvillian_matrix

def _com_matrix(s_constants, operator):
    # Computes C_O^\dagger C_O, where C_O is the
    # commutant matrix of operator O,
    # from the structure constants and O.

    liouvillian_matrix = _l_matrix(s_constants, operator)
    com_matrix = (liouvillian_matrix.H).dot(liouvillian_matrix)
        
    return com_matrix

def _orthogonalize_com_matrix(com_matrix, basis, orth_ops):
    # Update C_H^\dagger C_H to make its
    # ground state orthogonal to the given
    # operators.

    new_com_matrix = com_matrix.copy()
    
    # Create a projection matrix that projects
    # out of the space spanned by the given operators.
    proj_ops = ss.csc_matrix((len(basis),len(basis)), dtype=complex)
    for orth_op in orth_ops:
        row_inds = []
        col_inds = []
        data     = []
        for (coeff, os) in orth_op:
            if os in basis:
                data.append(coeff)
                row_inds.append(basis.index(os))
                col_inds.append(0)
        orth_vec = ss.csc_matrix((data, (row_inds, col_inds)), shape=(len(basis),1), dtype=complex)

        proj_ops += orth_vec.dot(orth_vec.H)
        
    large_number = 1e10
    new_com_matrix += large_number * proj_ops

    return new_com_matrix.real

def _symmetrize_com_matrix(com_matrix, basis, transformations):
    # Update the commutant matrix to make its
    # ground state obey the symmetries of the
    # given transformations.

    new_com_matrix = com_matrix.copy()

    large_number = 1e10
    for transformation in transformations:
        matrix = symmetry_matrix(basis, transformation)
        matrix = 0.5*(matrix + matrix.H)
        matrix = ss.eye(len(basis), format='csc') - matrix
        
        new_com_matrix += large_number * matrix
    
    return new_com_matrix

def _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data, allowed_labels=None):
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
    for (coeff, os) in H:
        # The new information found from exploring.
        row_inds_unexplored = [explored_extended_basis.index(unexplored_extended_basis.op_strings[ind_eb_os]) for ind_eb_os in unexplored_s_constants_data[os][0]]
        col_inds_unexplored = [explored_basis.index(unexplored_basis.op_strings[ind_b_os]) for ind_b_os in unexplored_s_constants_data[os][1]]
        data_unexplored     = unexplored_s_constants_data[os][2]

        # The old information from previous exploration.
        if os in explored_s_constants_data:
            old_row_inds = explored_s_constants_data[os][0]
            old_col_inds = explored_s_constants_data[os][1]
            old_data     = explored_s_constants_data[os][2]
        else:
            old_row_inds = []
            old_col_inds = []
            old_data     = []
            
        # The update
        explored_s_constants_data[os] = (old_row_inds + row_inds_unexplored, old_col_inds + col_inds_unexplored, old_data + data_unexplored)
    
    # From the collected information, find the
    # extended_basis corresponding to basis.
    inds_basis_to_x = [explored_basis.index(os) for os in basis]
    inds_x_to_basis = dict()
    for ind_b in range(len(basis)):
        inds_x_to_basis[inds_basis_to_x[ind_b]] = ind_b 
    
    extended_basis = Basis()
    inds_extended_basis_to_x = []
    for (coeff, os) in H:
        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is not None and len(set(os.orbital_labels).intersection(allowed_labels)) == 0:
            continue # TODO: check if I am doing this right
        
        (inds_x_eb, inds_x_b, _) = explored_s_constants_data[os]
        for (ind_x_eb, ind_x_b) in zip(inds_x_eb, inds_x_b):
            if ind_x_b in inds_x_to_basis and explored_extended_basis[ind_x_eb] not in extended_basis:
                extended_basis += explored_extended_basis[ind_x_eb]
                inds_extended_basis_to_x.append(ind_x_eb)
    inds_x_to_extended_basis = dict()
    for ind_eb in range(len(extended_basis)):
        inds_x_to_extended_basis[inds_extended_basis_to_x[ind_eb]] = ind_eb 

    # From the information collected from the
    # explored bases, construct the commutant matrix.
    row_inds = []
    col_inds = []
    data     = []
    for (coeff, os) in H:
        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is None or len(set(os.orbital_labels).intersection(allowed_labels)) != 0:
            (inds_explored_eb, inds_explored_b, explored_data) = explored_s_constants_data[os]
            
            for (ind_explored_eb, ind_explored_b, explored_datum) in zip(inds_explored_eb, inds_explored_b, explored_data):
                if ind_explored_b in inds_x_to_basis and ind_explored_eb in inds_x_to_extended_basis:
                    row_ind = inds_x_to_extended_basis[ind_explored_eb]
                    col_ind = inds_x_to_basis[ind_explored_b]
                    
                    row_inds.append(row_ind)
                    col_inds.append(col_ind)
                    data.append(coeff * explored_datum)
                
    l_matrix = ss.csc_matrix((data, (row_inds, col_inds)), shape=(len(extended_basis), len(basis)), dtype=complex)

    return (l_matrix, extended_basis)

# TODO: document
def selected_ci_simple(initial_operator, H, num_steps, threshold=1e-6, max_basis_size=100):
    
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
        l_matrix = liouvillian_matrix(basis, H)
        com_matrix = ((l_matrix.H).dot(l_matrix)).real

        if len(basis) < 20:
            (evals, evecs) = nla.eigh(com_matrix.toarray())
        else:
            (evals, evecs) = ssla.eigsh(com_matrix, k=8, which='SM')
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
def selected_ci(initial_operator, H, num_steps, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0, filter_fun=None, min_num_evecs=2, max_num_evecs=2, eval_threshold=None, filter_trunc_fun=None):
    
    # Keep track of all OperatorStrings considered
    # during the calculation.
    if explored_data is None:
        explored_basis = Basis()
        explored_extended_basis = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf

    previous_previous_basis = None
    previous_basis = copy.deepcopy(basis)
    for step in range(num_steps):
        # Expand basis twice.
        (_, temp_extended_basis) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        #basis += temp_extended_basis
        (_, temp_extended_basis) = _explore(temp_extended_basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        basis += temp_extended_basis

        # End early if the algorithm has converged to a basis.
        if set(basis.op_strings) == set(previous_basis.op_strings):
            break
        # Also end early if it gets into a cycle of switching between two bases.
        elif previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
            break
        else:
            previous_previous_basis = copy.deepcopy(previous_basis)
            previous_basis = copy.deepcopy(basis)
            
        print('Expanded basis: {}'.format(len(basis)))

        # Find best operator in basis
        (s_constants, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        com_matrix = _com_matrix(s_constants, H).real
        
        if len(basis) < 2*max_num_evecs:
            (evals, evecs) = nla.eigh(com_matrix.toarray())
            num_vecs = np.minimum(len(evals), max_num_evecs)
            evals = evals[0:num_vecs]
            evecs = evecs[:,0:num_vecs]
        else:
            maxiter = 10*int(com_matrix.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(com_matrix, k=max_num_evecs, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
                    
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]

        if eval_threshold is not None:
            inds_valid = np.where(evals < eval_threshold)[0]
            if len(inds_valid) > min_num_evecs:
                evals = evals[inds_valid]
                evecs = evecs[:, inds_valid]
            else:
                evals = evals[0:min_num_evecs]
                evecs = evecs[:, 0:min_num_evecs]

        if filter_fun is not None:
            (vector, com_norm) = filter_fun(evals, evecs, basis)
        else:
            vector   = evecs[:,0]
            com_norm = evals[0]
        
        operator = Operator(vector, copy.deepcopy(basis.op_strings))
        operators.append(copy.deepcopy(operator))
        com_norms.append(com_norm)

        if np.abs(com_norm) < best_com_norm:
            best_operator = copy.deepcopy(operator)
            best_com_norm = com_norm
        
        # Truncate basis
        basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size, filter_fun=filter_trunc_fun)
        print('Truncated basis: {}'.format(len(basis)))
        
    return (best_operator, best_com_norm, operators, com_norms)

# TODO: document
def selected_ci_sweep(initial_operator, H, num_sweeps, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0, pre_filter_fun=None, post_filter_fun=None, min_num_evecs=2, max_num_evecs=2, eval_threshold=None, trunc_filter_fun=None, expand_unitary=None, norm_fun=None, orth_ops=None):

    H_labels = np.array([label for os in H._basis for (_, label) in os], dtype=int)
    max_label = np.max(H_labels)
    
    # Keep track of all OperatorStrings considered
    # during the calculation.
    if explored_data is None:
        explored_basis = Basis()
        explored_extended_basis = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf

    previous_previous_basis = None
    previous_basis = copy.deepcopy(basis)
    direction = 1
    for sweep in range(num_sweeps):
        # All the orbital labels in the current basis.
        labels = np.array([label for os in basis for (_, label) in os], dtype=int)
        site0 = np.min(labels)
        sitef = np.max(labels)
        sites = np.arange(site0, sitef+1)[::direction]
        num_steps = len(sites)
        
        print('Sweep {}/{} (range {} to {})'.format(sweep+1, num_sweeps, site0, sitef))
        
        for site in sites:
            print(' site {}'.format(site))

            # The terms in the Hamiltonian that we consider
            # in each expansion are on the current two or
            # four sites of each step of our sweep.
            allowed_sites1 = set([site, np.minimum(site+1, max_label)])
            allowed_sites2 = set([np.maximum(0, site-1), site, np.minimum(site+1, max_label), np.minimum(site+2, max_label)])
            
            # Expand basis twice, but only add the terms
            # two steps away to the basis.
            (_, temp_extended_basis1) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data, allowed_labels=allowed_sites1)
            #basis += temp_extended_basis
            (_, temp_extended_basis2) = _explore(temp_extended_basis1, H, explored_basis, explored_extended_basis, explored_s_constants_data, allowed_labels=allowed_sites2)

            #print('basis = \n{}'.format(basis))
            #print('extended basis 1 = \n{}'.format(temp_extended_basis1))
            #print('extended basis 2 = \n{}'.format(temp_extended_basis2))
            
            basis += temp_extended_basis2

            # TODO: check if this works
            # Find out what terms you need to add to the
            # basis to help make the operator more unitary,
            # by cancelling the parts that don't square to I.
            if len(operators) > 0 and expand_unitary is not None:
                identity = Operator([1.0], [opstring('I', 'Pauli')])
                op = _truncate_operator(operators[-1], max_basis_size=expand_unitary)
                op_squared = product(op, op)
                non_unitary_part = op_squared - identity
                
                (_, temp_extended_basis3) = _explore(non_unitary_part._basis, H, explored_basis, explored_extended_basis, explored_s_constants_data, allowed_labels=allowed_sites2)
                basis += temp_extended_basis3

                #print('extended basis 3 = \n{}'.format(temp_extended_basis3))
            
            # Filter out operators before computing the commutant matrix
            # but after expansion.
            if pre_filter_fun is not None:
                basis = pre_filter_fun(basis)
            
            # Skip com_matrix calculation if the algorithm has converged to a basis.
            if set(basis.op_strings) == set(previous_basis.op_strings):
                continue
            # Also if it gets into a cycle of switching between two bases.
            #elif previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
            #    continue
            else:
                previous_previous_basis = copy.deepcopy(previous_basis)
                previous_basis = copy.deepcopy(basis)
            
            print('  Expanded basis: {}'.format(len(basis)))

            # Find best operator in basis
            (s_constants, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
            com_matrix = _com_matrix(s_constants, H).real

            orig_com_matrix = com_matrix.copy()
            
            # Project against the given operators.
            if orth_ops is not None:
                # Create a projection matrix that projects
                # out of the space spanned by the given operators.
                proj_ops = ss.csc_matrix((len(basis),len(basis)), dtype=complex)
                for orth_op in orth_ops:
                    row_inds = []
                    col_inds = []
                    data     = []
                    for (coeff, os) in orth_op:
                        if os in basis:
                            data.append(coeff)
                            row_inds.append(basis.index(os))
                            col_inds.append(0)
                    orth_vec = ss.csc_matrix((data, (row_inds, col_inds)), shape=(len(basis),1), dtype=complex)

                    proj_ops += orth_vec.dot(orth_vec.H)

                large_number = 1e12
                com_matrix += large_number * proj_ops
            
            # Rescale C^\dagger C to D^{-1/2} C^\dagger C D^{-1/2}
            # for a given normalization of the OperatorStrings D.
            if norm_fun is not None:
                basis_norms = np.zeros(len(basis),dtype=complex)
                for ind_basis in range(len(basis)):
                    basis_norms[ind_basis] = norm_fun(basis[ind_basis])

                #print('basis_norms = {}'.format(basis_norms))

                D_sqrt     = ss.diags(np.sqrt(basis_norms), format='csc')
                D_inv_sqrt = ss.diags(1.0 / np.sqrt(basis_norms), format='csc')
                
                com_matrix = D_inv_sqrt.dot(com_matrix.dot(D_inv_sqrt))
                com_matrix = com_matrix.real
                
            if len(basis) < 2*max_num_evecs:
                (evals, evecs) = nla.eigh(com_matrix.toarray())
                num_vecs = np.minimum(len(evals), max_num_evecs)
                evals = evals[0:num_vecs]
                evecs = evecs[:,0:num_vecs]
            else:
                maxiter = 10*int(com_matrix.shape[0])*maxiter_scale
                (evals, evecs) = ssla.eigsh(com_matrix, k=max_num_evecs, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
                
                inds_sort = np.argsort(np.abs(evals))
                evals = evals[inds_sort]
                evecs = evecs[:, inds_sort]

            if eval_threshold is not None:
                inds_valid = np.where(evals < eval_threshold)[0]
                if len(inds_valid) > min_num_evecs:
                    evals = evals[inds_valid]
                    evecs = evecs[:, inds_valid]
                else:
                    evals = evals[0:min_num_evecs]
                    evecs = evecs[:, 0:min_num_evecs]

            # Rescale back to the original basis's normalization.
            if norm_fun is not None:
                com_matrix = orig_com_matrix
                evecs = np.dot(D_inv_sqrt.toarray(), evecs)

                #print('Original eigenvalues: {}'.format(evals))
                for ind_vec in range(int(evecs.shape[1])):
                    evecs[:,ind_vec] /= nla.norm(evecs[:,ind_vec])

                    vector = ss.csc_matrix(evecs[:,ind_vec].reshape(len(basis),1))
                    evals[ind_vec] = np.real((vector.H).dot(com_matrix).dot(vector)[0,0])
                #print('Rescaled eigenvalues: {}'.format(evals))
                
            if post_filter_fun is not None:
                (vector, com_norm) = post_filter_fun(evals, evecs, basis)
            else:
                vector   = evecs[:,0]
                com_norm = evals[0]
        
            operator = Operator(vector, copy.deepcopy(basis.op_strings))
            operators.append(copy.deepcopy(operator))
            com_norms.append(com_norm)

            if np.abs(com_norm) < best_com_norm:
                best_operator = copy.deepcopy(operator)
                best_com_norm = com_norm
        
            # Truncate basis
            basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size, filter_fun=trunc_filter_fun)
            print('  Truncated basis: {}'.format(len(basis)))

        direction *= -1
            
    return (best_operator, best_com_norm, operators, com_norms)

# TODO: document
# Idea: instead of expanding the basis by doing O -> [H,O] -> [H, [H,O]],
# expand by considering the largest term in [H,O]=\sum_b g_b S_b, say S_{b'},
# and finding which term in H=\sum_a J_a S_a can cancel it (by going through [S_a. S_{b'}] ordered by the magnitude of J_a).
#
def selected_ci_greedy_simple(initial_operator, H, num_steps, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0):

    H_labels = np.array([label for os in H._basis for (_, label) in os], dtype=int)
    max_label = np.max(H_labels)

    # The indices that sort the terms in the Hamiltonian
    # by the magnitude of their coefficient.
    inds_sorted = np.argsort(np.abs(H.coeffs))[::-1]
    
    # Keep track of all OperatorStrings considered
    # during the calculation.
    if explored_data is None:
        explored_basis = Basis()
        explored_extended_basis = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf

    previous_previous_basis = None
    previous_basis = copy.deepcopy(basis)
    for step in range(num_steps):
        if step == 0:
            operator = copy.deepcopy(initial_operator)
        else:
            operator = operators[-1]
        
        # O -> [H,O]
        (s_constants1, extended_basis1) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)

        l_matrix1    = _l_matrix(s_constants1, H)
        com_H_operator = l_matrix1.dot(ss.csc_matrix(operator.coeffs).T)

        # largest term of [H,O] is A; A -> largest term B in [H,A]
        inds_largest_termsA = np.argsort(np.abs(com_H_operator.toarray().flatten()))[::-1]
        for indA in inds_largest_termsA:
            A_basis1 = Basis()
            A_basis1 += extended_basis1[indA]
            (s_constants2, extended_basis2) = _explore(A_basis1, H, explored_basis, explored_extended_basis, explored_s_constants_data)

            l_matrix2 = _l_matrix(s_constants2, H)

            #A = ss.csc_matrix(([1.0], ([0], [0])), shape=(1,1), dtype=complex)
            com_H_A = l_matrix2.toarray()[:,0]
            #print('com_H_A = {}'.format(com_H_A))
        
            inds_largest_termsB = np.argsort(np.abs(com_H_A))[::-1]
            ind_largest_termB = None
            for indB in inds_largest_termsB:
                if extended_basis2[indB] not in basis:
                    ind_largest_termB = indB
                    break
            if ind_largest_termB is not None:
                break
            else:
                print('Failed to find a B, picking new A.')
                
        # Insert B into the basis
        basis += extended_basis2[ind_largest_termB]
        
        print('Basis = {}'.format(len(basis)))
            
        # Skip com_matrix calculation if the algorithm has converged to a basis.
        if set(basis.op_strings) == set(previous_basis.op_strings):
            continue
        # Also if it gets into a cycle of switching between two bases.
        #elif previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
        #    continue
        else:
            previous_previous_basis = copy.deepcopy(previous_basis)
            previous_basis = copy.deepcopy(basis)
            
        # Find best operator in basis
        (s_constants, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        com_matrix = _com_matrix(s_constants, H).real
        
        if len(basis) < 20:
            (evals, evecs) = nla.eigh(com_matrix.toarray())
        else:
            maxiter = 10*int(com_matrix.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(com_matrix, k=2, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
            
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]

        vector   = evecs[:,0]
        com_norm = evals[0]
        
        operator = Operator(vector, copy.deepcopy(basis.op_strings))
        operators.append(copy.deepcopy(operator))
        com_norms.append(com_norm)

        if np.abs(com_norm) < best_com_norm:
            best_operator = copy.deepcopy(operator)
            best_com_norm = com_norm
        
        # Truncate basis
        #basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size, filter_fun=trunc_filter_fun)
        #print('  Truncated basis: {}'.format(len(basis)))
            
    return (best_operator, best_com_norm, operators, com_norms)


# TODO: document, test
# Idea: Expand the basis by commuting with the largest terms in H
# and anticommuting with the largest terms in O.
def selected_ci_greedy(initial_operator, com_ops, num_steps, num_H_terms=10, num_O_terms=0, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0, verbose=True, orth_ops=None, transformations=None):

    if isinstance(com_ops, Operator):
        H = com_ops
        com_ops = [H]
    elif isinstance(com_ops, list) and isinstance(com_ops[0], Operator):
        H = com_ops[0]
    else:
        raise ValueError('Invalid com_ops of type: {}'.format(type(com_ops)))

    H_labels = np.array([label for os in H._basis for (_, label) in os], dtype=int)
    max_label = np.max(H_labels)

    # The indices that sort the terms in the Hamiltonian
    # by the magnitude of their coefficient.
    inds_sorted = np.argsort(np.abs(H.coeffs))[::-1]
    
    # Keep track of how OperatorStrings
    # commute with H during the calculation.
    if explored_data is None:
        explored_basis            = Basis()
        explored_extended_basis   = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data

    # Keep track of how OperatorStrings
    # anti-commute with O during the calculation.
    explored_anticommutations = dict()
    
    basis = Basis()
    basis += initial_operator._basis
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    best_operator = copy.deepcopy(initial_operator)
    best_com_norm = np.inf

    previous_previous_basis = None
    previous_basis = copy.deepcopy(basis)
    for step in range(num_steps):
        if step == 0:
            operator = copy.deepcopy(initial_operator)

        # ==== 1st step: [H,[H,O]] to generate new terms ====
        # O -> [H,O]
        (l_matrix1, extended_basis1) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        
        #l_matrix1    = _l_matrix(s_constants1, H)
        com_H_operator = l_matrix1.dot(ss.csc_matrix(operator.coeffs).T)

        # largest terms of [H,O] are A; A -> largest terms B in [H,A]
        inds_largest_termsA = np.argsort(np.abs(com_H_operator.toarray().flatten()))[::-1]
        num_lterms = np.minimum(len(inds_largest_termsA), num_H_terms)
        inds_largest_termsA = inds_largest_termsA[0:num_lterms]
        A_basis1 = Basis([extended_basis1[ind] for ind in inds_largest_termsA])

        for com_op in com_ops:
            (l_matrix2, extended_basis2) = _explore(A_basis1, com_op, explored_basis, explored_extended_basis, explored_s_constants_data)
        
            #l_matrix2 = _l_matrix(s_constants2, com_op)
        
            com_H_A = l_matrix2.dot(com_H_operator[inds_largest_termsA,0]).toarray().flatten()
            #print('com_H_A = {}'.format(com_H_A))
        
            num_lterms = np.minimum(len(com_H_A), num_H_terms)
            inds_largest_termsB = np.argsort(np.abs(com_H_A))[::-1]
            #inds_largest_termsB = inds_largest_termsB[0:num_lterms]
        
            # Insert B into the basis
            num_terms_added = 0
            for ind in inds_largest_termsB:
                os = extended_basis2[ind]
                if os not in basis:
                    basis += os
                    num_terms_added += 1
                if num_terms_added >= num_lterms:
                    break
            
        #basis += Basis([extended_basis2[ind] for ind in inds_largest_termsB])
        #basis += extended_basis2
        
        # ==== 2nd step: {O, O} to generate new terms ====
        num_lterms = np.minimum(len(operator.coeffs), num_O_terms)
        inds_largest_terms_O = np.argsort(np.abs(operator.coeffs))[::-1]
        inds_largest_terms_O = inds_largest_terms_O

        if num_lterms > 0:
            num_terms_added = 0
            remainder_terms = []
            for ind1 in inds_largest_terms_O:
                os1 = operator._basis[ind1]
                for ind2 in inds_largest_terms_O:
                    if ind2 == ind1:
                        continue
                    
                    os2 = operator._basis[ind2]
                
                    os_pair = (os1, os2)
                    if os_pair in explored_anticommutations:
                        (coeff3, os3) = explored_anticommutations[os_pair]
                    else:
                        (coeff3, os3) = anticommutator(os1, os2)
                        explored_anticommutations[os_pair] = (coeff3, os3)
                        
                    if np.abs(coeff3) > 1e-16 and os3 not in basis:
                        basis += os3
                        remainder_terms.append(os3)
                        num_terms_added += 1
                        #print('Added: {}'.format(os3))

                    if num_terms_added >= num_lterms:
                        break
                if num_terms_added >= num_lterms:
                    break

            num_terms_added = 0
            for ind1 in inds_largest_terms_O:
                os1 = operator._basis[ind1]
                for os2 in remainder_terms:
                    os_pair = (os1, os2)
                    if os_pair in explored_anticommutations:
                        (coeff3, os3) = explored_anticommutations[os_pair]
                    else:
                        (coeff3, os3) = anticommutator(os1, os2)
                        explored_anticommutations[os_pair] = (coeff3, os3)
                    
                    if np.abs(coeff3) > 1e-16 and os3 not in basis:
                        basis += os3
                        num_terms_added += 1
                        #print('Added:: {}'.format(os3))

                    if num_terms_added >= num_lterms:
                        break
                if num_terms_added >= num_lterms:
                    break

        # ==== 3rd step: Use transformations to generate new terms ====
        if transformations is not None:
            # Apply all of the transformations to expand the basis
            # until no new operator strings can be added.
            ind_os = 0
            while ind_os < len(basis):
                os = basis[ind_os]
                for transformation in transformations:
                    new_op = transformation.apply(os)
                    for (new_coeff, new_os) in new_op:
                        if np.abs(new_coeff) > 1e-16 and new_os not in basis:
                            basis += new_os

                ind_os += 1
        
        if verbose:
            print('Step {}: Basis = {}'.format(step, len(basis)))
            
        # Skip com_matrix calculation if the algorithm has converged to a basis.
        if step > 0 and set(basis.op_strings) == set(previous_basis.op_strings):
            if num_H_terms >= max_basis_size:
                break
            else:
                num_H_terms *= 2
                if verbose:
                    print(' num_H_terms = {}'.format(num_H_terms))
        # Also if it gets into a cycle of switching between two bases.
        elif step > 0 and previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
            if num_H_terms >= max_basis_size:
                break
            else:
                num_H_terms *= 2
                if verbose:
                    print(' num_H_terms = {}'.format(num_H_terms))
        
        previous_previous_basis = copy.deepcopy(previous_basis)
        previous_basis = copy.deepcopy(basis)
            
        # Find best operator in basis
        com_matrix   = ss.csc_matrix((len(basis), len(basis)), dtype=float)
        com_matrix_H = ss.csc_matrix((len(basis), len(basis)), dtype=float)
        for ind_com_op in range(len(com_ops)):
            com_op = com_ops[ind_com_op]
            (l_matrix, _) = _explore(basis, com_op, explored_basis, explored_extended_basis, explored_s_constants_data)
            cdagc_real = ((l_matrix.H).dot(l_matrix)).real
            com_matrix += cdagc_real
            if ind_com_op == 0:
                com_matrix_H += cdagc_real

        # Orthogonalize against the given operators.
        if orth_ops is not None:
            com_matrix = _orthogonalize_com_matrix(com_matrix, basis, orth_ops)

        if transformations is not None:
            com_matrix = _symmetrize_com_matrix(com_matrix, basis, transformations)
            
        if len(basis) < 20:
            (evals, evecs) = nla.eigh(com_matrix.toarray())
        else:
            maxiter = 10*int(com_matrix.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(com_matrix, k=1, sigma=-1e-8, which='LM', maxiter=maxiter, tol=tol)

            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]

        # Recompute the commutator norms for just
        # commuting with the Hamiltonian.
        if len(com_ops) > 1 or orth_ops is not None:
            for ind_vec in range(int(evecs.shape[1])):
                vec = ss.csc_matrix(evecs[:, ind_vec].reshape((len(basis), 1)), dtype=complex)
                evals[ind_vec] = (vec.H).dot(com_matrix_H.dot(vec))[0,0].real

            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]

        vector   = evecs[:,0]
        com_norm = evals[0]
        
        operator = Operator(vector, copy.deepcopy(basis.op_strings))
        operators.append(copy.deepcopy(operator))
        com_norms.append(com_norm)

        if np.abs(com_norm) < best_com_norm:
            best_operator = copy.deepcopy(operator)
            best_com_norm = com_norm
        
        # Truncate basis
        basis = _truncate_basis(basis, evecs[:,0], threshold=threshold, max_basis_size=max_basis_size)
        if verbose:
            print('  Truncated basis: {}'.format(len(basis)))

        # Truncate the operator into the new basis
        new_coeffs = [vector[basis.index(os)] for os in basis]
        operator = Operator(new_coeffs, basis.op_strings)
        
    return (best_operator, best_com_norm, operators, com_norms)

# TODO: document, test
# Idea: Try to iteratively find many operators that commute with H.
def selected_ci_greedy_many_ops(initial_operators, H, num_steps, num_H_terms=10, num_O_terms=0, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0, verbose=True):

    if not (isinstance(max_basis_size, list) or isinstance(max_basis_size, np.ndarray)):
        max_basis_sizes = max_basis_size * np.ones(num_steps, dtype=int)
    else:
        max_basis_sizes = np.copy(max_basis_size)

    if not (isinstance(num_H_terms, list) or isinstance(num_H_terms, np.ndarray)):
        num_H_terms = num_H_terms * np.ones(num_steps, dtype=int)
        
    # Keep track of how OperatorStrings
    # commute with H during the calculation.
    if explored_data is None:
        explored_basis            = Basis()
        explored_extended_basis   = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data

    result_operators = []
    result_com_norms = []
        
    current_operators = copy.deepcopy(initial_operators)
    num_operators = len(current_operators)
    
    for step in range(num_steps):
        if verbose:
            print('==== STEP {}/{} ===='.format(step+1, num_steps))

        step_operators = []
        step_com_norms = []
        for ind_op in range(num_operators):
            if verbose:
                print('  (( OPERATOR {}/{} ))  '.format(ind_op+1, num_operators))

            # Orthogonalize against the other operators.
            orth_ops = [current_operators[ind] for ind in range(num_operators) if ind != ind_op]
                
            current_op = current_operators[ind_op]

            # Truncate the operator.
            if step > 0:
                current_op = _truncate_operator(current_op, threshold=threshold, max_basis_size=max_basis_size)

            com_ops = H
            #com_ops = [H] + [current_operators[ind_op2] for ind_op2 in range(num_operators) if ind_op2 != ind_op]
            
            (_, _, ops, c_norms) = selected_ci_greedy(current_op, com_ops, 1, num_H_terms=num_H_terms[step], num_O_terms=num_O_terms, threshold=threshold, max_basis_size=max_basis_sizes[step], explored_data=explored_data, maxiter_scale=maxiter_scale, tol=tol, verbose=verbose, orth_ops=orth_ops)

            current_operators[ind_op] = copy.deepcopy(ops[-1])
            
            step_operators.append(ops[-1])
            step_com_norms.append(c_norms[-1])
        
        result_operators.append(step_operators)
        result_com_norms.append(step_com_norms)
        
    return (result_operators, result_com_norms)


# TODO: document, test
# Idea: Expand the basis by commuting with the largest terms in H
# and anticommuting with the largest terms in O.
def selected_ci_greedy_many_ops2(initial_operators, H, num_steps, num_H_terms=10, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0, verbose=True, orth_ops=None):

    num_ops = len(initial_operators)

    if not (isinstance(num_H_terms, list) or isinstance(num_H_terms, np.ndarray)):
        num_H_terms = num_H_terms * np.ones(num_steps, dtype=int)

    H_labels = np.array([label for os in H._basis for (_, label) in os], dtype=int)
    max_label = np.max(H_labels)

    # The indices that sort the terms in the Hamiltonian
    # by the magnitude of their coefficient.
    inds_sorted = np.argsort(np.abs(H.coeffs))[::-1]
    
    # Keep track of how OperatorStrings
    # commute with H during the calculation.
    if explored_data is None:
        explored_basis            = Basis()
        explored_extended_basis   = Basis()
        explored_s_constants_data = dict()
    else:
        (explored_basis, explored_extended_basis, explored_s_constants_data) = explored_data

    basis = Basis()
    for op in initial_operators:
        basis += op._basis

    # The current operators found at the most recent
    # step of Selected CI.
    current_operators = []
    for op in initial_operators:
        new_coeffs = np.zeros(len(basis), dtype=complex)
        for (coeff, os) in op:
            new_coeffs[basis.index(os)] = coeff
        new_op = Operator(new_coeffs, basis.op_strings)
        current_operators.append(new_op)
    
    # Keep a record of the smallest commutator norms
    # observed during the iterations of Selected CI.
    operators = []
    com_norms = []

    previous_previous_basis = None
    previous_basis = copy.deepcopy(basis)
    for step in range(num_steps):
        new_os_to_add = Basis()
        for ind_op in range(num_ops):
            operator = current_operators[ind_op]
            #operator_vector = (operator.coeffs).reshape((len(basis),1))
            
            # ==== Use [H,[H,O]] to generate new terms ====
            # O -> [H,O]
            (l_matrix1, extended_basis1) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
            
            com_H_operator = l_matrix1.dot(ss.csc_matrix(operator.coeffs).T)

            # largest terms of [H,O] are A; A -> largest terms B in [H,A]
            inds_largest_termsA = np.argsort(np.abs(com_H_operator.toarray().flatten()))[::-1]
            num_lterms = np.minimum(len(inds_largest_termsA), num_H_terms[step])
            inds_largest_termsA = inds_largest_termsA[0:num_lterms]
            A_basis1 = Basis([extended_basis1[ind] for ind in inds_largest_termsA])

            (l_matrix2, extended_basis2) = _explore(A_basis1, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        
            com_H_A = l_matrix2.dot(com_H_operator[inds_largest_termsA,0]).toarray().flatten()
        
            num_lterms = np.minimum(len(com_H_A), num_H_terms[step])
            inds_largest_termsB = np.argsort(np.abs(com_H_A))[::-1]
            
            # Insert B into the basis
            num_terms_added = 0
            for ind in inds_largest_termsB:
                os = extended_basis2[ind]
                if os not in basis and os not in new_os_to_add:
                    new_os_to_add += os
                    num_terms_added += 1
                if num_terms_added >= num_lterms:
                    break
        basis += new_os_to_add
        
        if verbose:
            print('Step {}: Basis = {}'.format(step, len(basis)))
            
        # Skip com_matrix calculation if the algorithm has converged to a basis.
        if step > 0 and set(basis.op_strings) == set(previous_basis.op_strings):
            break
        # Also if it gets into a cycle of switching between two bases.
        elif step > 0 and previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
            break
        
        previous_previous_basis = copy.deepcopy(previous_basis)
        previous_basis = copy.deepcopy(basis)
        
        # Find best operators in basis
        (l_matrix, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        com_matrix = ((l_matrix.H).dot(l_matrix)).real
            
        # Orthogonalize against the given operators.
        if orth_ops is not None:
            com_matrix = _orthogonalize_com_matrix(com_matrix, basis, orth_ops)
            
        if len(basis) < 20 and num_ops < 20:
            (evals, evecs) = nla.eigh(com_matrix.toarray())
        else:
            maxiter = 10*int(com_matrix.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(com_matrix, k=num_ops, sigma=-1e-8, which='LM', maxiter=maxiter, tol=tol)
            
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]

        # Recompute the commutator norms for just
        # commuting with the Hamiltonian.
        if orth_ops is not None:
            for ind_vec in range(int(evecs.shape[1])):
                vec = ss.csc_matrix(evecs[:, ind_vec].reshape((len(basis), 1)), dtype=complex)
                evals[ind_vec] = (vec.H).dot(com_matrix.dot(vec))[0,0].real
                
            inds_sort = np.argsort(np.abs(evals))
            evals = evals[inds_sort]
            evecs = evecs[:, inds_sort]
            
        vector   = evecs[:,0]
        com_norm = evals[0]
        
        best_operators = [Operator(evecs[:,ind_vec], basis.op_strings) for ind_vec in range(num_ops)]
        best_com_norms = evals[0:num_ops]
        
        operators.append(best_operators)
        com_norms.append(best_com_norms)
        
        # Truncate basis
        random_sum_vec = np.zeros(len(basis))
        for ind_vec in range(num_ops):
            random_sum_vec += (2.0*np.random.rand()-1.0) * evecs[:,ind_vec]
        random_sum_vec /= nla.norm(random_sum_vec)    
        basis = _truncate_basis(basis, random_sum_vec, threshold=threshold, max_basis_size=max_basis_size)
        if verbose:
            print('  Truncated basis: {}'.format(len(basis)))

        # Truncate the operator into the new basis
        current_operators = []
        for ind_vec in range(num_ops):
            new_coeffs = [evecs[basis.index(os),ind_vec] for os in basis]
            new_op     = Operator(new_coeffs, basis.op_strings)
            current_operators.append(new_op)
        
    return (operators, com_norms)
