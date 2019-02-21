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

def _com_matrix(s_constants, operator):
    # Computes C_O, where C_O is the
    # commutant matrix of operator O,
    # from the structure constants and O.
    
    dim_extended_basis = int(s_constants[0].shape[0])
    dim_basis          = int(s_constants[0].shape[1])

    commutant_matrix = ss.csc_matrix((dim_extended_basis, dim_basis), dtype=complex)
    
    for ind_os in range(len(operator._basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os]
        
    return commutant_matrix

def _cdagc(s_constants, operator):
    # Computes C_O^\dagger C_O, where C_O is the
    # commutant matrix of operator O,
    # from the structure constants and O.

    commutant_matrix = _com_matrix(s_constants, operator)
    CDagC = (commutant_matrix.H).dot(commutant_matrix)
        
    return CDagC

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
        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is not None and len(set(H._basis[ind_os].orbital_labels).intersection(allowed_labels)) == 0:
            continue # TODO: check if I am doing this right
        
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
        row_inds = []
        col_inds = []
        data     = []

        # If only considering a part of the Hamiltonian,
        # with OperatorStrings with the given allowed_labels, then
        # only construct the extended basis for that part.
        if allowed_labels is None or len(set(H._basis[ind_os].orbital_labels).intersection(allowed_labels)) != 0:
            (inds_explored_eb, inds_explored_b, explored_data) = explored_s_constants_data[ind_os]
            
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
        com_matrix = commutant_matrix(basis, H)
        CDagC = ((com_matrix.H).dot(com_matrix)).real

        if len(basis) < 20:
            (evals, evecs) = nla.eigh(CDagC.toarray())
        else:
            (evals, evecs) = ssla.eigsh(CDagC, k=8, which='SM')
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
        explored_s_constants_data = [([],[],[])]*len(H)
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
        CDagC = _cdagc(s_constants, H).real
        
        if len(basis) < 2*max_num_evecs:
            (evals, evecs) = nla.eigh(CDagC.toarray())
            num_vecs = np.minimum(len(evals), max_num_evecs)
            evals = evals[0:num_vecs]
            evecs = evecs[:,0:num_vecs]
        else:
            maxiter = 10*int(CDagC.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(CDagC, k=max_num_evecs, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
                    
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
        explored_s_constants_data = [([],[],[])]*len(H)
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
            
            # Skip CDagC calculation if the algorithm has converged to a basis.
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
            CDagC = _cdagc(s_constants, H).real

            orig_CDagC = CDagC.copy()
            
            # Project against the given operators.
            if orth_ops is not None:
                # Create a projection matrix that projects
                # out of the space spanned by the given operators.
                proj_ops = ss.csr_matrix((len(basis),len(basis)), dtype=complex)
                for orth_op in orth_ops:
                    row_inds = []
                    col_inds = []
                    data     = []
                    for (coeff, os) in orth_op:
                        if os in basis:
                            data.append(coeff)
                            row_inds.append(basis.index(os))
                            col_inds.append(0)
                    orth_vec = ss.csr_matrix((data, (row_inds, col_inds)), shape=(len(basis),1), dtype=complex)

                    proj_ops += orth_vec.dot(orth_vec.H)

                large_number = 1e12
                CDagC += large_number * proj_ops
            
            # Rescale C^\dagger C to D^{-1/2} C^\dagger C D^{-1/2}
            # for a given normalization of the OperatorStrings D.
            if norm_fun is not None:
                basis_norms = np.zeros(len(basis),dtype=complex)
                for ind_basis in range(len(basis)):
                    basis_norms[ind_basis] = norm_fun(basis[ind_basis])

                #print('basis_norms = {}'.format(basis_norms))

                D_sqrt     = ss.diags(np.sqrt(basis_norms), format='csr')
                D_inv_sqrt = ss.diags(1.0 / np.sqrt(basis_norms), format='csr')
                
                CDagC = D_inv_sqrt.dot(CDagC.dot(D_inv_sqrt))
                CDagC = CDagC.real
                
            if len(basis) < 2*max_num_evecs:
                (evals, evecs) = nla.eigh(CDagC.toarray())
                num_vecs = np.minimum(len(evals), max_num_evecs)
                evals = evals[0:num_vecs]
                evecs = evecs[:,0:num_vecs]
            else:
                maxiter = 10*int(CDagC.shape[0])*maxiter_scale
                (evals, evecs) = ssla.eigsh(CDagC, k=max_num_evecs, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
                
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
                CDagC = orig_CDagC
                evecs = np.dot(D_inv_sqrt.toarray(), evecs)

                #print('Original eigenvalues: {}'.format(evals))
                for ind_vec in range(int(evecs.shape[1])):
                    evecs[:,ind_vec] /= nla.norm(evecs[:,ind_vec])

                    vector = ss.csr_matrix(evecs[:,ind_vec].reshape(len(basis),1))
                    evals[ind_vec] = np.real((vector.H).dot(CDagC).dot(vector)[0,0])
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
# expand by considering the largest term in [H,O]=\sum_b g_b h_b, say h_{b'},
# and finding which term in H=\sum_a J_a h_a can cancel it (by going through [h_a. h_{b'}] ordered by the magnitude of J_a).
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
        explored_s_constants_data = [([],[],[])]*len(H)
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

        com_matrix1    = _com_matrix(s_constants1, H)
        com_H_operator = com_matrix1.dot(ss.csr_matrix(operator.coeffs).T)

        # largest term of [H,O] is A; A -> largest term B in [H,A]
        inds_largest_termsA = np.argsort(np.abs(com_H_operator.toarray().flatten()))[::-1]
        for indA in inds_largest_termsA:
            A_basis1 = Basis()
            A_basis1 += extended_basis1[indA]
            (s_constants2, extended_basis2) = _explore(A_basis1, H, explored_basis, explored_extended_basis, explored_s_constants_data)

            com_matrix2 = _com_matrix(s_constants2, H)

            #A = ss.csr_matrix(([1.0], ([0], [0])), shape=(1,1), dtype=complex)
            com_H_A = com_matrix2.toarray()[:,0]
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
            
        # Skip CDagC calculation if the algorithm has converged to a basis.
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
        CDagC = _cdagc(s_constants, H).real
        
        if len(basis) < 20:
            (evals, evecs) = nla.eigh(CDagC.toarray())
        else:
            maxiter = 10*int(CDagC.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(CDagC, k=2, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
            
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
def selected_ci_greedy(initial_operator, H, num_steps, num_H_terms=10, num_O_terms=0, threshold=1e-6, max_basis_size=100, explored_data=None, maxiter_scale=1, tol=0.0):

    H_labels = np.array([label for os in H._basis for (_, label) in os], dtype=int)
    max_label = np.max(H_labels)

    # The indices that sort the terms in the Hamiltonian
    # by the magnitude of their coefficient.
    inds_sorted = np.argsort(np.abs(H.coeffs))[::-1]
    
    # Keep track of how OperatorStrings
    # commute with H during the calculation.
    if explored_data is None:
        explored_basis = Basis()
        explored_extended_basis = Basis()
        explored_s_constants_data = [([],[],[])]*len(H)
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
        (s_constants1, extended_basis1) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        
        com_matrix1    = _com_matrix(s_constants1, H)
        com_H_operator = com_matrix1.dot(ss.csr_matrix(operator.coeffs).T)

        # largest terms of [H,O] are A; A -> largest terms B in [H,A]
        inds_largest_termsA = np.argsort(np.abs(com_H_operator.toarray().flatten()))[::-1]
        num_lterms = np.minimum(len(inds_largest_termsA), num_H_terms)
        inds_largest_termsA = inds_largest_termsA[0:num_lterms]
        A_basis1 = Basis([extended_basis1[ind] for ind in inds_largest_termsA])

        (s_constants2, extended_basis2) = _explore(A_basis1, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        
        com_matrix2 = _com_matrix(s_constants2, H)
        
        com_H_A = com_matrix2.dot(com_H_operator[inds_largest_termsA,0]).toarray().flatten()
        #print('com_H_A = {}'.format(com_H_A))
        
        num_lterms = np.minimum(len(com_H_A), num_H_terms)
        inds_largest_termsB = np.argsort(np.abs(com_H_A))[::-1]
        inds_largest_termsB = inds_largest_termsB[0:num_lterms]
        
        # Insert B into the basis
        basis += Basis([extended_basis2[ind] for ind in inds_largest_termsB])

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

        print('Step {}: Basis = {}'.format(step, len(basis)))
            
        # Skip CDagC calculation if the algorithm has converged to a basis.
        if step > 0 and set(basis.op_strings) == set(previous_basis.op_strings):
            if num_H_terms >= max_basis_size:
                break
            else:
                num_H_terms *= 2
                print(' num_H_terms = {}'.format(num_H_terms))
        # Also if it gets into a cycle of switching between two bases.
        elif step > 0 and previous_previous_basis is not None and set(basis.op_strings) == set(previous_previous_basis.op_strings):
            if num_H_terms >= max_basis_size:
                break
            else:
                num_H_terms *= 2
                print(' num_H_terms = {}'.format(num_H_terms))
        
        previous_previous_basis = copy.deepcopy(previous_basis)
        previous_basis = copy.deepcopy(basis)
            
        # Find best operator in basis
        (s_constants, _) = _explore(basis, H, explored_basis, explored_extended_basis, explored_s_constants_data)
        CDagC = _cdagc(s_constants, H).real
        
        if len(basis) < 20:
            (evals, evecs) = nla.eigh(CDagC.toarray())
        else:
            maxiter = 10*int(CDagC.shape[0])*maxiter_scale
            (evals, evecs) = ssla.eigsh(CDagC, k=2, sigma=-1e-6, which='LM', maxiter=maxiter, tol=tol)
            
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
        print('  Truncated basis: {}'.format(len(basis)))

        # Truncate the operator into the new basis
        new_coeffs = [vector[basis.index(os)] for os in basis]
        operator = Operator(new_coeffs, basis.op_strings)
        
    return (best_operator, best_com_norm, operators, com_norms)
