#!/usr/bin/env python
import numpy as np
import scipy.sparse as ss

import tools
from .operatorstring import OperatorString

def symmetry_matrix(basis, transformation, tol=1e-12):
    dim_basis = len(basis)

    (rule, info) = transformation

    row_inds = []
    col_inds = []
    data     = []
    for indA in range(len(basis)):
        op_string_A = basis[indA]

        (coeffs_B, op_strings_B) = rule(op_string_A, info)

        for (coeff_B, op_string_B) in zip(coeffs_B, op_strings_B):
            
            # Ignore operators not currently in the basis.
            # If there are operators outside of the basis, this
            # will make the symmetry matrix non-unitary.
            if op_string_B not in basis:
                continue
            
            indB = basis.index(op_string_B)

            if np.abs(coeff_B) > tol:
                row_inds.append(indA)
                col_inds.append(indB)
                data.append(coeff_B)

    symmetry_matrix = ss.csr_matrix((data, (row_inds, col_inds)), dtype=complex, shape=(dim_basis, dim_basis))
            
    return symmetry_matrix

def _permutation_rule(op_string_A, info):
    """Transformation rule to perform a permutation of orbital labels.
    Transformation for spins:
        \sigma^a_i -> \sum_j U_{ji} \sigma^a_j 
    Transformation for (spinless) fermions:
        c_i -> \sum_j U_{ji} c_j
        a_i -> \sum_j U_{ji} a_j, b_i -> \sum_j U_{ji} b_j, d_i -> \sum_j U_{ji} d_j
    where U_{ji} is a permutation matrix with a single 1 on each row/column.
    """
    if not (op_type == 'Pauli' or op_type == 'Majorana'):
        raise ValueError('Invalid op_type: {}'.format(op_type))

    # The information is a function that specifies how labels map into other labels.
    relabeler = info
    
    coeffB = 1.0
    
    ops    = op_string_A.orbital_operators
    labels = op_string_A.orbital_labels
    
    new_labels_unsorted = [relabeler(l) for l in labels]
    inds_sorted         = np.argsort(new_labels_unsorted)

    new_ops     = [ops[ind] for ind in inds_sorted]
    new_labels  = [new_labels_unsorted[ind] for ind in inds_sorted]

    op_string_B = OperatorString(new_ops, new_labels, op_string_A.op_type)

    if op_type == 'Majorana':
        # TODO: test if this logic is correct
        inds_ab_sorted = [ind for ind in inds_sorted if new_ops[ind] in ['A','B']]
        
        # Make sure to compute the sign acquired from reordering
        # A and B operators in the Majorana strings.
        (_, sign) = tools.sort_sign(inds_ab_sorted)
        
        # Also, account for the prefactors in front of the Majorana
        # string operators before and after the transformation.
        coeffB *= np.real(op_string_B.prefactor/op_string_A.prefactor * sign)     

    return ([coeffB], [op_string_B])

def _time_reversal_rule(op_string_A, info):
    """Transformation rule for time-reversal symmetry T.
    Transformation for spins:
        \sigma^a_i -> -\sigma^a_i (means T \sigma^a_i T^{-1} = -\sigma^a_i)
    Transformation for (spinless) fermions:
        c_i -> +/- c_i, c_i^\dagger -> +/- c_i^\dagger, i -> -i (imaginary number)
        a_i -> +/- a_i, b_i -> -/+ b_i, d_i -> d_i, i -> -i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if not (op_type == 'Pauli' or op_type == 'Majorana'):
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # Extra information specifies whether time-reversal transformation
    # is of type T^2 = 1 or T^2 = (-1)^N
    extra_sign = float(info)
    
    coeffB      = 1.0
    op_string_B = op_string_A

    if op_type == 'Majorana':        
        numABs = 0
        numBs  = 0
        numDs  = 0
        for op in op_string_A.orbital_operators:
            if op == 'A':
                numABs += 1
            elif op == 'B':
                numBs  += 1
                numABs += 1
            elif op == 'D':
                numDs  += 1
            else:
                raise ValueError('Invalid op name: {}'.format(op))

        # A minus sign occurs for every b_i majorana operator.
        if numBs % 2 == 1:
            coeffB *= -1.0
        # A minus sign occurs if there is an imaginary number in front of the Majorana string.
        if numABs*(numABs-1)//2 % 2 == 1:
            coeffB *= -1.0
        # A minus sign occurs if T^2 = (-1)^N and there are an odd number of a_i,b_i operators.
        if numABs % 2 == 1:
            coeffB *= extra_sign
    else:
        # A minus sign occurs for every Pauli matrix in the Pauli string.
        num_ops  = len(op_string_B.orbital_operators)//2
        if num_ops % 2 == 1:
            coeffB *= -1.0
            
        # TODO: Implement T^2=(-1)^N symmetry for spin-1/2's if it makes sense.
        if np.abs(extra_sign - 1.0) > tol:
            raise NotImplementedError('Unsupported for spin-1/2 operators.')

    return ([coeffB], [op_string_B])

def _particle_hole_rule(op_string_A, info):
    """Transformation rule for particle hole symmetry C.
    Transformation for (spinless) fermions:
        c_i -> +/- c_i^\dagger, c_i^\dagger -> +/- c_i, i -> i (imaginary number)
        a_i -> +/- a_i, b_i -> -/+ b_i, d_i -> - d_i, i -> i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if op_type != 'Majorana':
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # Extra information specifies whether particle-hole transformation
    # is of type C^2 = 1 or C^2 = (-1)^N
    extra_sign = float(info)
    
    coeffB      = 1.0
    op_string_B = op_string_A

    numABs = 0
    numBs  = 0
    numDs  = 0
    for op in op_string_A.orbital_operators:
        if op == 'A':
            numABs += 1
        elif op == 'B':
            numBs  += 1
            numABs += 1
        elif op == 'D':
            numDs  += 1
        else:
            raise ValueError('Invalid op name: {}'.format(op))

    # Every b_i operator contributes a minus sign.
    if numBs % 2 == 1:
        coeffB *= -1.0
        
    # Every d_i operator contributes a minus sign.
    if numDs % 2 == 1:
        coeffB *= -1.0
        
    # A minus sign occurs if C^2 = (-1)^N and there are an odd number of a_i,b_i operators.
    if numABs % 2 == 1:
        coeffB *= extra_sign
            
    return ([coeffB], [op_string_B])

def _product_rule(op_string_A, info, tol=1e-12):
    """Transformation rule for an operator 
    that is a product of two operators: U = D*C.
    U H U^{-1} = D C H C^{-1} D^{-1}.
    """
    
    # The information contains the rules for the C and D operators.
    (ruleC, infoC, ruleD, infoD) = info

    (coeffs_C, op_strings_C) = ruleC(op_string_A, infoC)

    coeffs_B     = []
    op_strings_B = []
    for (coeffC, op_string_C) in zip(coeffs_C, op_strings_C):
        (coeffs_D, op_strings_D) = ruleD(op_string_C, infoD)

        for (coeffD, op_string_D) in zip(coeffs_D, op_strings_D):
            coeffB      = coeffC * coeffD
            op_string_B = op_string_D
            if np.abs(coeffB) > tol:
                coeffs_B.append(coeffB)
                op_strings_B.append(op_string_B)

    return (coeffs_B, op_strings_B)

def time_reversal(sign=1.0):
    info = sign
    return (_time_reversal_rule, info)

def particle_hole(sign=1.0):
    info = sign
    return (_particle_hole_rule, info)

def product(transformC, transformD):
    (ruleC, infoC) = transformC
    (ruleD, infoD) = transformD
    info_product = (ruleC, infoC, ruleD, infoD)

    return (_product_rule, info_product)
