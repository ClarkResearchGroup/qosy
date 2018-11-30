#!/usr/bin/env python
import numpy as np
import scipy.sparse as ss
import tools

from .operatorstring import OperatorString
from .basis import Basis, Operator
from .tools import sort_sign

def _operation(op_string_A, op_string_B, operation_mode='commutator', tol=1e-12):
    op_type = op_string_A.op_type
    if op_type == 'Pauli':
        (op1,op2,op3) = ('X', 'Y', 'Z')
    elif op_type == 'Majorana':
        (op1,op2,op3) = ('A', 'B', 'D')
    else:
        raise NotImplementedError('Unsupported commutator between operator strings of type: {}'.format(op_type))
    
    coeff = 1.0
    
    possible_labels_C = list(set(op_string_A.orbital_labels).union(op_string_B.orbital_labels))
    possible_labels_C = np.sort(possible_labels_C)

    if op_type == 'Majorana':
        coeff1  = op_string_A.prefactor
        coeff1 *= op_string_B.prefactor
        
        coeff2 = coeff1
        
        labels_ab_A_times_B = np.concatenate((op_string_A._labels_ab_operators, op_string_B._labels_ab_operators))
        labels_ab_B_times_A = np.concatenate((op_string_B._labels_ab_operators, op_string_A._labels_ab_operators))
        
        (_, sign1) = tools.sort_sign(labels_ab_A_times_B)
        (_, sign2) = tools.sort_sign(labels_ab_B_times_A)

        coeff1 *= sign1
        coeff2 *= sign2
        
    opsC    = []
    labelsC = []
    
    num_non_trivial_differences = 0
    for label in possible_labels_C:
        try:
            indA = op_string_A._indices_orbital_labels[label]
        except KeyError:
            indA = -1

        try:
            indB = op_string_B._indices_orbital_labels[label]
        except KeyError:
            indB = -1
                
        if indA != -1 and indB == -1:
            labelsC.append(label)
            opsC.append(op_string_A.orbital_operators[indA])
        elif indB != -1 and indA == -1:
            labelsC.append(label)
            opsC.append(op_string_B.orbital_operators[indB])
        else:
            opA = op_string_A.orbital_operators[indA]
            opB = op_string_B.orbital_operators[indB]

            if opA != opB:
                num_non_trivial_differences += 1
                labelsC.append(label)
                
                if opA == op1 and opB == op2:
                    coeff *= 1.0j
                    opsC.append(op3)
                elif opA == op2 and opB == op1:
                    coeff *= -1.0j
                    opsC.append(op3)
                elif opA == op1 and opB == op3:
                    coeff *= -1.0j
                    opsC.append(op2)
                elif opA == op3 and opB == op1:
                    coeff *= 1.0j
                    opsC.append(op2)
                elif opA == op2 and opB == op3:
                    coeff *= 1.0j
                    opsC.append(op1)
                elif opA == op3 and opB == op2:
                    coeff *= -1.0j
                    opsC.append(op1)

    if op_type == 'Pauli':
        # If an odd number of Pauli matrix pairs need to be multiplied, then the
        # two pauli strings from the commutator are imaginary and add together.
        if operation_mode=='product' or (num_non_trivial_differences % 2 == 1 and operation_mode == 'commutator') or (num_non_trivial_differences % 2 == 0 and operation_mode == 'anticommutator'):
            coeff *= 1.0 
        # Otherwise, they are real and cancel.
        else:
            coeff = 0.0
    elif op_type == 'Majorana':
        r = num_non_trivial_differences
        
        coeff1 *= coeff
        coeff2 *= np.conj(coeff) 

        if operation_mode=='product' or (np.abs(coeff1 - coeff2) > tol and operation_mode == 'commutator') or (np.abs(coeff1 - coeff2) < tol and operation_mode == 'anticommutator'):
            coeff = coeff1
        else:
            coeff = 0.0
        
    op_string_C = OperatorString(opsC, labelsC, op_type)
    
    coeff /= op_string_C.prefactor
    
    if operation_mode != 'product':
        coeff *= 2.0
    
    return (coeff, op_string_C)

def product(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, operation_mode='product')

def commutator(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, operation_mode='commutator')

def anticommutator(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, operation_mode='anticommutator')

def structure_constants(basisA, basisB, operation_mode='commutator', tol=1e-12):
    basisC = Basis()
    
    matrix_data = []
    for os_B in basisB:
        inds_os_C = []
        inds_os_A = []
        data      = []
        for ind_os_A in range(len(basisA)):
            os_A = basisA[ind_os_A]
            
            (coeff, os_C) = _operation(os_A, os_B, operation_mode=operation_mode)

            if np.abs(coeff) > tol:
                basisC += os_C
                ind_os_C = basisC.index(os_C)
                
                inds_os_C.append(ind_os_C)
                inds_os_A.append(ind_os_A)
                data.append(coeff)
                
        matrix_data.append((inds_os_C, inds_os_A, data))
        
    result = []
    for (inds_os_C, inds_os_A, data) in matrix_data:
        print(inds_os_C, inds_os_A, data)
        s_constants_B = ss.csc_matrix((data, (inds_os_C, inds_os_A)), dtype=complex, shape=(len(basisC), len(basisA)))
        result.append(s_constants_B)
    
    return (result, basisC)

def killing_form(basis):
    (s_constants, _) = structure_constants(basis, basis)
    
    # TODO

def commutant_matrix(basis, operator, operation_mode='commutator'):
    (s_constants, extended_basis) = structure_constants(basis, operator._basis, operation_mode=operation_mode)

    commutant_matrix = ss.csc_matrix(dtype=complex, shape=(len(basis), len(extended_basis)))

    for ind_os in range(len(operator_basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os]

    return commutant_matrix
