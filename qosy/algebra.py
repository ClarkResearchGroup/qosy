#!/usr/bin/env python
import numpy as np
from .operator import OperatorString
from .tools import sort_sign

def _operation(op_string_A, op_string_B, mode='commutator', tol=1e-12):
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
        
        labels_ab_A_times_B = op_string_A._labels_ab_operators + op_string_B._labels_ab_operators
        labels_ab_B_times_A = op_string_B._labels_ab_operators + op_string_A._labels_ab_operators
        
        (_, sign1) = sort_sign(labels_ab_A_times_B)
        (_, sign2) = sort_sign(labels_ab_B_times_A)

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
        if mode=='product' or (num_non_trivial_differences % 2 == 1 and mode == 'commutator') or (num_non_trivial_differences % 2 == 0 and mode == 'anticommutator'):
            coeff *= 1.0 
        # Otherwise, they are real and cancel.
        else:
            coeff = 0.0
    elif op_type == 'Majorana':
        r = num_non_trivial_differences
        
        coeff1 *= coeff
        coeff2 *= np.conj(coeff) 

        if mode=='product' or (np.abs(coeff1 - coeff2) > tol and mode == 'commutator') or (np.abs(coeff1 - coeff2) < tol and mode == 'anticommutator'):
            coeff = coeff1
        else:
            coeff = 0.0
        
    op_string_C = OperatorString(opsC, labelsC, op_type)
    
    coeff /= op_string_C.prefactor
    
    if mode != 'product':
        coeff *= 2.0
    
    return (coeff, op_string_C)

def product(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, mode='product')

def commutator(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, mode='commutator')

def anticommutator(op_string_A, op_string_B):
    return _operation(op_string_A, op_string_B, mode='anticommutator')
