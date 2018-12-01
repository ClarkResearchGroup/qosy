#!/usr/bin/env python
import numpy as np
import scipy.sparse as ss

from .tools import sort_sign
from .operatorstring import OperatorString
from .basis import Basis, Operator

def _operation(op_string_A, op_string_B, operation_mode='commutator', tol=1e-12):
    """Performs a binary operation between two OperatorStrings h_a and h_b.
    
    operation_mode='commutator'     returns [h_a, h_b] = h_a * h_b - h_b * h_a
    operation_mode='anticommutator' returns {h_a, h_b} = h_a * h_b + h_b * h_a
    operation_mode='product'        returns h_a * h_b
    """
    
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
    """Compute the product of two OperatorStrings.

    Parameters
    ----------
    op_string_A : OperatorString
        OperatorString :math:`\hat{h}_a`.
    op_string_B : OperatorString
        OperatorString :math:`\hat{h}_b`.

    Returns
    -------
    Operator
        Operator that represents the product :math:`\hat{h}_a \hat{h}_b`

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> YY = qosy.opstring('Y 1 Y 2')
        >>> qosy.product(XX, YY) # Z_1 Z_2
    """
    
    return _operation(op_string_A, op_string_B, operation_mode='product')

def commutator(op_string_A, op_string_B):
    """Compute the commutator of two OperatorStrings.

    Parameters
    ----------
    op_string_A : OperatorString
        OperatorString :math:`\hat{h}_a`.
    op_string_B : OperatorString
        OperatorString :math:`\hat{h}_b`.

    Returns
    -------
    Operator
        Operator that represents the commutator :math:`[\hat{h}_a, \hat{h}_b] = \hat{h}_a \hat{h}_b - \hat{h}_b \hat{h}_a`.

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> Y  = qosy.opstring('Y 1')
        >>> qosy.commutator(XX, Y) # 2i Z_1 X_2
    """
    
    return _operation(op_string_A, op_string_B, operation_mode='commutator')

def anticommutator(op_string_A, op_string_B):
    """Compute the anticommutator of two OperatorStrings.

    Parameters
    ----------
    op_string_A : OperatorString
        OperatorString :math:`\hat{h}_a`.
    op_string_B : OperatorString
        OperatorString :math:`\hat{h}_b`.
    
    Returns
    -------
    Operator
        Operator that represents the anticommutator :math:`\{\hat{h}_a, \hat{h}_b\} = \hat{h}_a \hat{h}_b + \hat{h}_b \hat{h}_a`.

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> YY = qosy.opstring('Y 1 Y 2')
        >>> qosy.anticommutator(XX, YY) # -2 Z_1 Z_2
    """
    
    return _operation(op_string_A, op_string_B, operation_mode='anticommutator')

def structure_constants(basisA, basisB, operation_mode='commutator', return_extended_basis=False, tol=1e-12):
    """Compute the structure constants obtained by taking
    the commutator between OperatorStrings from two different
    bases.

    The structure constants :math:`N_{ab}^c` are defined by
        :math:`[\hat{h}_a, \hat{h}_b] = \sum_{c} N_{ab}^c \hat{h}_c`

    Parameters
    ----------
    basisA : Basis
        The basis of OperatorStrings containing 
        :math:`\hat{h}_a`.
    basisB : Basis
        The basis of OperatorStrings containing 
        :math:`\hat{h}_b`.
    operation_mode : str, optional
        Specifies what binary algebraic operation to use to
        define the structure constants. By default, uses
        'commutator', but can also use 'anticommutator'
        or 'product'.
    return_extended_basis : bool, optional
        Specifies whether to return the "extended basis"
        of OperatorStrings, which contains the :math:`\hat{h}_c`
        OperatorStrings generated by the binary operation
        between :math:`\hat{h}_a` and :math:`\hat{h}_b`.
        Defaults to False.

    Returns
    -------
    list of scipy.sparse.csc_matrix or (list of scipy.sparse.csc_matrix, Basis)
        If `return_extended_basis` is False, returns only 
        the structure constants, which are collected into
        a list of scipy sparse matrices. The list is 
        organized as follows: the tensor :math:`N_{ab}^c` is
        split up into matrices :math:`A_{ca}^{(b)} =N_{ab}^c`.
        The list stores the matrices [:math:`A^{(1)},A^{(2)},\ldots`].
        For this reason, one should make sure that the
        dimension of `basisB` is less than or equal to `basisA`.
        If `return_extended_basis` is True, returns
        a tuple, of the structure constants and the extended 
        basis.

    Examples
    --------
    To compute the structure constants of all Pauli
    strings on three orbitals labeled `1,2,3`
        >>> basis      = qosy.cluster_basis(3, [1,2,3], 'Pauli')
        >>> sconstants = qosy.structure_constants(basis, basis)
    """
    
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

    if return_extended_basis:
        return (result, basisC)
    else:
        return result

"""
def killing_form(basis):
    (s_constants, _) = structure_constants(basis, basis)
    
    # TODO
"""

def commutant_matrix(basis, operator, operation_mode='commutator'):
    """Compute the commutant matrix that describes the effect
    of commuting with the given Operator :math:`\hat{\mathcal{O}}` 
    and express it in the given Basis of OperatorStrings :math:`\hat{h}_a`.
    
    The commutant matrix :math:`C_{\hat{\mathcal{O}}}` is defined
    through the relation
        :math:`[\hat{h}_a, \hat{\mathcal{O}}] \equiv \sum_c (C_{\hat{\mathcal{O}}})_{ca} \hat{h}_c`
    where the OperatorStrings :math:`\hat{h}_c` make up an "extended
    basis" of OperatorStrings that is in general different from
    the Basis spanned by :math:`\hat{h}_a`.
    
    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings to express the commutant matrix.
    
    Returns
    -------
    scipy.sparse.csc_matrix
        The commutant matrix as a complex, sparse matrix.

    Examples
    --------
    Consider the basis of all Pauli strings on 
    three orbitals labeled `1,2,3`:
        >>> basis = qosy.cluster_basis(3, [1,2,3], 'Pauli')
    The commutant matrix in this basis that describes
    how operators commute with the Heisenberg bond
    between orbitals 1 and 2 is
        >>> XX = qosy.opstring('X 1 X 2')
        >>> YY = qosy.opstring('Y 1 Y 2')
        >>> ZZ = qosy.opstring('Z 1 Z 2')
        >>> bond = qosy.Operator(np.ones(3), [XX,YY,ZZ])
        >>> com_matrix = qosy.commutant_matrix(basis, bond)
    """
    
    (s_constants, extended_basis) = structure_constants(basis, operator._basis, operation_mode=operation_mode, return_extended_basis=True)

    commutant_matrix = ss.csc_matrix(dtype=complex, shape=(len(basis), len(extended_basis)))

    for ind_os in range(len(operator_basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os]

    return commutant_matrix
