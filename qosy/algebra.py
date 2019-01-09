#!/usr/bin/env python
"""
This module provides methods for performing algebraic manipulations
of OperatorStrings, such as taking products, commutators, and anticommutators,
and computing structure constants. It also provides a method for
computing the effect of commuting (or anti-commuting) an Operator
on a Basis of OperatorStrings.
"""

import numpy as np
import scipy.sparse as ss

from .tools import sort_sign
from .operatorstring import OperatorString
from .basis import Basis, Operator

def _operation_opstring(op_string_A, op_string_B, operation_mode='commutator', tol=1e-12):
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

def _operation_operator(operatorA, operatorB, operation_mode='commutator', tol=1e-12):
    """Perform an algebraic binary operation between
    two Operators.
    """
    
    coeffs_of_opstrings = dict()
    for (coeffA, op_string_A) in operatorA:
        for (coeffB, op_string_B) in operatorB:
            (coeffC, op_string) = _operation_opstring(op_string_A, op_string_B, operation_mode)
            coeff = coeffA*coeffB*coeffC

            if np.abs(coeff) < tol:
                continue

            if op_string in coeffs_of_opstrings:
                coeffs_of_opstrings[op_string] += coeff
            else:
                coeffs_of_opstrings[op_string] = coeff

    coeffs     = []
    op_strings = []
    for op_string in coeffs_of_opstrings:
        coeff = coeffs_of_opstrings[op_string]
        coeffs.append(coeff)
        op_strings.append(op_string)

    return Operator(coeffs, op_strings)
    
def product(op_A, op_B):
    """Compute the product of two OperatorStrings or Operators.

    Parameters
    ----------
    op_A : OperatorString or Operator
        OperatorString :math:`\hat{h}_a` or Operator :math:`\hat{\mathcal{O}}_A`.
    op_B : OperatorString or Operator
        OperatorString :math:`\hat{h}_b` or Operator :math:`\hat{\mathcal{O}}_B`.

    Returns
    -------
    (float, OperatorString) or Operator
        Operator that represents the product :math:`\hat{h}_a \hat{h}_b` 
        or :math:`\hat{\mathcal{O}}_A \hat{\mathcal{O}}_B`.

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> YY = qosy.opstring('Y 1 Y 2')
        >>> qosy.product(XX, YY) # Z_1 Z_2
    """

    if isinstance(op_A, OperatorString) and isinstance(op_B, OperatorString):
        return _operation_opstring(op_A, op_B, operation_mode='product')
    elif isinstance(op_A, Operator) and isinstance(op_B, Operator):
        return _operation_operator(op_A, op_B, operation_mode='product')
    else:
        raise ValueError()
    
def commutator(op_A, op_B):
    """Compute the commutator of two OperatorStrings or Operators.

    Parameters
    ----------
    op_A : OperatorString or Operator
        OperatorString :math:`\hat{h}_a` or Operator :math:`\hat{\mathcal{O}}_A`.
    op_B : OperatorString or Operator
        OperatorString :math:`\hat{h}_b` or Operator :math:`\hat{\mathcal{O}}_B`.

    Returns
    -------
    (float, OperatorString) or Operator
        Operator that represents the commutator :math:`[\hat{h}_a, \hat{h}_b]` 
        or :math:`[\hat{\mathcal{O}}_A, \hat{\mathcal{O}}_B]`.

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> Y  = qosy.opstring('Y 1')
        >>> qosy.commutator(XX, Y) # 2i Z_1 X_2
    """

    if isinstance(op_A, OperatorString) and isinstance(op_B, OperatorString):
        return _operation_opstring(op_A, op_B, operation_mode='commutator')
    elif isinstance(op_A, Operator) and isinstance(op_B, Operator):
        return _operation_operator(op_A, op_B, operation_mode='commutator')
    else:
        raise ValueError()

def anticommutator(op_A, op_B):
    """Compute the anticommutator of two OperatorStrings or Operators.

    Parameters
    ----------
    op_A : OperatorString or Operator
        OperatorString :math:`\hat{h}_a` or Operator :math:`\hat{\mathcal{O}}_A`.
    op_B : OperatorString or Operator
        OperatorString :math:`\hat{h}_b` or Operator :math:`\hat{\mathcal{O}}_B`.

    Returns
    -------
    (float, OperatorString) or Operator
        Operator that represents the anticommutator :math:`\{\hat{h}_a, \hat{h}_b\}` 
        or :math:`\{\hat{\mathcal{O}}_A, \hat{\mathcal{O}}_B\}`.

    Examples
    --------
        >>> XX = qosy.opstring('X 1 X 2')
        >>> YY = qosy.opstring('Y 1 Y 2')
        >>> qosy.anticommutator(XX, YY) # -2 Z_1 Z_2
    """

    if isinstance(op_A, OperatorString) and isinstance(op_B, OperatorString):
        return _operation_opstring(op_A, op_B, operation_mode='anticommutator')
    elif isinstance(op_A, Operator) and isinstance(op_B, Operator):
        return _operation_operator(op_A, op_B, operation_mode='anticommutator')
    else:
        raise ValueError()
    
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
            
            (coeff, os_C) = _operation_opstring(os_A, os_B, operation_mode=operation_mode)

            if np.abs(coeff) > tol:
                basisC += os_C
                ind_os_C = basisC.index(os_C)
                
                inds_os_C.append(ind_os_C)
                inds_os_A.append(ind_os_A)
                data.append(coeff)
                
        matrix_data.append((inds_os_C, inds_os_A, data))
        
    result = []
    for (inds_os_C, inds_os_A, data) in matrix_data:
        s_constants_B = ss.csc_matrix((data, (inds_os_C, inds_os_A)), dtype=complex, shape=(len(basisC), len(basisA)))
        result.append(s_constants_B)

    if return_extended_basis:
        return (result, basisC)
    else:
        return result

def _commutant_matrix_opstrings(basis, operator, operation_mode):
    """Compute the commutant matrix given a Basis of OperatorStrings.
    """
    
    (s_constants, extended_basis) = structure_constants(basis, operator._basis, operation_mode=operation_mode, return_extended_basis=True)
    
    commutant_matrix = ss.csc_matrix((len(extended_basis), len(basis)), dtype=complex)
    
    for ind_os in range(len(operator._basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os]

    return commutant_matrix

def _commutant_matrix_operators(operators, operator, operation_mode):
    """Compute the commutant matrix corresponding to "basis"
    made from a list of Operators.
    """

    # "operators" is a list of Operators O_a (that are by assumption
    # linearly independent.)
    #
    # We use the O_a as a "basis" rather than the OperatorStrings h_a,
    # but the actual calculations are done in terms of the h_a since
    # we know the rules for how the h_a commute.

    num_operators = len(operators)
    
    # Assemble a Basis of OperatorStrings that contains all of the
    # OperatorStrings in the list of Operators O_a.
    operators_basis = Basis()
    for op in operators:
        operators_basis += op._basis

    # Compute the structure constants for the Basis.
    (s_constants, extended_basis) = structure_constants(operators_basis, operator._basis, operation_mode=operation_mode, return_extended_basis=True)

    # Create a conversion matrix from the Basis
    # of OperatorStrings to the list of Operators.
    data     = []
    row_inds = []
    col_inds = []
    for ind_op in range(num_operators):
        for (coeff, os) in operators[ind_op]:
            data.append(coeff)
            row_inds.append(operators_basis.index(os))
            col_inds.append(ind_op)
    coeffs_operators = ss.csc_matrix((data, (row_inds, col_inds)), shape=(len(operators_basis), num_operators), dtype=complex)

    # Assemble the commutant matrix from the relevant structure constants
    # and convert to the list of Operators "basis".
    commutant_matrix = ss.csc_matrix((len(extended_basis), num_operators), dtype=complex)
    for ind_os in range(len(operator._basis)):
        commutant_matrix += operator.coeffs[ind_os] * s_constants[ind_os].dot(coeffs_operators)
                
    return commutant_matrix

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
    basis : Basis or list of Operators
        The Basis of OperatorStrings :math:`\hat{h}_a` or list of
        Operators :math:`\hat{\mathcal{O}}_a` to express the commutant matrix in.
    operator : Operator
        The operator :math:`\hat{\mathcal{O}}`.
    
    Returns
    -------
    scipy.sparse.csc_matrix of complex
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

    if isinstance(basis, Basis):
        return _commutant_matrix_opstrings(basis, operator, operation_mode)
    elif isinstance(basis, list) and isinstance(basis[0], Operator):
        return _commutant_matrix_operators(basis, operator, operation_mode)
    else:
        raise ValueError('Invalid basis type: {}'.format(type(basis)))
    
# TODO: finish
def generate_lie_algebra(operators, tol=1e-12):
    """Construct the Lie algebra generated
    by the given basis of Hermitian operators.

    Parameters
    ----------
    operators : Basis or list of Operators
        The Basis of OperatorStrings or 
        list of Operators to generate the 
        Lie algebra from.
    tol : float
        Tolerance with which to consider
        numbers zero. Defaults to 1e-12.

    Returns
    -------
    Basis or list of Operators
        All of the Lie algebra generators in
        the same format as ``operators``.
    """
    
    raise NotImplementedError('Not finished yet.')
    
    if isinstance(operators, Basis):
        pass
    elif isinstance(operators, list) and isinstance(operators[0], Operator):
        num_operators = len(operators)
        
        combined_basis = Basis()
        for op in operators:
            combined_basis += operators._basis

        row_inds = []
        col_inds = []
        data     = []
        for ind_op in range(num_operators):
            for (coeff, op_string) in operators[ind_op]:
                ind_os = combined_basis.index(op_string)

                if np.abs(coeff) > tol:
                    row_inds.append(ind_os)
                    col_inds.append(ind_op)
                    data.append(coeff)

        # Express the operators as vectors
        # in the combined_basis.
        vectors = ss.csc_matrix((data, (row_inds, col_inds)), dtype=complex, shape=(len(combined_basis), num_operators))

        # The projection onto the vector space
        # spanned by vectors.
        vecsHvecs_inv = ssla.inv((vectors.H).dot(vectors))
        projector     = vectors.dot(vecsHvecs_inv.dot(vectors.H))
        
        new_operators = []
        for indA in range(num_operators):
            for indB in range(indA+1, num_operators):
                opA = operators[indA]
                opB = operators[indB]
                
                opC = commutator(opA, opB)
                
                # TODO: check whether opC is in the vector space
                # spanned by vectors.
                vecC    = ss.csc_matrix(opC.to_vector(combined_basis))
                overlap = (vecC.H).dot(projector.dot(vecC))

                # TODO: is it possible for opC to be a
                # linear combination of operators in
                # new_operators but not the original
                # operators?
                if np.abs(overlap - 1.0) < tol:
                    new_operators.append(opC)
    else:
        raise ValueError('Invalid operators type: {}'.format(type(operators)))

# TODO: test
def killing_form(operators):
    """Compute the Killing form 
    :math:`K_{ab}=\sum_{cd} N_{ac}^d N_{bd}^c`
    for the given operators :math:`\hat{h}_a` 
    with structure constants :math:`N_{ab}^c`.

    Parameters
    ----------
    operators : Basis or list of Operators
        The Basis of OperatorStrings or list 
        of Operators :math:`\hat{h}_a`.

    Returns
    -------
    scipy.sparse.csc_matrix
        The Killing form :math:`K_{ab}`.
    """

    raise NotImplementedError('Not finished yet.')
    
    if isinstance(operators, Basis):
        # s_constants = [S_{bc}^{(1)}, S_{bc}^{(2)}, ...]
        # N_{ab}^c = S_{ac}^{(b)} = -N_{ba}^c = -S_{bc}^{(a)}
        (s_constants, _) = structure_constants(operators, operators)

        dim_a = len(s_constants)
        dim_b = dim_a
        dim_c = int(s_constants[0].shape[1])
        
        # M_{a,(bc)} = N_{ab}^c = -N_{ba}^c = -S_{bc}^{(a)}
        row_inds    = []
        col_inds_bc = []
        col_inds_cb = []
        data        = []
        for ind_a in range(dim_a):
            for (ind_b, ind_c) in s_constants[ind_a].nonzero():
                col_ind_bc = ind_b*dim_b + ind_c
                col_ind_cb = ind_c*dim_c + ind_b

                row_inds.append(ind_a)
                col_inds_bc.append(col_ind_bc)
                col_inds_cb.append(col_ind_cb)

                data.append(-s_constants[ind_a][ind_b, ind_c])

        # M_{a,(bc)}
        matrix1 = ss.csc_matrix((data, (row_inds, col_inds_bc)), dtype=complex, shape=(dim_a, dim_b*dim_c))
        # M_{(cb),a}
        matrix1 = ss.csc_matrix((data, (col_inds_cb, row_inds)), dtype=complex, shape=(dim_b*dim_c, dim_a))

        # K_{ab} = \sum_{cd} M_{a(cd)} M_{b(dc)}
        killing = matrix_bc.dot(matrix_cb)
        
        return killing
    elif isinstance(operators, list) and isinstance(operators[0], Operator):
        pass
    else:
        raise ValueError('Invalid type of operators: {}'.format(type(operators)))

# TODO: test
def generate_center(operators, tol=1e-10):
    """Construct the center of the Lie algebra 
    generated by the given operators.
    
    Parameters
    ----------
    operators : Basis or list of Operators
        The Basis of OperatorStrings or 
        list of Operators that generate a
        Lie algebra.
    tol : float, optional
        The tolerance with which to consider
        eigenvalues to be zero. Defaults to 1e-10.

    Returns
    -------
    list of Operators
        The generators of the center of the Lie algebra.
    """
    
    raise NotImplementedError('Not finished yet.')

    # Compute the Killing form.
    killing = killing_form(operators)

    # Find the null vectors of the Killing form.
    (evals, evecs) = nla.eigh(killing)
    inds_ns = np.where(np.abs(evals) < tol)[0]
    null_space = evecs[:, inds_ns]

    num_null_vecs = int(null_space.shape[1])

    # From the null vectors, construct a
    # list of Operators that form the center
    # of the Lie algebra.
    center_operators = []
    for ind_vec in range(num_null_vecs):
        if isinstance(operators, list) and isinstance(operators[0], Operator):
            center_op = Operator([], [], operators[0].op_type)
            for ind_basis in range(len(operators)):
                coeff = null_space[ind_basis, ind_vec]
            
                center_op += coeff * operators[ind_basis]

            center_op.remove_zeros()
            center_op.normalize()
        elif isinstance(operators, Basis):
            center_op = Operator(null_space[:, ind_vec], operators)
            center_op.remove_zeros()
            center_op.normalize()
        else:
            raise ValueError('Invalid operators type: {}'.format(type(operators)))
        
        center_operators.append(center_op)

    return center_operators
