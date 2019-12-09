#!/usr/bin/env python
"""
This module provides functions for converting between different
types of OperatorStrings. 

Note
----
Currently, this module only supports the conversion between strings 
of fermions and strings of Majorana fermions.
"""

import itertools as it
import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from .tools import sort_sign, compare
from .config import *
from .operatorstring import OperatorString
from .basis import Basis, Operator
from .algebra import product

# TODO: add support for permutations or alternative X,Y,Z orderings.
def _jordan_wigner(op_string):
    # Convert a Pauli string to a Majorana string or vice-versa
    # using the Jordan-Wigner transformation: 
    #  a_i = (\prod_{j=1}^{i-1} Z_j) X_i
    #  b_i = (\prod_{j=1}^{i-1} Z_j) Y_i
    #  d_i = Z_i
    #  X_i = (\prod_{j=1}^{i-1} d_j) a_i
    #  Y_i = (\prod_{j=1}^{i-1} d_j) b_i
    #  Z_i = d_i

    op_type = op_string.op_type

    total_coeff = op_string.prefactor
    
    if op_type == 'Pauli':
        result = OperatorString([], [], op_type='Majorana')
        
        num_orbitals = len(op_string.orbital_operators)
        for ind_orb in range(num_orbitals):
            orb_op    = op_string.orbital_operators[ind_orb]
            orb_label = op_string.orbital_labels[ind_orb]

            if orb_op == 'X':
                jw_ops = ['D']*orb_label + ['A']
                jw_labels = np.arange(orb_label+1)
            elif orb_op == 'Y':
                jw_ops = ['D']*orb_label + ['B']
                jw_labels = np.arange(orb_label+1)
            elif orb_op == 'Z':
                jw_ops    = ['D']
                jw_labels = [orb_label]
            else:
                raise ValueError('Invalid operator {} in OperatorString.'.format(orb_op))
            
            jw_string       = OperatorString(jw_ops, jw_labels, 'Majorana')
            (coeff, result) = product(result, jw_string)
            total_coeff *= coeff
            
    elif op_type == 'Majorana':
        result = OperatorString([], [], op_type='Pauli')
        
        num_orbitals = len(op_string.orbital_operators)
        for ind_orb in range(num_orbitals):
            orb_op    = op_string.orbital_operators[ind_orb]
            orb_label = op_string.orbital_labels[ind_orb]

            if orb_op == 'A':
                jw_ops = ['Z']*orb_label + ['X']
                jw_labels = np.arange(orb_label+1)
            elif orb_op == 'B':
                jw_ops = ['Z']*orb_label + ['Y']
                jw_labels = np.arange(orb_label+1)
            elif orb_op == 'D':
                jw_ops    = ['Z']
                jw_labels = [orb_label]
            else:
                raise ValueError('Invalid operator {} in OperatorString.'.format(orb_op))
            
            jw_string       = OperatorString(jw_ops, jw_labels, 'Pauli')
            (coeff, result) = product(result, jw_string)
            total_coeff *= coeff
    else:
        raise ValueError('Cannot perform Jordan-Wigner transformation on OperatorString of op_type: {}'.format(op_type))

    return (total_coeff, result)
    
def _fermion_string_from_cdag_c_labels(prefactor, c_dag_labels, c_labels):
    # Construct a fermion string operator from the labels of the creation and
    # anhillation (c^\dagger and c) operators.
    
    c_labels_reversed = np.copy(c_labels)
    c_labels_reversed = c_labels_reversed[::-1]
    
    orbital_operators = ['CDag']*len(c_dag_labels) + ['C']*len(c_labels)
    orbital_labels    = np.concatenate((c_dag_labels, c_labels_reversed))
    
    return OperatorString(orbital_operators, orbital_labels, prefactor=prefactor, op_type='Fermion')

def _convert_majorana_string(op_string, include_identity=False):
    # Converts a Majorana string to an Operator
    # that is a linear combination of Fermion strings.
    
    if op_string.op_type != 'Majorana':
        raise ValueError('Trying to convert a Majorana string to a Fermion string but given an OperatorString of type {}'.format(op_string.op_type))
    
    ops    = op_string.orbital_operators
    labels = op_string.orbital_labels

    # The identity operator.
    if len(ops) == 0:
        if include_identity:
            return Operator(np.array([1.0]), [OperatorString([], [], 'Fermion')])
        else:
            return Operator([], [], 'Fermion')
        
    # Used to make sure that the fermion labels end up normal ordered.
    # I add this large number to the anhillation operators c_i labels
    # so that they end up last in the list of operators sorted by labels.
    large_number = 4*np.maximum(1,np.max(labels)) + 4

    [op1, op2, op3] = MAJORANA_OPS
    
    num_ops = len(ops)

    coeffs_fermion     = []
    op_strings_fermion = []

    # Perform the Majorana to Fermion string basis conversion.
    # Expand a_i = c_i + c_i^\dagger, b_i = -i c_i + i c_i^\dagger, d_i = - 2 c_i^\dagger c_i + I
    # A "term choice" of 0 (1) corresponds to the first (second) term, e.g.,
    # 1 for a_i is c_i^\dagger, 0 for d_i is -2 c_i^\dagger c_i.
    possible_term_choices = list(it.product([0,1], repeat=num_ops))
    for term_choices in possible_term_choices:
        coeff = op_string.prefactor

        fermion_labels = []
        
        num_cdags = 0
        num_cs    = 0

        for ind_op in range(num_ops):
            label = labels[ind_op]
            if ops[ind_op] == op1:   # a_j = c_j + c_j^\dagger
                if term_choices[ind_op] == 0:   # c_j
                    fermion_labels.append(-label + large_number)
                    num_cs += 1
                elif term_choices[ind_op] == 1: # c_j^\dagger
                    fermion_labels.append(label)
                    num_cdags += 1
            elif ops[ind_op] == op2: # b_j = -i c_j + i c_j^\dagger
                if term_choices[ind_op] == 0:   # -i c_j
                    coeff *= -1j
                    fermion_labels.append(-label + large_number)
                    num_cs += 1
                elif term_choices[ind_op] == 1: # i c_j^\dagger
                    coeff *= 1j
                    fermion_labels.append(label)
                    num_cdags += 1
            elif ops[ind_op] == op3: # d_j = - 2 c_j^\dagger c_j + I
                if term_choices[ind_op] == 0:   # -2 c_j^\dagger c_j
                    coeff *= -2.0
                    fermion_labels.append(label)
                    fermion_labels.append(-label + large_number)
                    num_cdags += 1
                    num_cs    += 1
                elif term_choices[ind_op] == 1: # I
                    coeff *= 1.0

        # Resulting operator is identity I.
        if len(fermion_labels) == 0 and not include_identity:
            continue
        
        (sorted_fermion_labels, sign) = sort_sign(fermion_labels)
        coeff *= sign
        
        # The i_1,\ldots,i_m labels
        cdag_labels = sorted_fermion_labels[0:num_cdags]
        # The j_1,\ldots,j_m labels
        c_labels    = large_number - sorted_fermion_labels[num_cdags:]
        c_labels    = c_labels[::-1]
        
        lex_order = compare(cdag_labels, c_labels)
        
        # Resulting operator is not lexicographically sorted. Ignore it.
        if lex_order < 0:
            continue
        # Resulting operator is of type 1: c^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{i_m} \cdots c_{i_1}. 
        elif lex_order == 0:
            coeffs_fermion.append(coeff)
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1.0, cdag_labels, c_labels))
        # Resulting operator is of type 2: c^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{j_l} \cdots c_{j_1} + H.c.
        elif lex_order > 0 and np.abs(np.imag(coeff)) < np.finfo(float).eps:
            coeffs_fermion.append(np.real(coeff))
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1.0, cdag_labels, c_labels))
        # Resulting operator is of type 3: ic^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{j_l} \cdots c_{j_1} + H.c.
        elif lex_order > 0 and np.abs(np.real(coeff)) < np.finfo(float).eps:
            coeffs_fermion.append(np.real(coeff/(1j)))
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1j, cdag_labels, c_labels))
        else:
            raise ValueError('Invalid lex_order = {} and coeff = {}'.format(lex_order, coeff))

    return Operator(np.array(coeffs_fermion), op_strings_fermion)


def _convert_fermion_string(op_string, include_identity=False):
    # Converts a Fermion string into an Operator
    # that is a linear combination of Majorana strings.

    # Obtain the labels of the CDag and C operators (in ascending order).
    cdag_labels = [o_lab for (o_lab, o_op) in zip(op_string.orbital_labels, op_string.orbital_operators) if o_op == 'CDag']
    c_labels    = [o_lab for (o_lab, o_op) in zip(op_string.orbital_labels, op_string.orbital_operators) if o_op == 'C']
    c_labels    = c_labels[::-1]
    
    
    # Store the operator type (C, CDag, CDagC) of every label.
    label_types = dict()
    for cdag_label in cdag_labels:
        label_types[cdag_label] = 'CDag'
    for c_label in c_labels:
        if c_label in label_types:
            label_types[c_label] = 'CDagC'
        else:
            label_types[c_label] = 'C'

    # Put all the labels together and reorder them so that the resulting
    # Majorana operator labels are in the correct order. Keep track of the
    # sign due to reordering when you do this.
    fermion_labels = cdag_labels + c_labels[::-1]
    (sorted_fermion_labels, sign) = sort_sign(fermion_labels)

    # Collect the information about the CDag, C, CDagC fermion operators and their labels into
    # the ops = [(orbital operator, orbital operator label), ...] list, which has the operators ordered correctly.
    ops     = []
    num_ops = 0
    for f_label in sorted_fermion_labels:
        f_op = label_types[f_label]
        if not (f_op, f_label) in ops:
            ops.append((f_op, f_label))
            num_ops += 1
            
    coeffs_majorana     = []
    op_strings_majorana = []
    
    # Perform the Fermion to Majorana string basis conversion.
    # Expand c_j = 1/2(a_j + ib_j), c^\dagger_j = 1/2(a_j - i b_j^\dagger), c^\dagger_j c_j = 1/2 (-d_j + I)
    # A "term choice" of 0 (1) corresponds to the first (second) term, e.g.,
    # 1 for c_j is i/2 b_j, 0 for c^\dagger_j c_j is -1/2 d_j.
    possible_term_choices = list(it.product([0,1], repeat=num_ops))
    for term_choice in possible_term_choices:
        coeffM  = 1.0 #op_string.prefactor * sign
        opNameM = ''
        orbital_operators = []
        orbital_labels    = []
        for ind_op in range(num_ops):
            (op, op_label) = ops[ind_op]

            if op == 'CDag' and term_choice[ind_op] == 0:
                coeffM *= 0.5
                orbital_operators.append('A')
                orbital_labels.append(op_label)
            elif op == 'CDag' and term_choice[ind_op] == 1:
                coeffM *= -0.5j
                orbital_operators.append('B')
                orbital_labels.append(op_label)
            elif op == 'C' and term_choice[ind_op] == 0:
                coeffM *= 0.5
                orbital_operators.append('A')
                orbital_labels.append(op_label)
            elif op == 'C' and term_choice[ind_op] == 1:
                coeffM *= 0.5j
                orbital_operators.append('B')
                orbital_labels.append(op_label)
            elif op == 'CDagC' and term_choice[ind_op] == 0:
                coeffM *= -0.5
                orbital_operators.append('D')
                orbital_labels.append(op_label)
            elif op == 'CDagC' and term_choice[ind_op] == 1:
                coeffM *= 0.5
            else:
                raise ValueError('Invalid op and term_choice: {} {}'.format(op, term_choice[ind_op]))

        # Ignore the identity operator.
        if len(orbital_operators) == 0 and not include_identity:
            continue

        op_string_M = OperatorString(orbital_operators, orbital_labels, 'Majorana')
        
        coeffM /= op_string_M.prefactor
        coeffM *= sign
        coeffM *= op_string.prefactor

        # The type 2 and 3 fermion strings have a Hermitian conjugate: (CDag ... C ...) + H.c.,
        # that ensures that the resulting operators are Hermitian. In our conversion,
        # if an operator ends up being anti-Hermitian, then it cancels with the Hermitian conjugate.
        # Otherwise, its coefficient doubles because it equals the Hermitian conjugate.
        if cdag_labels != c_labels:
            if np.abs(np.imag(coeffM)) > np.finfo(float).eps: #1e-16:
                continue
            else:
                coeffM *= 2.0

        coeffs_majorana.append(coeffM)
        op_strings_majorana.append(op_string_M)

    return Operator(np.array(coeffs_majorana), op_strings_majorana)

def _convert_operator_string(op_string, to_op_type, include_identity=False):
    # Converts an OperatorString to a linear combination of
    # OperatorStrings of the given op_type.
    
    if op_string.op_type == to_op_type:
        return Operator(np.array([1.0]), [op_string])
    
    if op_string.op_type == 'Majorana' and to_op_type == 'Fermion':
        return _convert_majorana_string(op_string, include_identity)
    elif op_string.op_type == 'Fermion' and to_op_type == 'Majorana':
        return _convert_fermion_string(op_string, include_identity)
    elif op_string.op_type == 'Majorana' and to_op_type == 'Pauli':
        (coeff, pauli_os) =  _jordan_wigner(op_string)
        return Operator(np.array([coeff]), [pauli_os], 'Pauli')
    elif op_string.op_type == 'Pauli' and to_op_type == 'Majorana':
        (coeff, maj_os) =  _jordan_wigner(op_string)
        return Operator(np.array([coeff]), [maj_os], 'Majorana')
    elif op_string.op_type == 'Fermion' and to_op_type == 'Pauli':
        maj_op = _convert_fermion_string(op_string, include_identity)

        pauli_coeffs     = []
        pauli_op_strings = []
        for (maj_coeff, maj_os) in maj_op:
            (pauli_coeff, pauli_os) = _jordan_wigner(maj_os)
            pauli_coeffs.append(pauli_coeff*maj_coeff)
            pauli_op_strings.append(pauli_os)
        pauli_op = Operator(np.array(pauli_coeffs), pauli_op_strings, op_type='Pauli')
        
        return pauli_op
    elif op_string.op_type == 'Pauli' and to_op_type == 'Fermion':
        (coeff, maj_os) = _jordan_wigner(op_string)
        ferm_op = _convert_majorana_string(maj_os, include_identity)
        return coeff*ferm_op
    else:
        raise NotImplementedError('Unsupported conversion of OperatorString of op_type {} to {}'.format(op_string.op_type, to_op_type))

def _convert_operator(operator, to_op_type, include_identity=False):
    # Create an empty Operator in the new basis.
    new_operator = Operator(op_type=to_op_type)

    # Convert each OperatorString to an Operator of the new op_type
    # and add up the results.
    for ind_os in range(len(operator)):
        coeff                 = operator.coeffs[ind_os]
        op_string             = operator._basis[ind_os] 
        new_op_from_op_string = _convert_operator_string(op_string, to_op_type, include_identity)
        
        new_operator += coeff * new_op_from_op_string

    # Remove the unnecessary entries that cancelled during the conversion. 
    #new_operator = new_operator.remove_zeros(tol=tol)
    
    return new_operator

def convert(operator, to_op_type, include_identity=False):
    """Convert an Operator or OperatorString from 
    one OperatorString type to another.

    Parameters
    ----------
    operator : Operator or OperatorString
        The Operator or OperatorString to convert.
    to_op_type : str
        The type of operator to convert to.
    include_identity : bool, optional
        Specifies whether to include the identity OperatorString
        in the output Operator. By default, the identity
        is not included. (This can be interpreted as
        subtracting multiples of the identity from
        traceful OperatorStrings to make them traceless.)

    Returns
    -------
    Operator
        An Operator representing the converted Operator or OperatorString.

    Examples
    --------
        >>> qosy.convert(qosy.opstring('CDag 1 C 1'), 'Majorana')
        >>> qosy.convert(qosy.opstring('1j A 1 B 2'), 'Fermion')

    Notes
    -----
    Note that an OperatorString of one type can be a 
    linear combination of exponentially many OperatorStrings 
    of another type. For example, the single Majorana string
        :math:`\hat{d}_1 \cdots \hat{d}_N = (\hat{I}-2\hat{c}^\dagger_1 \hat{c}_1) \cdots (\hat{I}-2\hat{c}^\dagger_N \hat{c}_N)`
    is a linear combination of :math:`2^N` Fermion strings.

    Therefore, it can be inefficient to perform this 
    conversion in general if converting an OperatorString 
    with support on many orbitals. In such
    circumstances, the `convert` method should be 
    avoided.

    Fermionic operator strings are converted to Pauli strings
    (and vice-versa) using the standard Jordan-Wigner transformation.
    """
    
    if isinstance(operator, OperatorString):
        return _convert_operator_string(operator, to_op_type, include_identity)
    elif isinstance(operator, Operator):
        return _convert_operator(operator, to_op_type, include_identity)
    else:
        raise TypeError('Cannot convert invalid operator type: {}'.format(type(operator)))

def conversion_matrix(basis_from, basis_to, tol=1e-16):
    """Construct a basis transformation matrix for
    converting between two different Bases of 
    OperatorStrings.

    Consider an operator of the form 
        :math:`\hat{\mathcal{O}} = \sum_a g_a \hat{h}_a = \sum_b g_b' \hat{h}_b'`
    where :math:`\hat{h}_a` and :math:`\hat{h}_b'` 
    are OperatorString basis vectors from two 
    different Bases.

    The transformation matrix :math:`B_{ba}` that
    this method computes satisfies 
        :math:`\hat{h}_a = \sum_b B_{ba} \hat{h}_b'`
    or 
          :math:`\\textbf{h} = \\textbf{B} \\textbf{h}'`
    in matrix-vector notation. If the :math:`h_a` 
    form an orthonormal basis, then
       :math:`g'_b = \sum_a B_{ba} g_a`
    or 
         :math:`\\textbf{g}' = \\textbf{B} \\textbf{g}`
    in matrix-vector notation
    
    Parameters
    ----------
    basis_from : Basis
        The Basis of OperatorStrings to convert from.
    basis_to : Basis
        The Basis of OperatorStrings to convert to.
    tol : float, optional
        The tolerance to truncate entries of the conversion matrix.
        Defaults to 1e-16.

    Returns
    -------
    scipy.sparse.csc_matrix of complex
        A sparse, invertible basis transformation matrix.

    Examples
    --------
    To convert between bases of two-orbital Majorana and Fermion strings:
        >>> basisA = qosy.cluster_basis(2, [1,2], 'Majorana')
        >>> basisB = qosy.cluster_basis(2, [1,2], 'Fermion')
        >>> B = qosy.conversion_matrix(basisA, basisB)
    The inverse transformation matrix can be built in two ways:
        >>> import scipy.sparse.linalg as sla
        >>> Binv1 = sla.inv(B)
        >>> Binv2 = qosy.conversion_matrix(basisB, basisA)
    """

    if len(basis_from) != len(basis_to):
        raise ValueError('Cannot create an invertible transformation matrix between bases of different sizes: {} {}'.format(len(basis_from), len(basis_to)))

    if len(basis_from) == 0:
        return ss.csc_matrix(dtype=complex,shape=(0,0))
    
    to_op_type = basis_to.op_strings[0].op_type
    
    # Collect the entries of the sparse conversion matrix.
    row_inds = []
    col_inds = []
    data     = []
    
    for ind_from in range(len(basis_from)):
        op_string_from        = basis_from[ind_from]
        converted_operator_to = convert(op_string_from, to_op_type)

        for (coeff, op_string_to) in converted_operator_to:
            if np.abs(coeff) > tol:
                ind_to = basis_to.index(op_string_to)

                row_inds.append(ind_to)
                col_inds.append(ind_from)
                data.append(coeff)

    conversion_matrix = ss.csc_matrix((data, (row_inds, col_inds)), dtype=complex, shape=(len(basis_from), len(basis_to)))

    conversion_matrix.sum_duplicates()
    conversion_matrix.eliminate_zeros()

    return conversion_matrix
    
