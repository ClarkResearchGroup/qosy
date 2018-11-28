#!/usr/bin/env python
import numpy as np
import itertools as it
#import lattice

from config import *
import tools

class OperatorString:
    """An OperatorString object represents a string 
    of operators that act on orbital degrees of freedom.
        
    Attributes
    ----------
    orbital_operators : list of str
        A list of the names of the operators on the orbitals. 
        Ex: 'X','Y', or 'Z' for Pauli strings.
    orbital_labels : list or ndarray of int
        A list or array of the (unique) integer label of each orbital.
    op_type : str
        Specifies the type of operator string considered: 'Pauli', 
        'Majorana', or 'Fermion'.
    name : str
        The string representation of the operator string, used
        for hashing.
    prefactor : complex
        A prefactor multiplying the operator.
    
    """
    
    def __init__(self, orbital_operators, orbital_labels, op_type, prefactor=1.0):
        """Construct an OperatorString object that represents
        a string of operators that act on orbital degrees of freedom.
        
        Parameters
        ----------
        orbital_operators : list of str
            A list of the names of the operators on the orbitals. 
            Ex: 'X','Y', or 'Z' for Pauli strings.
        orbital_labels : list or array of int
            A list or ndarray of the (unique) integer label of each orbital.
        op_type : str
            Specifies the type of operator string considered: 'Pauli', 
            'Majorana', or 'Fermion'.
        prefactor : complex, optional
            A prefactor multiplying the operator.

        Examples
        --------
        The Pauli string \sigma^x_1 \sigma^y_2 \sigma^z_4 can be constructed with
        >>> OperatorString(['X', 'Y', 'Z'], [1, 2, 4], 'Pauli')
        The Majorana string i a_1 b_3 d_5 d_6 can be constructed with
        >>> OperatorString(['A', 'B', 'D', 'D'], [1, 3, 5, 6], 'Majorana')
        The Fermion string c_1^\dagger c_2^\dagger c_3 + H.c. can be constructed with
        >>> OperatorString(['CDag', 'CDag', 'C'], [1, 2, 3], 'Fermion')
        """

        self.orbital_operators = orbital_operators
        self.orbital_labels    = np.array(orbital_labels, dtype=int)
        self.op_type           = op_type

        if len(self.orbital_operators) != len(self.orbital_labels):
            ValueError('The number of orbital operators and labels is inconsistent for the OperatorString: {} {}'.format(len(self.orbital_operators), len(self.orbital_labels)))

        self.prefactor = prefactor

        # Stored for use in computing commutators.
        # A dictionary of the labels to their index in the operator string.
        self._indices_orbital_labels = dict()
        for ind_orbital in range(len(self.orbital_labels)):
            self._indices_orbital_labels[self.orbital_labels[ind_orbital]] = ind_orbital
        
        # Compute the prefactor automatically if a Majorana operator.
        if self.op_type == 'Majorana':
            # Stored for use in computing commutators.
            # The labels of orbital operators that are 'A' or 'B'.
            self._labels_ab_operators = np.array([self.orbital_labels[ind] for ind in range(len(self.orbital_labels)) if self.orbital_operators[ind] in ['A', 'B']], dtype=int)
            num_ab = len(self._labels_ab_operators)

            # The prefactor is 1 or 1j, depending
            # on whether reversing the order of operators creates
            # a +1 or -1 sign due to anti-commutation operators.
            num_swaps_to_reorder = (num_ab*(num_ab-1))/2
            if num_swaps_to_reorder % 2 == 1:
                self.prefactor = 1j

        if (self.op_type == 'Pauli' and self.prefactor != 1) \
           or (self.op_type == 'Majorana' and self.prefactor not in [1, 1j]) \
           or (self.op_type == 'Fermion' and self.prefactor not in [1, 1j]):
            raise ValueError('Invalid prefactor {} for operator string of op_type {}'.format(self.prefactor, self.op_type))
                 
        name_list = [str(self.prefactor),' ']
        for (op, la) in zip(self.orbital_operators, self.orbital_labels):
            name_list.extend([op, ' ', str(la), ' '])

        self.name = ''.join(name_list)

    def __str__(self):
        return self.name
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        equals = (self.orbital_operators == other.orbital_operators) and (self.orbital_labels == other.orbital_labels).all() and (self.op_type == other.op_type)
        
        return equals



def opstring(op_string_name, op_type=None):
    """Creates an OperatorString from a python string description.
    
    Parameters
    ----------
    op_string_name : str
        The string representing the OperatorString. If no prefactor
        is provided, it is assumed to be 1.0. (Note: for Majorana strings,
        the prefactor is automatically computed, so does not need to be
        initialized.)
    
    Returns
    -------
    OperatorString
        The corresponding OperatorString object.

    Examples
    --------
    The Pauli string \sigma^x_1 \sigma^y_2 \sigma^z_4 can be constructed with
    >>> opstring('X 1 Y 2 Z 4')
    or
    >> opstring('1.0 X 1 Y 2 Z 4')
    The Majorana string i a_1 b_3 d_5 d_6 can be constructed with
    >>> opstring('1j A 1 B 3 D 5 D 6')
    The Fermion string c_1^\dagger c_2^\dagger c_3 + H.c. can be constructed with
    >>> opstring('CDag 1 CDag 2 C 3')
    An identity operator that can be used with Majorana strings can be constructed with
    >>> opstring('I','Majorana')
    """
    
    name_list = op_string_name.strip().split()

    if name_list == [] or (len(name_list) == 1 and name_list[0] in ['1', '1.0', 'I']):
        if op_type is None:
            raise ValueError('When specifying an identity operator, you need to provide an op_type.')
        else:
            return OperatorString([], [], op_type)
    
    if name_list[0] in VALID_OPS:
        orbital_operators = name_list[0::2]
        orbital_labels    = np.array(name_list[1::2], dtype=int)
        prefactor         = 1.0
    elif name_list[0] in ['1', '1.0', '1j'] and name_list[1] in VALID_OPS:
        orbital_operators = name_list[1::2]
        orbital_labels    = np.array(name_list[2::2], dtype=int)
        prefactor         = complex(name_list[0])
    else:
        raise ValueError('Invalid name for an operator string: {}'.format(op_string_name))

    if orbital_operators[-1] in PAULI_OPS:
        deduced_op_type = 'Pauli'
    elif orbital_operators[-1] in MAJORANA_OPS:
        deduced_op_type = 'Majorana'
    elif orbital_operators[-1] in FERMION_OPS:
        deduced_op_type = 'Fermion'
    else:
        raise ValueError('Invalid name for an operator string: {}'.format(orbital_operators[-1]))

    if op_type is not None and deduced_op_type != op_type:
        raise ValueError('The input op_type and the deduced op_type of the OperatorString do not agree: {} {}'.format(op_type, deduced_op_type))
        
    return OperatorString(orbital_operators, orbital_labels, deduced_op_type, prefactor=prefactor)

def _fermion_string_from_cdag_c_labels(prefactor, c_dag_labels, c_labels):
    # Construct a fermion string operator from the labels of the creation and
    # anhillation (c^\dagger and c) operators.
    
    c_labels_reversed = np.copy(c_labels)
    c_labels_reversed = c_labels_reversed[::-1]
    
    orbital_operators = ['CDag']*len(c_dag_labels) + ['C']*len(c_labels)
    orbital_labels    = c_dag_labels + c_labels_reversed
    
    return OperatorString(orbital_operators, orbital_labels, prefactor=prefactor, op_type='Fermion')

def _convert_majorana_string(op_string, include_identity=False):
    # Converts a Majorana StringOperator to an Operator
    # that is a linear combination of Fermion StringOperators.
    
    if op_string.op_type is not 'Majorana':
        raise ValueError('Trying to convert a Majorana string to a Fermion string but given an OperatorString of type {}'.format(op_type))
    
    ops    = op_string.orbital_operators
    labels = op_string.orbital_labels

    # The identity operator.
    if len(ops) == 0:
        return Operator([1.0], [OperatorString([], [], 'Fermion')])
    
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
        if len(fermion_labels) == 0:
            if include_identity:
                coeffs_fermion.append(coeff)
                op_strings_fermion.append(OperatorString([], [], 'Fermion'))
            else:
                continue
        
        (sorted_fermion_labels, sign) = tools.sort_sign(fermion_labels)
        coeff *= sign
        
        # The i_1,\ldots,i_m labels
        cdag_labels = sorted_fermion_labels[0:num_cdags]
        # The j_1,\ldots,j_m labels
        c_labels    = large_number - sorted_fermion_labels[num_cdags:]
        c_labels    = c_labels[::-1]

        lex_order = tools.compare(cdag_labels, c_labels)
        
        # Resulting operator is not lexicographically sorted. Ignore it.
        if lex_order < 0:
            continue
        # Resulting operator is of type 1: c^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{i_m} \cdots c_{i_1}. 
        elif lex_order == 0:
            coeffs_fermion.append(coeff)
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1.0, cdag_labels, c_labels))
        # Resulting operator is of type 2: c^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{j_l} \cdots c_{j_1} + H.c.
        elif lex_order > 0 and np.abs(np.imag(coeff)) < threshold:
            coeffs_fermion.append(np.real(coeff))
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1.0, cdag_labels, c_labels))
        # Resulting operator is of type 3: ic^\dagger_{i_1} \cdots c^\dagger_{i_m} c_{j_l} \cdots c_{j_1} + H.c.
        elif lex_order > 0 and np.abs(np.real(coeff)) < threshold:
            coeffs_fermion.append(np.real(coeff/(1j)))
            op_strings_fermion.append(_fermion_string_from_cdag_c_labels(1j, cdag_labels, c_labels))
        else:
            raise ValueError('Invalid lex_order = {} and coeff = {}'.format(lex_order, coeff))
    
    return (coeffs_fermion, op_strings_fermion)


def _convert_fermion_string(op_string):
    # Converts a Fermion OperatorString into a linear
    # combination of Majorana OperatorStrings.

    cdag_labels = [o_lab for (o_lab, o_op) in zip(op_string.orbital_labels, op_string.orbital_operators) if o_op == 'CDag']
    c_labels    = [o_lab for (o_lab, o_op) in zip(op_string.orbital_labels, op_string.orbital_operators) if o_op == 'C']
    
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
    (sorted_fermion_labels, sign) = tools.sort_sign(fermion_labels)

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
        coeffM  = op_string.prefactor * sign
        opNameM = ''
        orbital_operators = []
        orbital_labels    = []
        for indOp in range(num_ops):
            (op, op_label) = ops[indOp]

            if op == 'CDag' and term_choice[indOp] == 0:
                coeffM *= 0.5
                orbital_operators.append('A')
                orbital_labels.append(op_label)
            elif op == 'CDag' and term_choice[indOp] == 1:
                coeffM *= -0.5j
                orbital_operators.append('B')
                orbital_labels.append(op_label)
            elif op == 'C' and term_choice[indOp] == 0:
                coeffM *= 0.5
                orbital_operators.append('A')
                orbital_labels.append(op_label)
            elif op == 'C' and term_choice[indOp] == 1:
                coeffM *= 0.5j
                orbital_operators.append('B')
                orbital_labels.append(op_label)
            elif op == 'CDagC' and term_choice[indOp] == 0:
                coeffM *= -0.5
                orbital_operators.append('D')
                orbital_labels.append(op_label)
            elif op == 'CDagC' and term_choice[indOp] == 1:
                coeffM *= 0.5
            else:
                raise ValueError('Invalid op and term_choice: {} {}'.format(op, term_choice[indOp]))

        # Ignore the identity operator.
        if len(orbital_operators) == 0 and ignore_identity:
            continue

        op_string_M = OperatorString(orbital_operators, orbital_labels, 'Majorana')
        
        coeffM /= op_string_M.prefactor

        # The type 2 and 3 fermion strings have a Hermitian conjugate: (CDag ... C ...) + H.c.,
        # that ensures that the resulting operators are Hermitian. In our conversion,
        # if an operator ends up being anti-Hermitian, then it cancels with the Hermitian conjugate.
        # Otherwise, its coefficient doubles because it equals the Hermitian conjugate.
        if c_dag_labels != c_labels:
            if np.abs(np.imag(coeffM)) > threshold:
                continue
            else:
                coeffM *= 2.0

        coeffs_majorana.append(coeffM)
        op_strings_majorana.append(op_string_M)

    return (coeffs_majorana, op_strings_majorana)

def _convert_operator_string(op_string, to_op_type):
    # Converts an OperatorString to a linear combination of
    # OperatorStrings of the given op_type.
    
    if op_string.op_type == to_op_type:
        return ([1.0], [op_string])
    
    if op_string.op_type == 'Majorana' and to_op_type == 'Fermion':
        return _convert_majorana_string(op_string)
    elif op_string.op_type == 'Fermion' and to_op_type == 'Majorana':
        return _convert_fermion_string(op_string)
    else:
        raise NotImplementedError('Unsupported conversion of OperatorString of op_type {} to {}'.format(op_string.op_type, to_op_type))
