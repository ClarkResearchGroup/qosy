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
