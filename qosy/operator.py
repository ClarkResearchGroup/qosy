#!/usr/bin/env python
import numpy as np
#import lattice

class OperatorString:
    """An OperatorString object represents a string 
    of operators that act on orbital degrees of freedom.
        
    Attributes
    ----------
    orbital_operators : array_like of str
        An array of the names of the operators on the orbitals.
    orbital_labels : array_like of hashable
        An array of the labels identifying each orbital. (Should
        correspond to the labels used to define Lattices.)
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
        """Constructs an OperatorString object that represents
        a string of operators that act on orbital degrees of freedom.
        
        Parameters
        ----------
        orbital_operators : array_like of str
            An array of the names of the operators on the orbitals.
        orbital_labels : array_like of hashable
            An array of the labels identifying each orbital. (Should
            correspond to the labels used to define Lattices.)
        basis_type : str
            Specifies the type of operator string considered: 'Pauli', 
            'Majorana', or 'Fermion'.
        prefactor : number, optional
            A prefactor multiplying the operator to ensure that
            it is Hermitian (is non-trivial for Majorana strings, 
            otherwise is 1).

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
        self.orbital_labels    = orbital_labels
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
            self._labels_ab_operators = [self.orbital_labels[ind] for ind in range(len(self.orbital_labels)) if self.orbital_operators[ind] in ['A', 'B']]
            num_ab = len(self._labels_ab_operators)

            # The prefactor is 1 or I, depending
            # on whether reversing the order of operators creates
            # a +1 or -1 sign due to anti-commutation operators.
            num_swaps_to_reorder = (num_ab*(num_ab-1))/2
            if num_swaps_to_reorder % 2 == 1:
                self.prefactor *= 1j
            
        name_list = [str(self.prefactor),' ']
        for (op, la) in zip(self.orbital_operators, self.orbital_labels):
            name_list.extend([op, ' ', str(la), ' '])

        self.name = ''.join(name_list)

    def __str__(self):
        return self.name
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        equals = (self.orbital_operators == other.orbital_operators) and (self.orbital_labels == other.orbital_labels) and (self.name == other.name) and (self.op_type == other.op_type)
        
        return equals


class Operator:
    def __init__(self, coeffs, op_strings, op_type):
        """Constructs an object that represents a quantum operator.
        It is a linear combination of operator strings.
        
        Parameters
        ----------
        coeffs : array_like of numbers
            The coefficients in front of the pauli strings.
            If all coefficients are real, then the operator
            is Hermitian.
        op_strings : array_like of OperatorStrings
            The operator strings that make up this Operator.
        op_type : str
            The type of operator strings that make up the 
            Operator: 'Pauli', 'Majorana', or 'Fermion'.
        """
        
        self.coeffs     = coeffs
        self.op_strings = op_strings
        self.op_type    = op_type

    def __add__(self, other):
        return Operator(self.coeffs + other.coeffs, self.op_strings, self.op_type)

    def __mul__(self, other):
        return Operator(self.coeffs * other, self.op_strings, self.op_type)

    def __str__(self):
        list_strings = []
        for ind_coeff in range(len(self.coeffs)):
            coeff = self.coeffs[ind_coeff]
            os    = self.op_strings[ind_coeff]
            list_strings += [str(coeff), ' (', str(os), ')\n']

        result = ''.join(list_strings)
        return result

# TODO
def _convert_operator_string(op_string, to_op_type):
    pass

def _convert_operator(operator, to_op_type):
    pass

def convert(operator, to_op_type):
    if type(operator) is OperatorString:
        return _convert_operator_string(operator)
    elif type(operator) is Operator:
        return _convert_operator(operator)
    else:
        raise TypeError('Cannot convert invalid operator type: {}'.format(type(operator)))
    
def heisenberg(lat):
    pass

def zero_mode(lat):
    pass


    
