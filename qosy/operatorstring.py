#!/usr/bin/env python
import numpy as np
import itertools as it
from .config import *

class OperatorString:
    """An OperatorString :math:`\hat{h}_a` represents a string 
    of operators that act on orbital degrees of freedom.

    Currently, there are three types of OperatorStrings
    supported in `qosy`: *Pauli strings*, *Fermion strings*,
    and *Majorana strings*.

    A *Pauli string* is a tensor product of spin-1/2 Pauli operators:

        :math:`\hat{h}_a = \hat{\sigma}_1^{t_1}\otimes \cdots \otimes \hat{\sigma}_n^{t_n}`

    where :math:`\hat{\sigma}^0=\hat{I}`, :math:`\hat{\sigma}^a` 
    are the Pauli matrices, and :math:`a` is an index assigned
    to a particular choice of :math:`t_1,\ldots,t_n \in \{0,1,2,3\}`.

    A *Fermion string* is a product of Fermionic operators. It comes
    in three types:

        :math:`\hat{c}^\dagger_{i_1} \cdots \hat{c}^\dagger_{i_m} \hat{c}_{i_m} \cdots \hat{c}_{i_1}`

        :math:`\hat{c}^\dagger_{i_1} \cdots \hat{c}^\dagger_{i_m} \hat{c}_{j_l} \cdots \hat{c}_{j_1} + \\textrm{H.c.}`

        :math:`i\hat{c}^\dagger_{i_1} \cdots \hat{c}^\dagger_{i_m} \hat{c}_{j_l} \cdots \hat{c}_{j_1} + \\textrm{H.c.}`

    where :math:`(i_1,\ldots,i_m) > (j_1,\ldots,j_l)` are 
    lexicographically ordered orbital labels.

    A *Majorana string* is another product of Fermionic 
    operators built out of Majorana fermions

        :math:`\hat{a}_i = \hat{c}_i + \hat{c}_i^\dagger, \quad \hat{b}_i = -i(\hat{c}_i-\hat{c}^\dagger_i), \quad \hat{d}_i = -i \hat{a}_i \hat{b}_i = \hat{I} - 2\hat{c}^\dagger_i \hat{c}_i`.

    They take the form:

        :math:`\hat{h}_a = i^{\sigma_a} \hat{\\tau}_1^{t_1}\cdots \hat{\\tau}_n^{t_n}`
    
    where :math:`(\hat{\\tau}_i^0,\hat{\\tau}_i^1,\hat{\\tau}_i^2,\hat{\\tau}_i^3)=(\hat{I},\hat{a}_i,\hat{b}_i,\hat{d}_i)` 
    and :math:`\sigma_a \in \{0,1\}` is chosen to make the
    operator Hermitian.

    Attributes
    ----------
    orbital_operators : list of str
        A list of the names of the operators acting on the orbitals. 
        Ex: 'X','Y', or 'Z' for Pauli strings.
    orbital_labels : list or ndarray of int
        A list or array of the (unique) integer label of each orbital.
    op_type : str
        Specifies the type of OperatorString considered: 'Pauli', 
        'Majorana', or 'Fermion'.
    name : str
        The string representation of the OperatorString, used
        for hashing.
    prefactor : complex
        A prefactor multiplying the OperatorString. Needed to ensure
        Hermiticity.
    
    Examples
    --------
    The Pauli string representing a transverse field acting on
    orbital `1`, :math:`\hat{\sigma}_1^x`, can be constructed with
        >>> qosy.opstring('X 1')
    The Fermion string representing spin-less superconducting 
    pairing between orbitals `1` and `2`, :math:`\hat{c}^\dagger_1 \hat{c}^\dagger_2 + H.c.`,
    can be constructed with
        >>> qosy.opstring('CDag 1 CDag 2')
    The Majorana string representing the Fermion parity operator
    on orbitals `1,2,3,4`, :math:`(-1)^{\hat{N}}=\hat{d}_1\hat{d}_2\hat{d}_3\hat{d}_4`,
    can be constructed with
        >>> qosy.opstring('D 1 D 2 D 3 D 4')

    Notes
    -----
    OperatorStrings are Hermitian by construction. Moreover, 
    all Pauli strings and Majorana strings, except for the 
    identity, are traceless. Moreover, Pauli strings and 
    Majorana strings are orthonormal with respect to the 
    Hilbert-Schmidt inner product, making them particularly
    convenient for calculations.
    """
    
    def __init__(self, orbital_operators, orbital_labels, op_type, prefactor=1.0):
        """Construct an OperatorString object that represents
        a string of operators that act on orbital degrees of freedom.
        
        Parameters
        ----------
        orbital_operators : list or ndarray of str
            A list of the names of the operators on the orbitals. 
            Ex: 'X','Y', or 'Z' for Pauli strings.
        orbital_labels : list or ndarray of int
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

        self.orbital_operators = np.array(orbital_operators, dtype=str)
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
        """Return a human-readable string representation of the
        OperatorString.

        Returns
        -------
        str

        Examples
        --------
        >>> op_string = qosy.Operator(['X', 'Y', 'Z'], [1, 2, 3], 'Pauli')
        >>> print(op_string) # 'X 1 Y 2 Z 3 '
        >>> op_string = qosy.Operator(['CDag', 'CDag', 'C'], [1, 2, 3], 'Fermion', prefactor=1j)
        >>> print(op_string) # '1j CDag 1 CDag 2 C 3 '
        """
        
        return self.name
        
    def __hash__(self):
        """Return a hash code for the OperatorString. 

        For hashing in python dictionaries.

        Returns
        -------
        int
        """
        return hash(self.name)

    def __eq__(self, other):
        """Check equality between this OperatorString and another.

        Parameters
        ----------
        other : OperatorString
            OperatorString to compare against.

        Returns
        -------
        bool
            True if equal, False otherwise.

        Examples
        --------
        >>> op_string1 = qosy.Operator(['X', 'Y', 'Z'], [1, 2, 3], 'Pauli')
        >>> op_string2 = qosy.Operator(['Z', 'Z', 'Z'], [1, 2, 3], 'Pauli')
        >>> op_string1 == op_string2 # False
        >>> op_string3 = qosy.opstring('X 1 Y 2 Z 3')
        >>> op_string1 == op_string3 # True
        """
        
        equals = (len(self.orbital_operators) == len(other.orbital_operators)) and (self.orbital_operators == other.orbital_operators).all() and (self.orbital_labels == other.orbital_labels).all() and (self.op_type == other.op_type)
        
        return equals

def opstring(op_string_name, op_type=None):
    """Construct an OperatorString from a python string description.
    
    Parameters
    ----------
    op_string_name : str
        The string representing the OperatorString. If no prefactor
        is provided, it is assumed to be 1.0. (Note: for Majorana strings,
        the prefactor is automatically computed, so does not need to be
        initialized by hand.)
    
    Returns
    -------
    OperatorString
        The corresponding OperatorString object.

    Examples
    --------
    The Pauli string :math:`\hat{\sigma}^x_1 \hat{\sigma}^y_2 \hat{\sigma}^z_4` can 
    be constructed with
    >>> qosy.opstring('X 1 Y 2 Z 4')
    or
    >> qosy.opstring('1.0 X 1 Y 2 Z 4')
    The Majorana string :math:`i \hat{a}_1 \hat{b}_3 \hat{d}_5 \hat{d}_6` can be constructed with
    >>> qosy.opstring('1j A 1 B 3 D 5 D 6')
    or simply
    >>> qosy.opstring('A 1 B 3 D 5 D 6')
    The Fermion string :math:`\hat{c}_1^\dagger \hat{c}_2^\dagger \hat{c}_3 + H.c.` 
    can be constructed with
    >>> qosy.opstring('CDag 1 CDag 2 C 3')
    The Fermion string :math:`i \hat{c}_1^\dagger \hat{c}_2^\dagger \hat{c}_3 + H.c.` 
    can be constructed with
    >>> qosy.opstring('1j CDag 1 CDag 2 C 3')
    An identity operator :math:`\hat{I}` that can be used with Majorana 
    strings can be constructed with the special syntax
    >>> qosy.opstring('I','Majorana')
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
