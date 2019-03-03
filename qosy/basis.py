#!/usr/bin/env python
"""
This module defines a Basis (of OperatorStrings) and an Operator object.
These objects are convenient containers for OperatorStrings that are
designed to handle the indexing and book-keeping necessary for calculations
in ``qosy``.

It also provides methods for constructing useful bases of OperatorStrings,
that can be used, e.g., to automatically construct bases of local operators
on arbitrary lattices.

"""

import warnings
import numpy as np
import scipy.sparse as ss
import itertools as it
import copy

from .config import *
from .operatorstring import OperatorString

from .tools import maximal_cliques, compare, cmp_to_key

class Basis:
    """A Basis object represents a basis of operators
    spanned by OperatorStrings :math:`\\hat{h}_a`.
        
    Attributes
    ----------
    op_strings : list of OperatorStrings
        The list of OperatorStrings :math:`\\hat{h}_a` to store in the Basis.
    """
    
    def __init__(self, op_strings=None):
        """Construct an object that represents an ordered 
        Basis of OperatorStrings.
        
        Parameters
        ----------
        op_strings : list of OperatorStrings, optional
            The Basis is created from this list of OperatorStrings.
            If None are specified, then creates an empty Basis by
            default.

        """
        
        if op_strings is None:
            op_strings = []
        
        self.op_strings = op_strings

        # A dictionary that maps each OperatorString
        # to its index in the list op_strings.
        self._indices = dict()
        
        for ind_os in range(len(self.op_strings)):
            op_string = self.op_strings[ind_os]
            self._indices[op_string] = ind_os

        if len(self._indices) != len(self.op_strings):
            raise ValueError('Tried to create a Basis with multiple copies of the same OperatorString. There were {} OperatorStrings provided, but only {} uniques ones.'.format(len(self.op_strings),len(self._indices)))

    def index(self, op_string):
        """Return the index of the OperatorString in the Basis.

        Parameters
        ----------
        op_string : OperatorString
            The OperatorString to find in the Basis.

        Returns
        -------
        int
            The index of the OperatorString in the Basis.

        Raises
        ------
        KeyError
            When the OperatorString is not in the Basis.
        """
        return self._indices[op_string]

    def __iter__(self):
        """Return an iterator over the OperatorStrings in the Basis in order.

        Returns
        -------
        iterator of OperatorStrings
            An iterator over the OperatorStrings in the Basis in order.

        Examples
        --------
            >>> for op_string in basis:
            >>>     print(op_string)
        """
        
        return iter(self.op_strings)

    def __getitem__(self, index):
        """Get the OperatorString at the given index.

        Parameters
        ----------
        index : int
            The index to query.

        Returns
        -------
        OperatorString
            The OperatorString at the given index.

        Examples
        --------
        To get the last element of a basis:
            >>> op_string = basis[-1]
        """
        
        return self.op_strings[index]

    def __len__(self):
        """Compute the size (dimensionality) of the basis.

        Returns
        -------
        int
            The size of the basis.

        Examples
        --------
            >>> dim_basis = len(basis)
        """
        return len(self.op_strings)

    def __str__(self):
        """Return a string representation of the Basis in human-readable format.

        Returns
        -------
        str

        Examples
        --------
            >>> print(basis) # Prints a string representation of the Basis
        """

        list_strings = []
        for ind_os in range(len(self.op_strings)):
            os = self.op_strings[ind_os]
            list_strings += [str(os), '\n']

        result = ''.join(list_strings)
        return result

    def __contains__(self, item):
        """Check if an OperatorString is in the Basis.

        Parameters
        ----------
        item : OperatorString
        
        Returns
        -------
        bool

        Examples
        --------
        To check if the identity operator is in a basis
        of Pauli strings
            >>> identity = qosy.opstring('I', 'Pauli')
            >>> print(identity in basis)
        """
        
        return item in self._indices

    def __add__(self, other):
        """Create a new Basis with an additional OperatorString
        or many new OperatorStrings provided by another Basis.

        Parameters
        ----------
        other : OperatorString or Basis
            The OperatorString or Basis of OperatorStrings to add
            to the current Basis. Any repeated OperatorStrings will not
            be double-counted.

        Returns
        -------
        Basis
            The enlarged Basis.

        Examples
        --------
        You can combine two bases together with
            >>> combined_basis = basis1 + basis2
        """
        
        if isinstance(other, Basis):
            new_op_strings = [os for os in other.op_strings if os not in self]
        elif isinstance(other, OperatorString):
            if other not in self:
                new_op_strings = [other]
            else:
                new_op_strings = []
        else:
            raise TypeError('Cannot add object of type {} to basis.'.format(type(other)))

        return Basis(self.op_strings + new_op_strings)

    def __iadd__(self, other):
        """Add an OperatorString or Basis of OperatorStrings
        in-place to the Basis.

        Parameters
        ----------
        other : OperatorString or Basis
            The OperatorString or Basis of OperatorStrings to add
            to the current Basis. Any repeated OperatorStrings will 
            not be double-counted.

        Examples
        --------
        To incrementally grow a basis, you can initialize
        an empty basis and add elements as follows:
            >>> basis1 = qosy.Basis()
            >>> basis1 += qosy.opstring('X 1 X 2')
            >>> basis1 += qosy.opstring('Y 1 Z 3')
            >>> basis2 = qosy.Basis()
            >>> basis2 += basis1
        """
        
        if isinstance(other, Basis):
            ind = len(self.op_strings)
            for os in other.op_strings:
                if os not in self:
                    self.op_strings.append(os)
                    self._indices[os] = ind
                    ind += 1
        elif isinstance(other, OperatorString):
            ind = len(self._indices)
            if other not in self:
                self.op_strings.append(other)
                self._indices[other] = ind
        else:
            raise TypeError('Cannot add object of type {} to basis.'.format(other.__class__.__name__))

        return self

    def __eq__(self, other):
        """Check if two Bases are equal.

        Parameters
        ----------
        other : Basis
            Basis to compare against.
        
        Returns
        -------
        bool
            True if they are equal and False otherwise.
            Order of OperatorStrings matters for equality.
        """
        
        return self.op_strings == other.op_strings and self._indices == other._indices

class Operator:
    """An Operator object represents a quantum operator
    :math:`\\hat{\\mathcal{O}}=\\sum_a g_a \\hat{h}_a` that is a linear
    combination of OperatorStrings :math:`\\hat{h}_a`.
        
    Attributes
    ----------
    coeffs : list or ndarray of complex
        The coefficients :math:`g_a` in front of the 
        OperatorStrings :math:`\\hat{h}_a`. If all 
        coefficients are real, then the operator
        is Hermitian.
    op_strings : list of OperatorStrings
        The OperatorStrings :math:`\\hat{h}_a` that 
        make up this operator.
    op_type : str
        The type of OperatorStrings that make up the 
        operator: 'Pauli', 'Majorana', or 'Fermion'.
        Deduced from op_strings if non-empty.

    Examples
    --------
    To construct a zero operator to use with Pauli strings:
        >>> zero_op = qosy.Operator(op_type='Pauli')
    To construct a Heisenberg bond between orbitals 1 and 2:
        >>> xx = qosy.opstring('X 1 X 2')
        >>> yy = qosy.opstring('Y 1 Y 2')
        >>> zz = qosy.opstring('Z 1 Z 2')
        >>> bond = qosy.Operator(0.25*numpy.ones(3), [xx,yy,zz])
    """
    
    def __init__(self, coeffs=None, op_strings=None, op_type=None):
        """Construct an object that represents a quantum operator.
        It is a linear combination of operator strings.
        
        Parameters
        ----------
        coeffs : ndarray of complex, optional
            The coefficients in front of the OperatorStrings.
            If all coefficients are real, then the operator
            is Hermitian.
        op_strings : list of OperatorStrings, optional
            The OperatorStrings that make up this operator.
        op_type : str, optional
            The type of OperatorStrings that make up the 
            operator: 'Pauli', 'Majorana', or 'Fermion'.
            Deduced from op_strings if non-empty.

        Examples
        --------
        To construct a zero operator to use with Pauli strings:
        >>> zero_op = qosy.Operator(op_type='Pauli')
        To construct a Heisenberg bond between orbitals 1 and 2:
        >>> xx = qosy.opstring('X 1 X 2')
        >>> yy = qosy.opstring('Y 1 Y 2')
        >>> zz = qosy.opstring('Z 1 Z 2')
        >>> bond = qosy.Operator(0.25*numpy.ones(3), [xx,yy,zz])
        """

        if coeffs is None:
            self.coeffs = np.array([], dtype=complex)
        else:
            self.coeffs = np.array(coeffs, dtype=complex)
            
        if op_strings is None:
            op_strings = []

        self._basis  = Basis(list(op_strings)) # Copies the op_strings list.
        self.op_type = op_type

        if len(self._basis.op_strings) != 0:
            self.op_type = self._basis.op_strings[0].op_type
        elif len(self._basis.op_strings) == 0 and op_type is None:
            raise ValueError('Cannot create an empty (zero) operator without specifying the op_type.')

    def remove_zeros(self, tol=1e-12):
        """Remove OperatorStrings with zero coefficients
        from the Operator.

        Returns
        -------
        Operator
            A new Operator with the zero-coefficient 
            OperatorStrings removed.

        Examples
        --------
            >>> cleaned_operator = operator.remove_zeros()
        """
        
        inds_nonzero = np.where(np.abs(self.coeffs) > tol)[0]

        # The non-zero coefficients.
        new_coeffs = self.coeffs[inds_nonzero]
        
        # The non-zero operator strings.
        new_op_strings = [self._basis.op_strings[ind] for ind in inds_nonzero]
        
        # Create a new operator with copies of the old
        # operators' information, excluding the zeros.
        new_operator = Operator(new_coeffs, new_op_strings, op_type=self.op_type)

        return new_operator

    def norm(self, order=None):
        """Compute the Hilbert-Schmidt norm of the Operator. 

        For an Operator :math:`\\hat{\\mathcal{O}}=\\sum_a g_a \\hat{h}_a`  
        made of orthonormal OperatorStrings :math:`\\hat{h}_a` 
        that satisfy :math:`\\langle \\hat{h}_a, \\hat{h}_b\\rangle = \\textrm{tr}(\\hat{h}_a^\\dagger \\hat{h}_b)/\\textrm{tr}(\\hat{I}) =\\delta_{ab}`,
        the (squared) Hilbert-Schmidt norm is
            :math:`||\\hat{\\mathcal{O}}||^2 = \\sum_a g_a^2`
        which is just the :math:`\\ell_2`-norm of the :math:`g_a`
        vector.

        Parameters
        ----------
        order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (see `numpy.norm`). Defaults to None,
            which is the :math:`\\ell_2`-norm. Another useful norm
            is the `inf` order norm, which returns the maximum
            :math:`|g_a|` value.

        Returns
        -------
        float
            The Hilbert-Schmidt norm of the Operator.

        Examples
        --------
        >>> operator = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator.norm()          # sqrt(2) = 1.414213
        >>> operator.norm(order=numpy.inf) # 1.0
        """

        if len(self._basis) > 0 and self._basis[0].op_type == 'Fermion':
            warnings.warn('Computing the normalization of Operators made of Fermion strings is not supported yet. Fermion strings do not form an orthonormal basis, so one would need to compute an overlap matrix. The norm is only reliable for zero operators.')
        
        return np.linalg.norm(self.coeffs, ord=order)

    def normalize(self, order=None):
        """Normalize the Operator to have unit norm.

        Parameters
        ----------
        order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm (see `numpy.norm`). Defaults to None,
            which is the :math:`\\ell_2`-norm. Another useful norm
            is the `inf` order norm, which returns the maximum
            :math:`|g_a|` value.

        Examples
        --------
        >>> operator = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator.normalize()
        >>> operator.norm() # 1.0
        """
        
        self.coeffs /= self.norm(order=order)

    def to_vector(self, basis, fmt='numpy'):
        """Convert the Operator to a vector
        in the given Basis of OperatorStrings.

        Parameters
        ----------
        basis : Basis
            The Basis of OperatorStrings in which
            to represent the Operator as a vector.
        fmt : str
            The array format to return the vector
            as. Specifying 'numpy' returns a ndarray
            of shape (n,); 'csc' returns a scipy.sparse.csc_matrix
            of shape (n,1). Defaults to 'numpy'.

        Returns
        -------
        ndarray or scipy.sparse.csc_matrix
            The vector representation of the Operator.
        """

        if fmt == 'numpy':
            vector = np.zeros(len(basis), dtype=complex)
            
            for (coeff, op_string) in self:
                ind_vec = basis.index(op_string)
                vector[ind_vec] = coeff

            return vector
        elif fmt == 'csc':
            row_inds = []
            data     = []
            
            for (coeff, op_string) in self:
                ind_vec = basis.index(op_string)
                row_inds.append(ind_vec)
                data.append(coeff)

            col_inds = [0]*len(row_inds)
            vector = ss.csc_matrix((data, (row_inds, col_inds)), shape=(len(basis),1), dtype=complex)
            
            return vector
        else:
            raise ValueError('Cannot create a vector representation of the Operator with fmt {}'.format(fmt))    
                
    def __add__(self, other):
        """Add an Operator to this Operator.

        Parameters
        ----------
        other : Operator
            The Operator to add to this one.

        Returns
        -------
        Operator
            The sum of the two Operators.

        Examples
        --------
        >>> operator1 = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator2 = qosy.Operator([1.0], [qosy.opstring('Z 1')])
        >>> operator1 + operator2 # X_1 + Y_1 + Z_1
        """
        
        if self.op_type != other.op_type:
            raise ValueError('Cannot add operator of op_type {} to operator of op_type {}'.format(self.op_type, other.op_type))

        # Create a new operator with copies of the old
        # operators' information.
        new_operator = Operator(np.copy(self.coeffs), copy.deepcopy(self._basis.op_strings), op_type=self.op_type)

        # Enlarge the basis.
        new_operator._basis += other._basis

        # Pad the coefficients vector with zeros
        # to match the new basis size.
        new_operator.coeffs = np.pad(new_operator.coeffs, (0,len(new_operator._basis)-len(self._basis)), 'constant')

        # Collect the indices of the other Operator
        # in the new enlarged basis.
        inds_new_basis = np.array([new_operator._basis.index(op_string) for op_string in other._basis], dtype=int)

        # Add the other Operator's coefficients
        # to the enlarged coefficients vector.
        new_operator.coeffs[inds_new_basis] += other.coeffs

        return new_operator

    def __iadd__(self, other):
        """Add an Operator in-place to this Operator.

        Parameters
        ----------
        other : Operator
            The Operator to add to this one.

        Examples
        --------
        >>> operator1 = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator2 = qosy.Operator([1.0], [qosy.opstring('Z 1')])
        >>> operator1 += operator2 # X_1 + Y_1 + Z_1
        """
        
        if self.op_type != other.op_type:
            raise ValueError('Cannot add operator of op_type {} to operator of op_type {}'.format(self.op_type, other.op_type))

        # Enlarge the basis.
        old_len = len(self._basis)
        self._basis += other._basis
        new_len = len(self._basis)

        # Pad the coefficients vector with zeros
        # to match the new basis size.
        self.coeffs = np.pad(self.coeffs, (0,new_len-old_len), 'constant')

        # Collect the indices of the other Operator
        # in the new enlarged basis.
        inds_new_basis = np.array([self._basis.index(op_string) for op_string in other._basis], dtype=int)

        # Add the other Operator's coefficients
        # to the enlarged coefficients vector.
        self.coeffs[inds_new_basis] += other.coeffs

        return self

    def __mul__(self, other):
        """Compute the product of this Operator and a scalar.

        Parameters
        ----------
        other : float or complex
            The scalar to multiply the Operator.

        Returns
        -------
        Operator
            The Operator with its coefficient's 
            multipled by the scalar.

        Examples
        --------
        >>> operator = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator * 2.0 # 2 (X_1 + Y_1)
        """
        
        return Operator(self.coeffs * other, self._basis.op_strings, self.op_type)

    def __rmul__(self, other):
        """Compute the product of this Operator and a scalar.

        Parameters
        ----------
        other : float or complex
            The scalar to multiply the Operator.

        Returns
        -------
        Operator
            The Operator with its coefficient's 
            multipled by the scalar.

        Examples
        --------
        >>> operator = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> 2 * operator # 2 (X_1 + Y_1)
        """
        
        return Operator(self.coeffs * other, self._basis.op_strings, self.op_type)
    
    def __sub__(self, other):
        """Subtract an Operator from this Operator.

        Parameters
        ----------
        other : Operator
            The Operator to subtract from this one.

        Returns
        -------
        Operator
            The difference of the two Operators.

        Examples
        --------
        >>> operator1 = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator2 = qosy.Operator([1.0], [qosy.opstring('Z 1')])
        >>> operator1 - operator2 # X_1 + Y_1 - Z_1
        """
        
        return self.__add__(-1.0*other)

    def __isub__(self, other):
        """Subtract an Operator in-place from this Operator.

        Parameters
        ----------
        other : Operator
            The Operator to subtract from this one.

        Examples
        --------
        >>> operator1 = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
        >>> operator2 = qosy.Operator([1.0], [qosy.opstring('Z 1')])
        >>> operator1 -= operator2 # X_1 + Y_1 - Z_1
        """
        
        return self.__iadd__(-1.0*other)

    def __neg__(self):
        """Return the negation of the Operator.

        Returns
        -------
        Operator
            The Operator with its coefficients negated.
        """

        return -1.0*self
        
    def __iter__(self):
        """Return an iterator over the (coefficient, OperatorString)
        pairs of the Operator.

        Returns
        -------
        iterator of (number, OperatorString)
        
        Examples
        --------
            >>> operator = qosy.Operator([1.0, -1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
            >>> positive_ops = [op_string for (coeff, op_string) in operator if coeff > 0.0] # [X_1]
        """
        
        return iter(zip(self.coeffs, self._basis.op_strings))
    
    def __len__(self):
        """Return the number of OperatorStrings that make up
        the Operator.

        Returns
        -------
        int
            The number of OperatorStrings making up the Operator.

        Examples
        --------
            >>> operator = qosy.Operator([1.0, -1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
            >>> len(operator) # 2
        Note that zero-coefficient OperatorStrings are still counted:
            >>> operator = qosy.Operator([1.0, 0.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
            >>> len(operator) # 2
        """
        
        return len(self._basis)
    
    def __str__(self):
        """Return the python string representation of the Operator
        in human-readable form.

        Returns
        -------
        str
            The string repesentation of the Operator.

        Examples
        --------
            >>> operator = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
            >>> print(operator)
        """
        
        list_strings = []
        for ind_coeff in range(len(self.coeffs)):
            coeff = self.coeffs[ind_coeff]
            os    = self._basis.op_strings[ind_coeff]
            list_strings += [str(coeff), ' (', str(os), ')\n']

        result = ''.join(list_strings)
        return result

    def __eq__(self, other):
        """Checks if an Operator is equivalent to this one.

        Parameters
        ----------
        other : Operator
            The Operator to compare against.

        Returns
        -------
        bool
            True if equal, False otherwise. Note that
            the order of OperatorStrings matters.

        Examples
        --------
            >>> operator1 = qosy.Operator([1.0, 1.0], [qosy.opstring('X 1'), qosy.opstring('Y 1')])
            >>> operator2 = qosy.Operator([1.0, 1.0], [qosy.opstring('Y 1'), qosy.opstring('X 1')])
            >>> operator1 == operator2 # False
        
        Notes
        -----
        There is an alternative less-strict way to check 
        equality betweeen two Operators. This alternative 
        approach checks if they are made of the same linear
        combination of OperatorStrings by taking their 
        difference:
         
            >>> (operator1 - operator2).norm() < 1e-12 # True
        """
        
        return self.op_type == other.op_type and (len(self.coeffs) == len(other.coeffs)) and (self.coeffs == other.coeffs).all() and self._basis == other._basis
    
def cluster_basis(k, cluster_labels, op_type, include_identity=False):
    """Constructs a Basis of OperatorStrings from the labels
    of a "cluster" of orbitals. The OperatorStrings in the cluster Basis
    are all possible combinations of up to k-local OperatorStrings on the 
    cluster.

    Parameters
    ----------
    k : int or list or ndarray of int
        Specifies the allowed support of OperatorStrings in the
        Basis. If k is an integer, then OperatorStrings can have 
        support up to k. If a list or array, then OperatorStrings
        can only have support on the given numbers of orbitals.
    cluster_labels : list or ndarray of int
        The integer labels of the orbitals that are in the cluster.
    include_identity : bool, optional
        Specifies whether to include the identity operator in the Basis.
        The default is to not include it to keep the Basis's OperatorStrings 
        traceless.

    Returns
    -------
    Basis
        The Basis of OperatorStrings.

    Examples
    --------
    To construct all possible (traceless) one and two-site operators 
    made of Pauli strings on sites 1, 2, and 4, one can use
        >>> basis1 = qosy.cluster_basis(2, [1,2,4], 'Pauli')
    To construct `only` two-site operators on these orbitals, one can use
        >>> basis2 = qosy.cluster_basis([2], [1,2,4], 'Pauli')
    """
    
    # The labels of the cluster in sorted order.
    cluster = copy.deepcopy(cluster_labels)
    cluster.sort()
    
    op_strings     = []
    op_strings_set = set()
    
    if include_identity:
        identity = OperatorString([], [], op_type) # Identity operator
        op_strings.append(identity)
        op_strings_set.add(identity)

    if type(k) is int:
        allowed_num_operators = np.arange(k+1)
    elif type(k) is list or isinstance(k, np.ndarray):
        allowed_num_operators = np.array(k, dtype=int)
    else:
        raise ValueError('k={} must be an integer or a list or array of integers.'.format(k))
        
    max_num_operators = np.minimum(len(cluster), np.max(allowed_num_operators))

    if op_type == 'Pauli' or op_type == 'Majorana':
        orbital_ops = PAULI_OPS
        if op_type == 'Majorana':
            orbital_ops = MAJORANA_OPS
        
        for num_operators in np.arange(1,max_num_operators+1):
            if num_operators not in allowed_num_operators:
                continue
            
            possible_ops    = list(it.product(orbital_ops, repeat=num_operators))
            possible_labels = list(it.combinations(cluster, num_operators))
            
            for labels in possible_labels:
                for ops in possible_ops:
                    op_string = OperatorString(list(ops), list(labels), op_type)
                    if op_string not in op_strings_set:
                        op_strings.append(op_string)
                        op_strings_set.add(op_string)
    elif op_type == 'Fermion':
        for num_operators_forward in np.arange(1,max_num_operators+1):
            if num_operators_forward not in allowed_num_operators:
                continue
            
            possible_labels_forward = list(it.combinations(cluster, num_operators_forward))
            
            # Operator of type 1: c^\dagger_{i_1} ... c^\dagger_{i_m} c_{i_m} ... c_{i_1}
            for labels in possible_labels_forward:
                labels_forward  = np.copy(labels)
                labels_backward = np.copy(labels[::-1])

                ops    = ['CDag']*len(labels_forward) + ['C']*len(labels_backward)
                labels = np.concatenate((labels_forward, labels_backward))

                op_string = OperatorString(ops, labels, op_type)

                if op_string not in op_strings_set:
                    op_strings.append(op_string)
                    op_strings_set.add(op_string)

            # Operator of type 2 and 3: i^s * c^\dagger_{i_1} ... c^\dagger_{i_m} c_{j_l} ... c_{j_1} + H.c.
            for ind_labels_forward in range(len(possible_labels_forward)):
                labels_forward = possible_labels_forward[ind_labels_forward]
                
                for num_operators_backward in np.arange(0,num_operators_forward+1):
                    possible_labels_backward = list(it.combinations(cluster, num_operators_backward))
                    
                    max_ind_labels_backward = len(possible_labels_backward)
                    if num_operators_backward == num_operators_forward:
                        max_ind_labels_backward = ind_labels_forward
                
                    possible_prefactors = [1.0, 1j]
                    for prefactor in possible_prefactors:
                        for ind_labels_backward in range(max_ind_labels_backward):
                            labels_backward = possible_labels_backward[ind_labels_backward]
                            
                            ops    = ['CDag']*len(labels_forward) + ['C']*len(labels_backward)
                            labels = np.concatenate((labels_forward, labels_backward))
                            
                            op_string = OperatorString(ops, labels, op_type, prefactor)

                            if op_string not in op_strings_set:
                                op_strings.append(op_string)
                                op_strings_set.add(op_string)
    else:
        raise ValueError('Unknown operator string type: {}'.format(op_type))
    
    return Basis(op_strings)

def distance_basis(lattice, k, R, op_type, allowed_orbitals=None, include_identity=False, tol=1e-10):
    """Constructs a Basis of OperatorStrings 
    made of orbitals that are spatially local.

    Parameters
    ----------
    lattice : Lattice
        The Lattice whose orbitals we want to use in this basis.
    k : int or list or ndarray of int
        Specifies the allowed support of OperatorStrings in the
        Basis. If k is an integer, then OperatorStrings can have 
        support up to k. If a list or array, then OperatorStrings
        can only have support on the given numbers of orbitals.    
    R : float
        The maximum spatial distance between orbitals in an OperatorString.
    op_type : str
        The type of OperatorStrings in the Basis: 
        'Pauli', 'Fermion', or 'Majorana'.
    allowed_orbitals : list or ndarray of int, optional
        If provided, the OperatorStrings in the Basis must
        be composed of only orbitals with these labels.
        Defaults to None.
    include_identity : bool, optional
        Specifies whether to include the identity operator in the Basis.
        The default is to not include it to keep the Basis's OperatorStrings 
        traceless.
    tol : float, optional
        The tolerance used to compare distances between orbitals. Default
        is 1e-10.

    Returns
    -------
    Basis
        The Basis of OperatorStrings.

    Examples
    --------
    To construct all possible (traceless) 1 and 2-site operators 
    made of nearest and next-nearest neighbor Pauli strings on 
    a periodic chain of twelve sites, one can use
        >>> lattice = qosy.lattice.chain(12, periodic=True)
        >>> basis   = qosy.distances_basis(lattice, 2, 2.0, 'Pauli')
    """
    
    num_orbitals = len(lattice)
    
    if allowed_orbitals is None:
        allowed_orbitals = np.arange(num_orbitals)
    
    # Store the distances between the orbitals into a matrix.
    distances = np.zeros((num_orbitals, num_orbitals))
    for ind1 in range(num_orbitals):
        for ind2 in range(ind1+1,num_orbitals):
            distances[ind1,ind2] = lattice.distance(ind1, ind2)
            distances[ind2,ind1] = distances[ind1,ind2]

    # For each orbital, compute which orbitals are within distance R.
    orbitals_within_R = [[ind2 for ind2 in allowed_orbitals if distances[ind1,ind2] <= R+tol and ind2 != ind1] for ind1 in range(num_orbitals)]

    # The clusters are maximal cliques of the graph
    # with orbitals as nodes and nodes marked as adjacent
    # if they are within a distance R of each other
    # in real space.
    cliques = maximal_cliques(orbitals_within_R)

    # Only include clusters containing allowed orbitals.
    clusters = []
    for clique in cliques:
        allowed = True
        for orbital_label in clique:
            if orbital_label not in allowed_orbitals:
                allowed = False
                break

        if allowed:
            clusters.append(clique)
    
    # Sort the clusters for consistency.
    sorted_clusters = sorted(clusters, key=cmp_to_key(compare))
    
    total_basis = Basis()
    for cluster_labels in sorted_clusters:
        total_basis += cluster_basis(k, cluster_labels, op_type)
        
    return total_basis
