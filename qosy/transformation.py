#!/usr/bin/env python
"""
This module includes a Transformation object that represents
a discrete transformation, such as charge-conjugation or a
space group symmetry, and a method for computing the effect
of a Transformation on a Basis of OperatorStrings.
"""

import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss

from .tools import sort_sign, swap
from .operatorstring import OperatorString
from .basis import Basis, Operator
from .lattice import Lattice

class Transformation:
    """A Transformation object represents a 
    symmetry transformation that acts on the 
    Hilbert space of states.

    Attributes
    ----------
    rule : Operator = function(OperatorString, info_type)
        The transformation rule is a 
        user-supplied function that takes in
        an OperatorString and additional info
        and returns an Operator representing
        the effect of the transformation on 
        the OperatorString.
    info : info_type
        The information provided to the
        transformation rule function.
    """
    
    def __init__(self, rule, info):
        """Construct a Transformation object that 
        represents a symmetry transformation that
        acts on the Hilbert space of states.

        Parameters
        ----------
        rule : Operator = function(OperatorString, info_type)
            The transformation rule is a 
            user-supplied function that takes in
            an OperatorString and additional info
            and returns an Operator representing
            the effect of the transformation on 
            the OperatorString.
        info : info_type
            The information provided to the
            transformation rule function.
        """
        
        self.rule = rule
        self.info = info

    def apply(self, operator, tol=1e-12):
        """Apply the transformation :math:`\hat{\mathcal{U}}` to an 
        operator string :math:`\hat{h}_a` or operator 
        :math:`\hat{O}=\sum_a J_a \hat{h}_a`.

        Parameters
        ----------
        operator : OperatorString or Operator
            The OperatorString or Operator to 
            apply the transformation to.

        Returns
        -------
        Operator
            The transformed operator string 
                :math:`\hat{\mathcal{U}} \hat{h}_a \hat{\mathcal{U}}^{-1}`
            or operator
                :math:`\sum_a J_a \hat{\mathcal{U}} \hat{h}_a \hat{\mathcal{U}}^{-1}`
            (assumes :math:`J_a` are real).
        """
        
        if isinstance(operator, OperatorString):
            return self.rule(operator, self.info)
        elif isinstance(operator, Operator):
            result = Operator(op_type=operator.op_type)
            for (coeff, op_string) in operator:
                if np.abs(np.imag(coeff)) > tol:
                    raise NotImplementedError('Operators with complex coefficients are currently not supported with transformations (time-reversal might behave incorrectly).')
                
                result += coeff * self.rule(op_string, self.info)
        else:
            raise TypeError('Cannot apply the transformation to object of type {}'.format(type(operator)))

    @staticmethod
    def _product_rule(op_string_A, info, tol=1e-12):
        # Transformation rule for an operator 
        # that is a product of two operators: U = D*C.
        # U \hat{h}_a U^{-1} = D C \hat{h}_a C^{-1} D^{-1}
        
        # The information contains the rules for the C and D operators.
        (ruleC, infoC, ruleD, infoD) = info
        
        result = Operator(op_type=op_string_A.op_type)
        
        operatorC = ruleC(op_string_A, infoC)
            
        for (coeffC, op_string_C) in operatorC:
            operatorD = ruleD(op_string_C, infoD)
            for (coeffD, op_string_D) in operatorD:
                coeffB      = coeffC * coeffD
                op_string_B = op_string_D
                
                if np.abs(coeffB) > tol:
                    result += Operator(np.array([coeffB]), [op_string_B])
                        
        return result

    def __mul__(self, other):
        """Take the product of two Transformations :math:`\hat{\mathcal{C}}` and :math:`\hat{\mathcal{D}}`.
        
        Parameters
        ----------
        other : Transformation
            The transformation :math:`\hat{\mathcal{D}}` to multiply on the right of :math:`\hat{\mathcal{C}}`.

        Returns
        -------
        Transformation
            The resulting Transformation :math:`\hat{\mathcal{C}}\hat{\mathcal{D}}`.

        Examples
        --------
        We can define the chiral symmetry transformation
        as a product of time-reversal and charge-conjugation
            >>> T = qosy.time_reversal()
            >>> C = qosy.charge_conjugation()
            >>> S = T * C # chiral symmetry transformation
        """
        
        product_info = (other.rule, other.info, self.rule, self.info)
        return Transformation(Transformation._product_rule, product_info)      

def _permutation_rule(op_string_A, info):
    """Transformation rule to perform a permutation of orbital labels.
    Transformation for spins:
        \sigma^a_i -> \sum_j U_{ji} \sigma^a_j 
    Transformation for (spinless) fermions:
        c_i -> \sum_j U_{ji} c_j
        a_i -> \sum_j U_{ji} a_j, b_i -> \sum_j U_{ji} b_j, d_i -> \sum_j U_{ji} d_j
    where U_{ji} is a permutation matrix with a single 1 on each row/column.
    """
    if not (op_string_A.op_type == 'Pauli' or op_string_A.op_type == 'Majorana'):
        raise ValueError('Invalid op_type: {}'.format(op_string_A.op_type))

    # The information is a permutation that specifies how labels map into other labels.
    permutation = np.array(info, dtype=int)
    
    coeffB = 1.0
    
    ops    = op_string_A.orbital_operators
    labels = op_string_A.orbital_labels
    
    new_labels_unsorted = np.array([permutation[l] for l in labels], dtype=int)
    inds_sorted         = np.argsort(new_labels_unsorted)

    new_ops     = [ops[ind] for ind in inds_sorted]
    new_labels  = new_labels_unsorted[inds_sorted]

    op_string_B = OperatorString(new_ops, new_labels, op_string_A.op_type)

    if op_string_A.op_type == 'Majorana':
        inds_ab_sorted = [ind for ind in inds_sorted if new_ops[ind] in ['A','B']]
        
        # Make sure to compute the sign acquired from reordering
        # A and B operators in the Majorana strings.
        (_, sign) = sort_sign(inds_ab_sorted)
        
        # Also, account for the prefactors in front of the Majorana
        # string operators before and after the transformation.
        coeffB *= np.real(op_string_A.prefactor/op_string_B.prefactor * sign)     

    return Operator([coeffB], [op_string_B])

def _time_reversal_rule(op_string_A, info, tol=1e-12):
    """Transformation rule for time-reversal symmetry T.
    Transformation for spins:
        \sigma^a_i -> -\sigma^a_i (means T \sigma^a_i T^{-1} = -\sigma^a_i)
    Transformation for (spinless) fermions:
        c_i -> +c_i, c_i^\dagger -> +c_i^\dagger, i -> -i (imaginary number)
        a_i -> +a_i, b_i -> -b_i, d_i -> d_i, i -> -i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if not (op_type == 'Pauli' or op_type == 'Majorana'):
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # No extra information needed.
    
    coeffB      = 1.0
    op_string_B = op_string_A

    if op_type == 'Majorana':        
        numABs = 0
        numBs  = 0
        numDs  = 0
        for op in op_string_A.orbital_operators:
            if op == 'A':
                numABs += 1
            elif op == 'B':
                numBs  += 1
                numABs += 1
            elif op == 'D':
                numDs  += 1
            else:
                raise ValueError('Invalid op name: {}'.format(op))

        # A minus sign occurs for every b_i majorana operator.
        if numBs % 2 == 1:
            coeffB *= -1.0
        # A minus sign occurs if there is an imaginary number in front of the Majorana string.
        if numABs*(numABs-1)//2 % 2 == 1:
            coeffB *= -1.0
    else:
        # A minus sign occurs for every Pauli matrix in the Pauli string.
        num_ops = len(op_string_B.orbital_operators)
        if num_ops % 2 == 1:
            coeffB *= -1.0
           
    return Operator([coeffB], [op_string_B])

def _particle_hole_rule(op_string_A, info):
    """Transformation rule for particle hole symmetry C.
    Transformation for (spinless) fermions:
        c_i -> +c_i^\dagger, c_i^\dagger -> +c_i, i -> i (imaginary number)
        a_i -> +a_i, b_i -> -b_i, d_i -> -d_i, i -> i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if op_type != 'Majorana':
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # No extra information needed.
    
    coeffB      = 1.0
    op_string_B = op_string_A

    numABs = 0
    numBs  = 0
    numDs  = 0
    for op in op_string_A.orbital_operators:
        if op == 'A':
            numABs += 1
        elif op == 'B':
            numBs  += 1
            numABs += 1
        elif op == 'D':
            numDs  += 1
        else:
            raise ValueError('Invalid op name: {}'.format(op))

    # Every b_i operator contributes a minus sign.
    if numBs % 2 == 1:
        coeffB *= -1.0
        
    # Every d_i operator contributes a minus sign.
    if numDs % 2 == 1:
        coeffB *= -1.0
            
    return Operator([coeffB], [op_string_B])

def label_permutation(permutation):
    """Create a Transformation that permutes
    the orbital labels of OperatorStrings.

    The transformation acts on operator strings
    as
       :math:`\hat{h}_a = \hat{O}_{l_1} \cdots \hat{O}_{l_n} \\rightarrow \hat{h}_b = \hat{O}_{\pi(l_1)} \cdots \hat{O}_{\pi(l_n)}`
    where :math:`\pi` is a permutation.

    Parameters
    ----------
    permutation : ndarray or list of int
        The permutation of integer orbital labels.

    Returns
    -------
    Transformation
        The Transformation that implements the
        orbital relabeling.

    Examples
    --------
    Consider a chain of four equally spaced
    orbitals labeled `0,1,2,3`. A reflection
    about the center of the chain can be 
    created with
        >>> R = qosy.label_permutation([3,2,1,0])
    If the chain is periodic, then a 
    translation to the right can be made with
        >>> P = qosy.label_permutation([1,2,3,0])
    """
    return Transformation(_permutation_rule, permutation)

def spin_flip_symmetry(lattice, up_name='Up', dn_name='Dn'):    
    """Create the symmetry that flips all spin-:math:`1/2` 
    degrees of freedom between up and down.

    The effect of this transformation is to relabel all
    orbitals so that :math:`(i,\\uparrow) \\leftrightarrow (i,\\downarrow)`

    Parameters
    ----------
    lattice : Lattice
        The Lattice object that keeps track of the orbitals
        in the system.
    up_name : str, optional
        Specifies the name of the :math:`+1/2` spin. Defaults to 'Up'.
    dn_name : str, optional
        Specifies the name of the :math:`-1/2` spin. Defaults to 'Dn'.

    Returns
    -------
    Transformation
        The discrete spin-flip symmetry transformation.
    """
    
    spinflip_permutation = [lattice.index(r, swap(orbital_name, up_name, dn_name)) \
                            for (r, orbital_name,_) in lattice]

    if -1 in spinflip_permutation:
        raise ValueError('Invalid spin-flip permutation:\n{}'.format(spinflip_permutation))
    
    spinflip = qy.label_permutation(spinflip_permutation)

    return spinflip

def space_group_symmetry(lattice, R, d):
    """Create a space group symmetry specified by
    the given rotation matrix and displacement
    vector.

    The space group symmetry transformation
    relabels the spatial coordinate of an 
    orbital from position :math:`r`
    to :math:`r'` according to
       :math:`r \rightarrow r'=Rr + d.`

    Parameters
    ----------
    lattice : Lattice
        An `M`-dimensional Lattice.
    R : ndarray, (M,M)
        The spatial rotation matrix :math:`R`.
    d : ndarray, (M,)
        The displacement vector :math:`d`.

    Returns
    -------
    Transformation
        The discrete space group symmetry transformation.
    """
    
    orbital_label_permutation = [lattice.index(np.dot(R,r) + d, orbital_name) \
                                 for (r, orbital_name,_) in lattice]

    if -1 in orbital_label_permutation:
        raise ValueError('Invalid permutation of orbital labels:\n{}'.format(orbital_label_permutation))

    symmetry = label_permutation(orbital_label_permutation)
    
    return symmetry

def time_reversal():
    """Create a (spin-less) time-reversal 
    Transformation :math:`\hat{\mathcal{T}}`.
    
    The most important aspect of time reversal
    is that it is antiunitary and acts as a 
    complex conjugation operator:
       :math:`\hat{\mathcal{T}} i \hat{\mathcal{T}}^{-1} = -i \quad (i \\rightarrow -i)`
    
    The transformation acts on (spin-less)
    Fermionic operators as
        :math:`c_j \\rightarrow c_j, \quad c_j^\dagger \\rightarrow  c_j^\dagger`
    and Majorana operators as
        :math:`a_j \\rightarrow a_j, \quad b_j \\rightarrow -b_j, \quad d_j \\rightarrow d_j`

    On spin operators, the transformation acts as:
        :math:`\sigma^a_j \\rightarrow -\sigma^a_j \quad (\hat{\mathcal{T}} \sigma^a_j \hat{\mathcal{T}}^{-1} = -\sigma^a_j)`

    Returns
    -------
    Transformation
        The Transformation that implements
        spin-less time-reversal.

    Examples
    --------
    >>> T = qosy.time_reversal()
    """
    
    return Transformation(_time_reversal_rule, None)

def particle_hole():
    """Create a particle-hole (or charge-conjugation) 
    Transformation :math:`\hat{\mathcal{C}}`.
    
    The transformation acts on (spin-less)
    Fermionic operators as
        :math:`c_j \\rightarrow c_j^\dagger, \quad c_j^\dagger \\rightarrow c_j`
    and Majorana operators as
        :math:`a_j \\rightarrow a_j, \quad b_j \\rightarrow -b_j, \quad d_j \\rightarrow -d_j`

    Returns
    -------
    Transformation
        The Transformation that implements
        charge-conjugation.

    Examples
    --------
        >>> C = qosy.particle_hole()
    """
    
    return Transformation(_particle_hole_rule, None)

def charge_conjugation():
    """Same as `particle_hole()`.
    """
    
    return particle_hole()

def _sz_rule(op_string, info):
    # Helper function to define the spin_parity transformation.
    info = (up_labels, dn_labels)
    
    num_up = 0
    num_dn = 0
    for orbital_label in op_string.orbital_labels:
        if orbital_label in up_labels:
            num_up += 1
        elif orbital_label in dn_labels:
            num_dn += 1
            
    sign = 1.0
    if num_dn % 2 == 1:
        sign = -1.0
        
    return Operator([sign], [op_string])

def spin_parity_symmetry(lattice, up_name='Up', dn_name='Dn'):
    """Create a transformation whose effect is to assign
    :math:`-1` to an OperatorString with an odd number
    of down spins orbitals and :math:`+1` otherwise.

    Parameters
    ----------
    lattice : Lattice
        The Lattice object that keeps track of the orbitals
        in the system.
    up_name : str, optional
        Specifies the name of the :math:`+1/2` spin. Defaults to "Up".
    dn_name : str, optional
        Specifies the name of the :math:`-1/2` spin. Defaults to "Dn".

    Returns
    -------
    Transformation
        The transformation.
    """

    up_labels = set([lattice.index(r, name) \
                     for (r, name, _) in lattice if up_name in name])
    dn_labels = set([lattice.index(r, name) \
                     for (r, name, _) in lattice if dn_name in name])

    info = (up_labels, dn_labels)
    
    return Transformation(_sz_rule, info)
    
def _symmetry_matrix_opstrings(basis, transformation, tol=1e-12):
    """Compute the symmetry matrix for a Basis of OperatorStrings.
    """
    
    dim_basis = len(basis)

    row_inds = []
    col_inds = []
    data     = []
    for indA in range(dim_basis):
        op_string_A = basis[indA]

        operatorB = transformation.apply(op_string_A)
        
        for (coeff_B, op_string_B) in operatorB:
            
            # Ignore operators not currently in the basis.
            # If there are operators outside of the basis, this
            # will make the symmetry matrix non-unitary.
            if op_string_B not in basis:
                continue
            
            indB = basis.index(op_string_B)

            if np.abs(coeff_B) > tol:
                row_inds.append(indB)
                col_inds.append(indA)
                data.append(coeff_B)

    symmetry_matrix = ss.csc_matrix((data, (row_inds, col_inds)), dtype=complex, shape=(dim_basis, dim_basis))
            
    return symmetry_matrix

def _symmetry_matrix_operators(operators, transformation, tol=1e-12):
    """Compute the symmetry matrix for a list of Operators that specify a "basis".
    """
    
    # Assemble the combined Basis of all the OperatorStrings
    # in the list of Operators.
    operators_basis = Basis()
    for op in operators:
        operators_basis += op._basis

    num_operators = len(operators)
    dim_basis     = len(operators_basis)

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
    coeffs_operators = ss.csc_matrix((data, (row_inds, col_inds)), shape=(dim_basis, num_operators), dtype=complex)

    print('coeffs_operators=\n{}'.format(coeffs_operators.toarray()))
    
    # Calculate the Moore-Penrose pseudo-inverse using numpy.
    pinv_operators = ss.csc_matrix(nla.pinv(coeffs_operators.toarray()))

    print('pinv_operators=\n{}'.format(pinv_operators.toarray()))
    
    # Assemble the symmetry matrix in the combined Basis of OperatorStrings.
    symmetry_matrix_opstrings = _symmetry_matrix_opstrings(operators_basis, transformation, tol)

    # Project this symmetry matrix into the list of Operators "basis".
    symmetry_matrix_operators = (pinv_operators).dot(symmetry_matrix_opstrings.dot(coeffs_operators))
    
    return symmetry_matrix_operators

def symmetry_matrix(basis, transformation, tol=1e-12):
    """Create the symmetry matrix that represents the
    effect of the transformation on the basis of 
    OperatorStrings.

    For a unitary (or antiunitary) transformation
    operator :math:`\hat{\mathcal{U}}` and a basis 
    of operator strings :math:`\hat{h}_a`,
    the symmetry matrix :math:`M` is defined through
        :math:`\hat{\mathcal{U}} \hat{h}_a \hat{\mathcal{U}}^{-1} = \sum_b M_{ba} \hat{h}_b`

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings used to
        represent the symmetry matrix.
    transformation : Transformation
        The Transformation to perform 
        on the OperatorStrings.
    
    Returns
    -------
    scipy.sparse.csc_matrix of complex
        A sparse, not-necessarily-unitary complex matrix.
    """

    if isinstance(basis, Basis):
        return _symmetry_matrix_opstrings(basis, transformation, tol)
    elif isinstance(basis, list) and isinstance(basis[0], Operator):
        return _symmetry_matrix_operators(basis, transformation, tol)
    else:
        raise ValueError('Cannot find the symmetry matrix for a basis of type: {}'.format(type(basis)))
