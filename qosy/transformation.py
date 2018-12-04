#!/usr/bin/env python
import numpy as np
import scipy.sparse as ss

from .tools import sort_sign
from .operatorstring import OperatorString
from .basis import Operator

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
        c_i -> +/- c_i, c_i^\dagger -> +/- c_i^\dagger, i -> -i (imaginary number)
        a_i -> +/- a_i, b_i -> -/+ b_i, d_i -> d_i, i -> -i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if not (op_type == 'Pauli' or op_type == 'Majorana'):
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # Extra information specifies whether time-reversal transformation
    # is of type T^2 = 1 or T^2 = (-1)^N
    extra_sign = float(info)
    
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
        # A minus sign occurs if T^2 = (-1)^N and there are an odd number of a_i,b_i operators.
        if numABs % 2 == 1:
            coeffB *= extra_sign
    else:
        # A minus sign occurs for every Pauli matrix in the Pauli string.
        num_ops = len(op_string_B.orbital_operators)
        if num_ops % 2 == 1:
            coeffB *= -1.0
            
        # TODO: Implement T^2=(-1)^N symmetry for spin-1/2's if it makes sense.
        if np.abs(extra_sign - 1.0) > tol:
            raise NotImplementedError('Unsupported for spin-1/2 operators.')

    return Operator([coeffB], [op_string_B])

def _particle_hole_rule(op_string_A, info):
    """Transformation rule for particle hole symmetry C.
    Transformation for (spinless) fermions:
        c_i -> +/- c_i^\dagger, c_i^\dagger -> +/- c_i, i -> i (imaginary number)
        a_i -> +/- a_i, b_i -> -/+ b_i, d_i -> - d_i, i -> i (imaginary number)
    """
    
    op_type = op_string_A.op_type
    if op_type != 'Majorana':
        raise ValueError('Invalid op_type: {}'.format(op_type))
        
    # Extra information specifies whether particle-hole transformation
    # is of type C^2 = 1 or C^2 = (-1)^N
    extra_sign = float(info)
    
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
        
    # A minus sign occurs if C^2 = (-1)^N and there are an odd number of a_i,b_i operators.
    if numABs % 2 == 1:
        coeffB *= extra_sign
            
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

def time_reversal(sign=1.0):
    """Create a (spin-less) time-reversal 
    Transformation :math:`\hat{\mathcal{T}}`.
    
    The most important aspect of time reversal
    is that it is antiunitary and acts as a 
    complex conjugation operator:
       :math:`\hat{\mathcal{T}} i \hat{\mathcal{T}}^{-1} = -i \quad (i \\rightarrow -i)`
    
    The transformation acts on (spin-less)
    Fermionic operators as
        :math:`c_j \\rightarrow \pm c_j, \quad c_j^\dagger \\rightarrow  \pm c_j^\dagger`
    and Majorana operators as
        :math:`a_j \\rightarrow \pm a_j, \quad b_j \\rightarrow \mp b_j, \quad d_j \\rightarrow d_j`

    On spin operators, the transformation acts as:
        :math:`\sigma^a_j \\rightarrow -\sigma^a_j \quad (\hat{\mathcal{T}} \sigma^a_j \hat{\mathcal{T}}^{-1} = -\sigma^a_j)`

    Parameters
    ----------
    sign : 1.0 or -1.0, optional
        Specifies whether :math:`\hat{\mathcal{T}}^2 = 1` or :math:`\hat{\mathcal{T}}^2 = (-1)^\hat{N}`.
        Defaults to 1.0.

    Returns
    -------
    Transformation
        The Transformation that implements
        time-reversal.

    Examples
    --------
    >>> T = qosy.time_reversal()
    """
    
    return Transformation(_time_reversal_rule, sign)

def particle_hole(sign=1.0):
    """Create a particle-hole (or charge-conjugation) 
    Transformation :math:`\hat{\mathcal{C}}`.
    
    The transformation acts on (spin-less)
    Fermionic operators as
        :math:`c_j \\rightarrow \pm c_j^\dagger, \quad c_j^\dagger \\rightarrow \pm c_j`
    and Majorana operators as
        :math:`a_j \\rightarrow \pm a_j, \quad b_j \\rightarrow \mp b_j, \quad d_j \\rightarrow - d_j`

    Parameters
    ----------
    sign : 1.0 or -1.0, optional
        Specifies whether :math:`\hat{\mathcal{C}}^2 = 1` 
        or :math:`\hat{\mathcal{C}}^2 = (-1)^\hat{N}`.
        Defaults to 1.0.

    Returns
    -------
    Transformation
        The Transformation that implements
        charge-conjugation.

    Examples
    --------
    >>> C = qosy.particle_hole()
    """
    
    return Transformation(_particle_hole_rule, sign)

def charge_conjugation(sign=1.0):
    """Same as particle_hole(sign).
    """
    
    return particle_hole(sign)

def symmetry_matrix(transformation, basis, tol=1e-12):
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
    transformation : Transformation
        The Transformation to perform 
        on the OperatorStrings.
    basis : Basis
        The Basis of OperatorStrings used to
        represent the symmetry matrix.
    
    Returns
    -------
    scipy.sparse.csc_matrix of complex
        A sparse, not-necessarily-unitary complex matrix.
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
