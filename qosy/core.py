#!/usr/bin/env python
"""
This module provides the main functionality of ``qosy``:
it provides high-level functions and classes for generating
operators that commute with given symmetries.

The method :py:func:`commuting_operators`
generates operators in a given Basis that commute (or anti-commute)
with a given Hermitian operator, which for example
can be a generator of a continuous symmetry.

The method :py:func:`invariant_operators`
generates operators in a given Basis that are left
invariant (or change by a minus sign) under a discrete
symmetry Transformation.

The :py:class:`SymmetricOperatorGenerator` class 
generates the operators in a given Basis that have
many desired continuous and discrete symmetries at one.
"""

import warnings
import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

from .tools import intersection, gram_schmidt
from .basis import Operator
from .algebra import commutant_matrix
from .transformation import Transformation, symmetry_matrix

def commuting_operators(basis, operator, operation_mode='commutator', return_com_matrix=False, num_vecs=None, tol=1e-10):
    """Find operators in the vector space spanned by the
    OperatorStrings :math:`\hat{h}_a` that commute (or 
    anti-commute) with the given Operator 
    :math:`\hat{\mathcal{O}} = \sum_{a} g_a \hat{h}_a`.

    These operators are null vectors of the commutant matrix.

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings :math:`\hat{h}_a` to search in.
    operator : Operator
        The Operator :math:`\hat{\mathcal{O}}` to commute (or 
        anti-commute) with.
    operation_mode : str, optional
        Specifies whether to search for operators that
        commute with `operator` ('commutator') or anti-commute
        with `operator` ('anticommutator'). Defaults to 
        'commutator'.
    return_com_matrix : bool, optional
        If True, return the commutant matrix along with
        its null space. Default is False.
    num_vecs : int, optional
        Default to None. If None, then the commutant matrix
        is converted to a dense matrix and full diagonalization
        is performed. Otherwise, keeps the matrix sparse
        and computes the `num_vecs` lowest eigenvectors during
        diagonalization.
    tol : float, optional
        The numerical cutoff used to determine what eigenvectors
        are in the null space of the commutant matrix.

    Returns
    -------
    ndarray or (scipy sparse matrix, scipy sparse matrix, scipy sparse matrix, scipy sparse matrix)
        If `return_com_matrix` is False, returns a numpy array
        whose columns are operators in the given Basis. In particular,
        the column vectors :math:`J^{(1)},\ldots,J^{(M)}` are the 
        coefficients :math:`J_a^{(i)}` of operators 
        :math:`\hat{H}^{(i)}=\sum_{a} J_a^{(i)} \hat{h}_a` that commute 
        (or anti-commute) with the given Operator :math:`\hat{\mathcal{O}}`.
        If `return_com_matrix` is True, returns a tuple of the null
        vectors, the matrix :math:`C^\dagger C`, its eigenvalues, and its eigenvectors
        as sparse scipy matrices where :math:`C` is the commutant matrix.
    """
    
    com_matrix = commutant_matrix(basis, operator, operation_mode=operation_mode)

    CDagC = (com_matrix.H).dot(com_matrix)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(CDagC.toarray())
    else:
        # Use a small negative sigma for the shift-invert method.
        sigma = -1e-2
        (evals, evecs) = ssla.eigsh(CDagC, k=num_vecs, sigma=sigma)

    inds_ns    = np.where(np.abs(evals) < tol)[0]
    null_space = evecs[:,inds_ns]

    if num_vecs is not None and len(inds_ns) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the null space. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs))

    if return_com_matrix:
        return (null_space, CDagC, evals, evecs)
    else:
        return null_space

def invariant_operators(basis, transform, operation_mode='commutator', num_vecs=None, return_sym_matrix=False, tol=1e-10):
    """Find operators in the vector space spanned by the
    OperatorStrings :math:`\hat{h}_a` that commute (or anti-commute)
    with the Transformation :math:`\hat{\mathcal{U}}`.

    These operators are :math:`(\pm 1)`-eigenvalue eigenvectors 
    of the symmetry matrix.

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings :math:`\hat{h}_a` to search in.
    transform : Transformation
        The Transformation :math:`\hat{\mathcal{U}}` to commute (or 
        anti-commute) with.
    operation_mode : str, optional
        Specifies whether to search for operators that
        commute with `transform` ('commutator') or anti-commute
        with `transform` ('anticommutator'). Defaults to 
        'commutator'.
    return_sym_matrix : bool, optional
        If True, return the symmetry matrix and its spectrum along with
        its :math:`(\pm 1)` eigenspace. Default is False.
    num_vecs : int, optional
        Default to None. If None, then the symmetry matrix
        is converted to a dense matrix and full diagonalization
        is performed. Otherwise, keeps the matrix sparse
        and computes the `num_vecs` eigenvectors closest
        to :math:`(\pm 1)` during diagonalization.
    tol : float, optional
        The numerical cutoff used to determine what eigenvectors
        are in the :math:`(\pm 1)` eigenspace of the symmetry matrix.

    Returns
    -------
    ndarray or (sparse matrix, sparse matrix, sparse matrix, sparse matrix)
        If `return_sym_matrix` is False, returns a numpy array
        whose columns are operators in the given Basis. In particular,
        the column vectors :math:`J^{(1)},\ldots,J^{(M)}` are the 
        coefficients :math:`J_a^{(i)}` of operators 
        :math:`\hat{H}^{(i)}=\sum_{a} J_a^{(i)} \hat{h}_a` that commute 
        (or anti-commute) with the given Transformation :math:`\hat{\mathcal{U}}`.
        If `return_sym_matrix` is True, returns a tuple of the null
        vectors, the symmetry matrix, its eigenvalues, and its eigenvectors
        as sparse scipy matrices.
    """
    
    sym_matrix = symmetry_matrix(basis, transform)
    
    if operation_mode in ['commute', 'Commute', 'symmetry', 'commutator']:
        sign = 1.0
    elif operation_mode in ['anticommute', 'Anticommute', 'antisymmetry', 'anticommutator']:
        sign = -1.0
    else:
        raise ValueError('Unknown operation_mode: {}'.format(operation_mode))

    matrix = 0.5*(sym_matrix + sym_matrix.H)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(matrix.toarray())
    else:
        (evals, evecs) = ssla.eigsh(matrix, k=num_vecs, sigma=sign)

    inds = np.where(np.abs(evals - sign) < tol)[0]
    vecs = evecs[:,inds]

    if num_vecs is not None and len(inds) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the ({})-eigenvalue subspace. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs, sign))

    if return_sym_matrix:
        return (vecs, sym_matrix, evals, evecs)
    else:
        return vecs

class SymmetricOperatorGenerator:
    """A SymmetricOperatorGenerator object can be used to generate
    operators with many desired continuous and discrete symmetries
    at once.

    Attributes
    ----------
    basis : Basis
        The Basis of OperatorStrings to search for symmetric operators in.
    input_symmetries : list of Operators and Transformations
        A list of Hermitian Operators or symmetry Transformations
        that we want to commute (or anti-commute) with our output operators.
    output_operators : list of ndarray
        A list of numpy arrays that represents the operators in the
        given Basis that obey the given symmetries. This list is not
        cumulative. The first operators in the list satisfy the first
        symmetry; the second operators in the list satisfy the second
        symmetry; etc.

    operation_modes : list of str
        A list that specifies whether to search for operators to 
        commute or anticommute with each symmetry. 'commutator' 
        means commutes, 'anticommutator' means anticommutes.
    superoperators : list of scipy.sparse.csc_matrix
        A list of the "superoperators" (the commutant and 
        symmetry matrices) computed for each symmetry (Operator and 
        Transformation).
    eigenvalues : list of ndarray
        A list of the eigenvalues of the superoperators.
    eigenvectors : list of ndarray
        A list of the eigenvectors of the superoperators.

    projected_superoperators : list of scipy.sparse.csc_matrix
        The superoperators projected onto the symmetric operators
        generated from the previously considered symmetries.
    projected_eigenvalues : list of ndarray
        A list of the eigenvalues of the projected superoperators. 
    projected_eigenvectors : list of ndarray
        A list of the eigenvectors of the projected superoperators.
    projected_output_operators : list of ndarray
        A cumulative list of the operators that obey the given symmetries.
        The first operators in the list satisfy the first symmetry; the 
        second operators in the list satisfy the first and second symmetries; etc.

    num_vecs : None or int
        Specifies whether to use scipy.sparse.linalg.eigsh
        or numpy.linalg.eigh to diagonalize matrices. If None,
        uses eigh, otherwise finds ``num_vecs`` closest
        eigenvectors to the desired eigenvalue with eigsh.

    Examples
    --------
    To use the generator, first initialize it with 
    a Basis of OperatorStrings
        >>> orbitals = [1,2,3,4] # 4 orbitals
        >>> basis = qosy.cluster_basis(2, orbitals, 'Pauli') # All 2-local Pauli strings on the orbitals. 
        >>> generator = qosy.SymmetricOperatorGenerator(basis)
    Then add symmetries that you want to enforce
    on your operators to the generator
        >>> # Discrete time-reversal symmetry transformation
        >>> T = qosy.time_reversal()
        >>> # The generators of the continuous global SU(2) symmetry:
        >>> totalX = qosy.Operator(np.ones(4), [qosy.opstring('X {}'.format(o)) for o in orbitals])
        >>> totalY = qosy.Operator(np.ones(4), [qosy.opstring('Y {}'.format(o)) for o in orbitals])
        >>> totalZ = qosy.Operator(np.ones(4), [qosy.opstring('Z {}'.format(o)) for o in orbitals])
        >>> generator.add_symmetry(T)
        >>> generator.add_symmetry(totalX)
        >>> generator.add_symmetry(totalY)
        >>> generator.add_symmetry(totalZ)
    Then generate the operators with
        >>> generator.generate()
    The results are stored in the ``generator`` object:
        >>> generator.output_operators[0]  # the operators obeying the first symmetry (T)
        >>> generator.output_operators[1]  # the operators obeying the second symmetry (totalX)
        >>> generator.projected_output_operators[1]  # the operators obeying the first two symmetries (T, totalX)
        >>> generator.projected_output_operators[-1] # the operators obeying all of the desired symmetries
    """
    
    def __init__(self, basis, num_vecs=None):
        # Input
        self.basis            = basis
        self.operation_modes  = []
        self.input_symmetries = []

        # Output
        self.superoperators   = []
        self.eigenvalues      = []
        self.eigenvectors     = []
        self.output_operators = []

        # Projected output
        self.projected_superoperators   = []
        self.projected_eigenvalues      = []
        self.projected_eigenvectors     = []
        self.projected_output_operators = []

        # Specifies whether to use scipy.sparse.linalg.eigh
        # or numpy.linalg.eigh.
        self.num_vecs = num_vecs
    
    def add_symmetry(self, symmetry, operation_mode='commutator'):
        """Add a desired symmetry to the generator.

        Parameters
        ----------
        symmetry : Operator or Transformation
            The symmetry to consider.
        operation_mode : str, optional
            Specifies whether to search for operators
            that commute ('commutator') or anticommute 
            ('anticommutator') with the given symmetry.
            Defaults to 'commutator'.

        Notes
        -----
        The order in which symmetries are added to the generator matters
        as it affects the order of projection during ``generate``.
        """
        
        self.operation_modes.append(operation_mode)
        self.input_symmetries.append(symmetry)

    def generate(self, orthogonalize=True, verbose=True, tol=1e-10):
        """Generate the operators in the given Basis,
        that satisfy the given symmetries.

        This procedure occurs in three steps:
          1. *Generation:* In order, calculate the commutant matrix of 
             Operator :math:`\hat{\mathcal{O}}` or the symmetry matrix 
             of Transformation :math:`\hat{\mathcal{U}}` and diagonalize it.
          2. *Projection:* In order, project the results for the previous 
             symmetries onto the current symmetry.
          3. *Orthogonalization:* (optional) In reverse order, orthogonalize the results
             for the later symmetries on the earlier symmetries.

        The results of step 1 are stored in ``output_operators``, while the results
        of steps 2 and 3 are stored in ``projected_output_operators``.

        Parameters
        ----------
        orthogonalize : bool, optional
            Specifies whether to perform step 3. Defaults to True.
        verbose : bool, optional
            Specifies whether to print the status of the operator generator.
            Defaults to True.
        tol : float, optional
            Specifies the threshold used to consider whether an eigenvalue is
            close to :math:`0,\pm 1`. Defaults to 1e-10.
        """
        
        num_inputs = len(self.input_symmetries)

        if verbose:
            print('===== 1. GENERATING OPERATORS =====')
            print(' STARTING WITH BASIS OF DIM {}'.format(len(self.basis)))
            
        # Go through the Operators or Transformations, one by one,
        # in order, and find the null spaces of the (anti-)commutant matrix
        # or the +/-1 eigenspaces of the symmetry matrix. Save the results.
        for ind_input in range(num_inputs):
            input_symmetry = self.input_symmetries[ind_input]
            operation_mode = self.operation_modes[ind_input]

            op_mode_print = ' COMMUTING WITH'
            if operation_mode == 'anticommutator':
                op_mode_print = ' ANTICOMMUTING WITH'
            
            if isinstance(input_symmetry, Operator):
                if verbose:
                    print('{} OPERATOR {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, com_matrix, eigvals, eigvecs) = commuting_operators(self.basis, input_symmetry, operation_mode=operation_mode, return_com_matrix=True, num_vecs=self.num_vecs, tol=tol)

                self.superoperators.append(com_matrix)
                self.eigenvectors.append(eigvecs)
                self.eigenvalues.append(eigvals)
                self.output_operators.append(output_ops)

                if verbose:
                    dim_output = int(self.output_operators[-1].shape[1])
                    print('  Generated a vector space of operators of dimension: {}'.format(dim_output))
                
            elif isinstance(input_symmetry, Transformation):
                if verbose:
                    print('{} TRANSFORMATION {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, sym_matrix, eigvals, eigvecs) = invariant_operators(self.basis, input_symmetry, operation_mode=operation_mode, return_sym_matrix=True, num_vecs=self.num_vecs, tol=tol)

                self.superoperators.append(sym_matrix)
                self.eigenvectors.append(eigvecs)
                self.eigenvalues.append(eigvals)
                self.output_operators.append(output_ops)

                if verbose:
                    dim_output = int(self.output_operators[-1].shape[1])
                    print('  Generated a vector space of operators of dimension: {}'.format(dim_output))
            else:
                raise ValueError('Invalid input symmetry of type {}. Must be an Operator or a Transformation.'.format(type(input_symmetry)))


        if verbose:
            print('===== 2. POST-PROCESSING: PROJECTION =====')
        # Post processing: Go through the output vector spaces
        # V_1,V_2,...,V_N and project them in order:
        # Project V_2 into V_1. Project V_3 into V_1 and V_2, etc.
        for ind_output in range(num_inputs):
            if ind_output == 0:
                # Do not project the first iteration.
                self.projected_output_operators.append(self.output_operators[ind_output])

                self.projected_superoperators.append(self.superoperators[ind_output])
                self.projected_eigenvalues.append(self.eigenvalues[ind_output])
                self.projected_eigenvectors.append(self.eigenvectors[ind_output])
            else:
                # Project onto the previous iteration's results.
                curr_ops   = self.output_operators[ind_output]
                prev_ops   = self.projected_output_operators[ind_output-1]
                
                # The projected operators in the null space (or +/-1 eigenspace)
                # can be found by taking intersections of vector spaces.
                self.projected_output_operators.append(intersection(curr_ops, prev_ops))

                # Compute the projected superoperator and its eigenvalues and eigenvectors.
                if self.num_vecs is not None:
                    # Using scipy.sparse.linalg.eigsh
                    
                    prev_ops_sparse = ss.csc_matrix(prev_ops)
                    projected_superoperator = ((prev_ops_sparse.H).dot(self.superoperators[ind_output])).dot(prev_ops_sparse)
                    self.projected_superoperators.append(projected_superoperator)

                    sigma = -1e-2
                    if isinstance(self.input_symmetries[ind_output], Transformation):
                        if self.operation_modes[ind_output] == 'commutator':
                            sigma = 1.0
                        else:
                            sigma = -1.0

                    (evalsPSO, evecsPSO) = ssla.eigsh(projected_superoperator, k=self.num_vecs, sigma=sigma)
                    self.projected_eigenvalues.append(evalsPSO)
                    projected_evecs = np.dot(prev_ops, evecsPSO)
                    self.projected_eigenvectors.append(projected_evecs)
                else:
                    # Using numpy.linalg.eigh

                    prev_ops_sparse = ss.csc_matrix(prev_ops)
                    projected_superoperator = ((prev_ops_sparse.H).dot(self.superoperators[ind_output])).dot(prev_ops_sparse)
                    self.projected_superoperators.append(projected_superoperator)

                    (evalsPSO, evecsPSO) = nla.eigh(projected_superoperator.toarray())
                    self.projected_eigenvalues.append(evalsPSO)
                    projected_evecs = np.dot(prev_ops, evecsPSO)
                    self.projected_eigenvectors.append(projected_evecs)    

            if verbose:
                dim_output = int(self.projected_output_operators[-1].shape[1])
                print(' ({}) The projected vector space of operators has dimension: {}'.format(ind_output+1, dim_output))

        if orthogonalize:
            if verbose:
                print('===== 3. POST-PROCESSING: ORTHOGONALIZATION =====')
            # Post processing: Go through the projected output
            # vector spaces in reverse order, V_N', V_{N-1}', ..., V_1', and use the
            # Gram-Schmidt procedure to orthogonalize them.
            # Orthogonalize V_{N-1}' against V_N'. Orthogonalize V_{N-2}' against V_{N-1}', etc.
            # The goal here is to find a nicer basis for each of the vector spaces.
            for ind_output in range(num_inputs-1,0,-1):
                # Orthogonalize against the previous iteration's results.
                curr_ops   = self.projected_output_operators[ind_output]
                prev_ops   = self.projected_output_operators[ind_output-1]

                num_curr_ops = int(curr_ops.shape[1])
            
                matrix                    = np.copy(prev_ops)
                matrix[:, 0:num_curr_ops] = curr_ops
            
                self.projected_output_operators[ind_output-1] = gram_schmidt(matrix, tol=0.0)

                if verbose:
                    dim_curr = int(curr_ops.shape[1])
                    dim_prev = int(prev_ops.shape[1])
                    print(' ({}) Orthogonalized a vector space of dim {} against one with dim {}'.format(ind_output, dim_prev, dim_curr))
        
