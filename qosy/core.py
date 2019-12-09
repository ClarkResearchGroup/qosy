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

from .tools import intersection, gram_schmidt, sparsify
from .basis import Basis, Operator
from .algebra import liouvillian_matrix
from .transformation import Transformation, symmetry_matrix

def commuting_operators(basis, operator, operation_mode='commutator', return_superoperator=False, num_vecs=None, tol=1e-10):
    """Find operators in the vector space spanned by the
    OperatorStrings :math:`\\hat{\\mathcal{S}}_a` that commute (or 
    anti-commute) with the given Operator 
    :math:`\\hat{\\mathcal{O}} = \\sum_{a} g_a \\hat{\\mathcal{S}}_a`.

    These operators are null vectors of the commutant matrix.

    Parameters
    ----------
    basis : Basis or list of Operators
        The Basis of OperatorStrings :math:`\\hat{\\mathcal{S}}_a` or list 
        of Operators :math:`\\hat{\\mathcal{O}}_a` to search in.
    operator : Operator
        The Operator :math:`\\hat{\\mathcal{O}}` to commute (or 
        anti-commute) with.
    operation_mode : str, optional
        Specifies whether to search for operators that
        commute with `operator` ('commutator') or anti-commute
        with `operator` ('anticommutator'). Defaults to 
        'commutator'.
    return_superoperator : bool, optional
        If True, return :math:`C^\\dagger C` where :math:`C`
        is the commutant matrix, along with eigenvalues and
        eigenvectors. Default is False.
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
        the column vectors :math:`J^{(1)},\\ldots,J^{(M)}` are the 
        coefficients :math:`J_a^{(i)}` of operators 
        :math:`\\hat{\\mathcal{S}}^{(i)}=\\sum_{a} J_a^{(i)} \\hat{\\mathcal{S}}_a` that commute 
        (or anti-commute) with the given Operator :math:`\\hat{\\mathcal{O}}`.
        If `return_superoperator` is True, returns a tuple of the null
        vectors, the commutant matrix :math:`C`, its eigenvalues, and its eigenvectors
        as sparse scipy matrices.
    """
    
    l_matrix = liouvillian_matrix(basis, operator, operation_mode=operation_mode)

    com_matrix = (l_matrix.H).dot(l_matrix)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(com_matrix.toarray())
    else:
        # Use a small negative sigma for the shift-invert method.
        sigma = -1e-6
        (evals, evecs) = ssla.eigsh(com_matrix, k=num_vecs, sigma=sigma)

    inds_ns    = np.where(np.abs(evals) < tol)[0]
    null_space = evecs[:,inds_ns]

    if num_vecs is not None and len(inds_ns) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the null space. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs))

    if return_superoperator:
        return (null_space, com_matrix, evals, evecs)
    else:
        return null_space

def invariant_operators(basis, transform, operation_mode='commutator', num_vecs=None, return_superoperator=False, tol=1e-10):
    """Find operators in the vector space spanned by the
    OperatorStrings :math:`\\hat{\\mathcal{S}}_a` that commute (or anti-commute)
    with the Transformation :math:`\\hat{\\mathcal{U}}`.

    These operators are :math:`(\\pm 1)`-eigenvalue eigenvectors 
    of the symmetry matrix.

    Parameters
    ----------
    basis : Basis or list of Operators
        The Basis of OperatorStrings :math:`\\hat{\\mathcal{S}}_a` or list 
        of Operators :math:`\\hat{\\mathcal{O}}_a` to search in.
    transform : Transformation
        The Transformation :math:`\\hat{\\mathcal{U}}` to commute (or 
        anti-commute) with.
    operation_mode : str, optional
        Specifies whether to search for operators that
        commute with `transform` ('commutator') or anti-commute
        with `transform` ('anticommutator'). Defaults to 
        'commutator'.
    return_superoperator : bool, optional
        If True, returns :math:`I \\mp (S + S^\\dagger)/2` where
        :math:`S` is the symmetry matrix. The null space of 
        this superoperator corresponds to the :math:`(\\pm 1)` 
        eigenspace of the symmetry matrix. Default is False.
    num_vecs : int, optional
        Default to None. If None, then the symmetry matrix
        is converted to a dense matrix and full diagonalization
        is performed. Otherwise, keeps the matrix sparse
        and computes the `num_vecs` eigenvectors closest
        to :math:`(\\pm 1)` during diagonalization.
    tol : float, optional
        The numerical cutoff used to determine what eigenvectors
        are in the :math:`(\\pm 1)` eigenspace of the symmetry matrix.

    Returns
    -------
    ndarray or (sparse matrix, sparse matrix, sparse matrix, sparse matrix)
        If `return_sym_matrix` is False, returns a numpy array
        whose columns are operators in the given Basis. In particular,
        the column vectors :math:`J^{(1)},\\ldots,J^{(M)}` are the 
        coefficients :math:`J_a^{(i)}` of operators 
        :math:`\\hat{H}^{(i)}=\\sum_{a} J_a^{(i)} \\hat{\\mathcal{S}}_a` that commute 
        (or anti-commute) with the given Transformation :math:`\\hat{\\mathcal{U}}`.
        If `return_superoperator` is True, returns a tuple of the null
        vectors, a modified symmetry matrix, its eigenvalues, and its eigenvectors
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

    if return_superoperator:
        superoperator = ss.eye(len(basis), format='csc') - sign * matrix
        return (vecs, superoperator, 1.0 - sign*evals, evecs)
    else:
        return vecs


    
class SymmetricOperatorGenerator:
    """A SymmetricOperatorGenerator object can be used to generate
    operators with many desired continuous and discrete symmetries
    at once.

    Attributes
    ----------
    basis : Basis or list of Operators
        The Basis of OperatorStrings or list of Operators to 
        search for symmetric operators in.
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
    
    def __init__(self, basis):
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
        # or numpy.linalg.eigh for each symmetry.
        self.num_vecs = []
    
    def add_symmetry(self, symmetry, operation_mode='commutator', num_vecs=None):
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
        num_vecs : int, optional
            If not None, uses Lanczos and returns the lowest
            `num_vecs` number of lowest eigenvalues instead
            of performing full diagonalization. Defaults to
            None.

        Notes
        -----
        The order in which symmetries are added to the generator matters
        as it affects the order of projection during ``generate``.
        """
        
        self.operation_modes.append(operation_mode)
        self.input_symmetries.append(symmetry)
        self.num_vecs.append(num_vecs)

    def _generate_noncumulative(self, sparsification=True, orthogonalization=False, verbose=True, tol=1e-10):
        
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
                
                (output_ops, superoperator, eigvals, eigvecs) = commuting_operators(self.basis, input_symmetry, operation_mode=operation_mode, return_superoperator=True, num_vecs=self.num_vecs[ind_input], tol=tol)              
            elif isinstance(input_symmetry, Transformation):
                if verbose:
                    print('{} TRANSFORMATION {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, superoperator, eigvals, eigvecs) = invariant_operators(self.basis, input_symmetry, operation_mode=operation_mode, return_superoperator=True, num_vecs=self.num_vecs[ind_input], tol=tol)
            else:
                raise ValueError('Invalid input symmetry of type {}. Must be an Operator or a Transformation.'.format(type(input_symmetry)))

            self.superoperators.append(superoperator)
            self.eigenvectors.append(eigvecs)
            self.eigenvalues.append(eigvals)
            self.output_operators.append(output_ops)
            
            if verbose:
                dim_output = int(self.output_operators[-1].shape[1])
                print('  Generated a vector space of operators of dimension: {}'.format(dim_output))

        if verbose:
            print('===== 2. POST-PROCESSING: PROJECTION =====')
        # Post processing: Go through the output vector spaces
        # V_1,V_2,...,V_N and project them in order:
        # Project V_2 into V_1. Project V_3 into V_1 and V_2, etc.
        for ind_output in range(num_inputs):
            if ind_output == 0:
                # NOTE: The convention is that there is no
                # projection in the first iteration. The "projected_superoperator"
                # and its eigenvalues/eigenvectors are the same as the unprojected
                # one. TODO: This should probably be changed in the future.
                self.projected_output_operators.append(self.output_operators[ind_output])

                self.projected_superoperators.append(self.superoperators[ind_output])
                self.projected_eigenvalues.append(self.eigenvalues[ind_output])
                self.projected_eigenvectors.append(self.eigenvectors[ind_output])
            else:
                # Project onto the previous iteration's results.
                curr_ops   = self.output_operators[ind_output]
                prev_ops   = self.projected_output_operators[ind_output-1]
                
                # The current projected operators are the intersection
                # of the current unprojected operators and the previous
                # projected operators.
                proj_curr_ops = intersection(curr_ops, prev_ops)

                # Compute the projected superoperator.
                proj_curr_ops_sparse    = ss.csc_matrix(proj_curr_ops)
                projected_superoperator = ((proj_curr_ops_sparse.H).dot(self.superoperators[ind_output])).dot(proj_curr_ops_sparse)
                self.projected_superoperators.append(projected_superoperator)
                
                # Compute the projected superoperator's eigenvalues and eigenvectors.
                if self.num_vecs[ind_output] is not None:
                    # Using scipy.sparse.linalg.eigsh
                    sigma = -1e-6
                    (evalsPSO, evecsPSO) = ssla.eigsh(projected_superoperator, k=self.num_vecs[ind_output], sigma=sigma)
                else:
                    # Using numpy.linalg.eigh
                    (evalsPSO, evecsPSO) = nla.eigh(projected_superoperator.toarray())

                self.projected_eigenvalues.append(evalsPSO)
                projected_evecs = np.dot(proj_curr_ops, evecsPSO)
                self.projected_eigenvectors.append(projected_evecs)
            
                inds_nullspace = np.where(np.abs(evalsPSO) < tol)[0]
                self.projected_output_operators.append(projected_evecs[:, inds_nullspace])
                    
            if verbose:
                dim_output = int(self.projected_output_operators[-1].shape[1])
                print(' ({}) The projected vector space of operators has dimension: {}'.format(ind_output+1, dim_output))

        if sparsification:
            if verbose:
                print('===== 3. POST-PROCESSING: SPARSIFICATION =====')

            for ind_output in range(num_inputs):
                curr_ops = self.projected_output_operators[ind_output]
                self.projected_output_operators[ind_output] = sparsify(curr_ops)
                print(' ({}) Sparsified a vector space of dim {}'.format(ind_output+1, int(curr_ops.shape[1])))
        
        if orthogonalization:
            if verbose:
                print('===== 4. POST-PROCESSING: ORTHOGONALIZATION =====')
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

    def _generate_cumulative(self, sparsification=True, orthogonalization=True, verbose=True, tol=1e-10):
        num_inputs = len(self.input_symmetries)

        if verbose:
            print('===== GENERATING OPERATORS =====')
            print(' STARTING WITH BASIS OF DIM {}'.format(len(self.basis)))
            
        for ind_input in range(num_inputs):
            # For the first iteration, use the given "basis",
            # which can be a Basis of OperatorStrings or a
            # list of Operators.
            if ind_input == 0:
                basis = self.basis
            # For the remaining iterations, use a list of Operators
            # which are in the null space of the previous superoperator,
            # as the "basis."
            else:
                basis = self.projected_output_operators[-1]

            if len(basis) == 0:
                print(' SKIPPING SYMMETRY {}'.format(ind_input+1))
                continue
                
            input_symmetry = self.input_symmetries[ind_input]
            operation_mode = self.operation_modes[ind_input]

            op_mode_print = ' COMMUTING WITH'
            if operation_mode == 'anticommutator':
                op_mode_print = ' ANTICOMMUTING WITH'

            # Construct the superoperator and find its null space.
            if isinstance(input_symmetry, Operator):
                if verbose:
                    print('{} OPERATOR {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, superoperator, eigvals, eigvecs) = commuting_operators(basis, input_symmetry, operation_mode=operation_mode, return_superoperator=True, num_vecs=self.num_vecs[ind_input], tol=tol)              
            elif isinstance(input_symmetry, Transformation):
                if verbose:
                    print('{} TRANSFORMATION {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, superoperator, eigvals, eigvecs) = invariant_operators(basis, input_symmetry, operation_mode=operation_mode, return_superoperator=True, num_vecs=self.num_vecs[ind_input], tol=tol)
            else:
                raise ValueError('Invalid input symmetry of type {}. Must be an Operator or a Transformation.'.format(type(input_symmetry)))

            self.projected_superoperators.append(superoperator)
            self.projected_eigenvectors.append(eigvecs)
            self.projected_eigenvalues.append(eigvals)

            # Sparsify the vectors representing the
            # operators in the null space.
            if sparsification:
                # Always sparsify and orthogonalize first.
                output_ops = sparsify(output_ops, orthogonalize=True)

                # Then sparsify without orthogonalizing if necessary.
                if not orthogonalization:
                    output_ops = sparsify(output_ops, orthogonalize=False)

            # Convert the vectors into a list of Operators.
            projected_output_ops = []
            for ind_nullvec in range(int(output_ops.shape[1])):
                if isinstance(basis, Basis):
                    coeffs     = []
                    op_strings = []
                    for ind_basis in range(int(output_ops.shape[0])):
                        coeff = output_ops[ind_basis, ind_nullvec]
                        if np.abs(coeff) > tol:
                            coeffs.append(coeff)
                            op_strings.append(basis[ind_basis])
                    proj_op = Operator(coeffs, op_strings)
                elif isinstance(basis, list) and isinstance(basis[0], Operator):
                    # Zero operator
                    proj_op = Operator([], [], basis[0].op_type)
                
                    for ind_basis in range(int(output_ops.shape[0])):
                        coeff = output_ops[ind_basis, ind_nullvec]
                        if np.abs(coeff) > tol:
                            proj_op += coeff * basis[ind_basis]

                    proj_op.remove_zeros()
                    proj_op.normalize()
                else:
                    raise ValueError('Invalid basis of type: {}'.format(type(basis)))        

                projected_output_ops.append(proj_op)
                
            self.projected_output_operators.append(projected_output_ops)
            
            if verbose:
                dim_output = len(self.projected_output_operators[-1])
                print('  Generated a vector space of operators of dimension: {}'.format(dim_output))

    def generate(self, mode='cumulative', sparsification=True, orthogonalization=False, verbose=True, tol=1e-10):
        """Generate the operators in the given Basis,
        that satisfy the given symmetries.

        This can happen in two modes: cumulative or noncumulative.
        In the cumulative mode, the symmetric operators of one 
        symmetry are used as the starting basis for the next symmetry
        calculation. In the non-cumulative mode, all calculations
        use the originally provided basis.

        The results of the noncumulative mode calculations 
        are stored in ``output_operators`` and ``projected_output_operators``.
        The results of the cumulative mode calculations are
        only stored in ``projected_output_operators``.

        Parameters
        ----------
        mode : str
            'cumulative' or 'noncumulative' as explained above. Defaults
            to 'cumulative'.
        sparsification : bool, optional
            Specifies whether to sparsify vectors during the calculations. 
            Defaults to True.
        orthogonalization : bool, optional
            Specifies whether to perform orthogonalize vectors as a
            post-processing step in the noncumulative mode. Defaults to False.
        verbose : bool, optional
            Specifies whether to print the status of the operator generator.
            Defaults to True.
        tol : float, optional
            Specifies the threshold used to consider whether an eigenvalue is
            close to :math:`0,\\pm 1`. Defaults to 1e-10.

        Notes
        -----
        The noncumulative mode calculation occurs in up to four steps:
          1. *Generation:* In order, calculate the commutant matrix of 
             Operator :math:`\\hat{\\mathcal{O}}` or the symmetry matrix 
             of Transformation :math:`\\hat{\\mathcal{U}}` and diagonalize it.
          2. *Projection:* In order, project the results for the previous 
             symmetries onto the current symmetry.
          3. *Sparsification:* (optional) In order, sparsify the vectors obtained from steps 1-2.
          4. *Orthogonalization:* (optional) In reverse order, orthogonalize the results
             for the later symmetries on the earlier symmetries.
        """
        
        if mode == 'cumulative':
            return self._generate_cumulative(sparsification=sparsification, orthogonalization=orthogonalization, verbose=verbose, tol=tol)
        elif mode == 'noncumulative':
            return self._generate_noncumulative(sparsification=sparsification, orthogonalization=orthogonalization, verbose=verbose, tol=tol)
        else:
            raise ValueError('Invalid operator generation mode: {}'.format(mode))


def _ladder_operators_original(basis, operator, sparsification=False, tol=1e-14, return_operators=True):
    """Find operators :math:`\\hat{R}` in the vector space spanned by the
    OperatorStrings :math:`\\hat{\\mathcal{S}}_a` that are (raising) 
    ladder operators with respect to the operator 
    :math:`\\hat{\\mathcal{O}} = \\sum_{a} g_a \\hat{\\mathcal{S}}_a`.

    These operators satisfy :math:`[\\hat{O}, \\hat{R}] = (\\delta O) \\hat{R}`
    where :math:`\\delta O > 0`.

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings :math:`\\hat{\\mathcal{S}}_a` to search in.
    operator : Operator
        The Operator :math:`\\hat{\\mathcal{O}}` to be a raising operator of.
    sparsification : bool, optional
        Specifies whether to sparsify vectors during the calculations. 
        Defaults to False.
    tol : float, optional
        The numerical cutoff used to determine what eigenvectors
        are in a null space.
    return_operators : bool, optional
        Flag that specifies whether to return a list of Operators
        instead of numpy array representing vectors in the given basis. 
        Defaults to True.

    Returns
    -------
    list of Operators or ndarray
        If `return_operators` is False, returns a numpy array
        whose columns are operators in the given Basis.
        If `return_operators` is True, returns a list of Operators.
    """
    
    basisA = basis

    # Liouvillian matrix between basisA and basisB
    (L_A_B, basisB) = liouvillian_matrix(basisA, -operator, return_extended_basis=True)

    # Make a combined AB basis that includes the elements of basisA
    # (in the same order as before) and the new elements of basisB.
    op_stringsAB = list(basisA.op_strings)
    for os_b in basisB:
        if os_b not in basisA:
            op_stringsAB.append(os_b)
    basisAB = Basis(op_stringsAB)

    # Liouvillian matrix between basisAB and basisC
    (L_AB_C, basisC) = liouvillian_matrix(basisAB, -operator, return_extended_basis=True)

    print('Dim of basisAB = {}\nDim of basisC = {}'.format(len(basisAB), len(basisC)))

    # Project the Liouvillian matrix into basisAB
    L_AB_C_coo = L_AB_C.tocoo()
    row_inds = []
    col_inds = []
    data     = []
    for indC, indAB1, val in zip(L_AB_C_coo.row, L_AB_C_coo.col, L_AB_C_coo.data):
        osC = basisC.op_strings[indC]
        if osC in basisAB:
            indAB2 = basisAB.index(osC)
            row_inds.append(indAB2)
            col_inds.append(indAB1)
            data.append(val)
    
    L = ss.csr_matrix((data, (row_inds, col_inds)), shape=(len(basisAB), len(basisAB)), dtype=complex)
    # Make it Hermitian
    #L = 0.5*(L + L.H)
    
    #check_hermitian = np.allclose(L.toarray(), L.getH().toarray())
    #assert(check_hermitian)
    
    # Print the different bases.
    #print('Basis A:\n{}\nBasis AB:\n{}'.format(basisA, basisAB))
    
    # Print the final, square, projected, Hermitian Liouvillian matrix in
    # the basisAB basis.
    #print('L=\n{}'.format(L.toarray()))
    
    # Diagonalize the projected Liouvillian matrix.
    (evals, evecsAB) = nla.eig(L.toarray())
    
    print('All eigenvalues:\n{}'.format(evals))
    #print('All eigenvectors:\n{}'.format(evecsAB.shape))
    
    # Focus on the (real) positive eigenvalue eigenvectors.
    inds_pos_evals = np.where(np.logical_and(np.imag(evals) < tol, np.real(evals) > tol))[0]
    
    #print('inds_pos_evals = {}'.format(inds_pos_evals))
    
    # Project the positive eigenvectors out of basisA.
    projected_evecsAB = evecsAB[:, inds_pos_evals]
    projected_evecsAB = projected_evecsAB[len(basisA):len(basisAB), :]
    
    #print('Projected eigenvectors out of basisA:\n{}'.format(projected_evecsAB.shape))
    
    # TODO: modify. Go through each unique positive eigenvalue
    # subspace separately.
    # Compute the overlaps of these projected eigenvectors.
    overlaps_projected = np.dot(np.conj(projected_evecsAB.T), projected_evecsAB)
    
    # Find the null space of this overlap matrix. All eigenstates
    # in the null space must correspond to vectors that only
    # exist in the original basisA basis.
    (evals_overlaps, evecs_overlaps) = nla.eigh(overlaps_projected)
    
    #print('size(evals_overlaps) = {}'.format(len(evals_overlaps)))
    print('evals_overlaps = {}'.format(evals_overlaps))

    inds_null_space = np.where(np.abs(evals_overlaps) < tol)[0]
    num_ladder_ops  = len(inds_null_space)

    if num_ladder_ops == 0:
        if return_operators:
            evecs_ladder_ops = []
        else:
            evecs_ladder_ops = np.zeros((len(basisA), 0), dtype=complex)
            
        return (evecs_ladder_ops, np.zeros(0, dtype=complex))
    
    evecs_ladder_ops = np.dot(evecsAB[:, inds_pos_evals], evecs_overlaps[:, inds_null_space])
    for ind_ev in range(num_ladder_ops):
        evecs_ladder_ops[:, ind_ev] /= nla.norm(evecs_ladder_ops[:, ind_ev])

    # Project into the positive eigenvalue basis of L (that exists completely in basisA).
    L_ladder_ops = np.dot(np.conj(evecs_ladder_ops.T), np.dot(L.toarray(), evecs_ladder_ops))
    check_hermitian = np.allclose(L_ladder_ops, np.conj(L_ladder_ops.T))
    assert(check_hermitian)
    
    (evals_ladder_ops, evecs_lo) = nla.eig(L_ladder_ops)

    evecs_ladder_ops = np.dot(evecs_ladder_ops, evecs_lo)
    
    print('evals_ladder_ops = {}'.format(evals_ladder_ops))
    
    # Go through each degenerate subspace and sparsify the basis of that subspace.
    if sparsification:
        inds_sort = np.argsort(np.real(evals_ladder_ops))
        evals_ladder_ops = evals_ladder_ops[inds_sort]
        evecs_ladder_ops = evecs_ladder_ops[:, inds_sort]

        ind_ev = 0
        while ind_ev < len(evals_ladder_ops):
            inds_jump = np.where(np.abs(evals_ladder_ops[ind_ev:] - evals_ladder_ops[ind_ev]) > 1e-10)[0]

            if len(inds_jump) == 0:
                ind_jump = 0
            else:
                ind_jump = inds_jump[0]

            inds_degenerate_eval = np.arange(ind_ev, ind_ev+ind_jump)
            if ind_jump > 1:
                evecs_ladder_ops[:,inds_degenerate_eval] = sparsify(evecs_ladder_ops[:,inds_degenerate_eval])

                ind_ev += ind_jump
            else:
                ind_ev += 1
        
    if not return_operators:
        return (evecs_ladder_ops, evals_ladder_ops)
    else:
        ladder_ops = []
        if isinstance(basis, Basis):
            for ind_ev in range(num_ladder_ops):
                ladder_op = Operator(evecs_ladder_ops[:, ind_ev], basisAB.op_strings)
                ladder_ops.append(ladder_op)

            #print('Ladder operators:')
            #for op in ladder_ops:
            #    print(op)
        elif isinstance(basis, list) and isinstance(basis[0], Operator):
            raise NotImplementedError('A basis made of a list of Operators is not supported yet.')
        else:
            raise ValueError('Invalid basis of type: {}'.format(type(basis)))

        return (ladder_ops, evals_ladder_ops)

def _ladder_operators(basis, operator, sparsification=False, tol=1e-12, return_operators=True):
    # Alternative version of _ladder_operators().
    
    basisA = basis

    # Liouvillian matrix between basisA and basisB.
    (L_A_B, basisB) = liouvillian_matrix(basisA, -operator, return_extended_basis=True)

    # The smallest basis that includes only operators from basisA and basisB.
    smallest_basisAB_opstrings = [os for os in basisA if os in basisB]
    smallest_basisAB           = Basis(smallest_basisAB_opstrings)

    if len(smallest_basisAB) == 0:
        if return_operators:
            evecs_ladder_ops = []
        else:
            evecs_ladder_ops = np.zeros((len(basisA), 0), dtype=complex)
            
        return (evecs_ladder_ops, np.zeros(0, dtype=complex))
    
    # Find the relevant indices of operator strings in different bases.
    inds_os_AB_in_A    = []
    inds_os_AB_in_B    = []
    inds_os_notAB_in_A = []
    inds_os_notAB_in_B = []
    ind_os_B  = 0
    for os in basisB:
        if os in basisA:
            inds_os_AB_in_A.append(basisA.index(os))
            inds_os_AB_in_B.append(ind_os_B)
        else:
            inds_os_notAB_in_B.append(ind_os_B)
        ind_os_B += 1
    ind_os_A = 0
    for os in basisA:
        if os not in basisB:
            inds_os_notAB_in_A.append(ind_os_A)
        ind_os_A += 1
        
    # The full Liouvillian matrix.
    L = L_A_B #.toarray()

    # The original Liouvillian matrix projected into the smallest
    # basis made of operators in both basisA and basisB.
    # L_AB: A \intersect B -> A \intersect B
    L_AB = L[inds_os_AB_in_B, :]
    L_AB = L_AB[:, inds_os_AB_in_A].toarray()

    print('basisA     = \n{}'.format(basisA))
    print('basisB     = \n{}'.format(basisB))
    print('smallest_basisAB = \n{}'.format(smallest_basisAB))
    
    if len(inds_os_notAB_in_B) > 0:
        # The Liouvillian matrix whose output is projected out of basisAB.
        # L_notAB: A \intersect B -> B/(A \intersect B) = B/A
        L_notAB = L[inds_os_notAB_in_B, :]
        L_notAB = L_notAB[:, inds_os_AB_in_A].toarray()

        # Perform SVD on the L_notAB to determine the right null vectors.
        # These are the vectors we care about: they stay in basisAB after
        # applying the Liouvillian matrix.
        (left_svecs, svals, right_svecsH) = nla.svd(L_notAB)

        inds_zero_svals = np.where(np.abs(svals) < tol)[0]
        print('inds_zero_svals = {}'.format(inds_zero_svals))
        valid_vecs      = np.conj(np.transpose(right_svecsH))[:, inds_zero_svals]
        
        print('svals      = {}'.format(svals))
        #print('valid_vecs = {}'.format(valid_vecs))

        # The L_A matrix projected onto the valid vectors that stay in basisA.
        projected_L = np.dot(np.conj(np.transpose(valid_vecs)), np.dot(L_AB, valid_vecs))
    else:
        projected_L = L_AB

    print('projected_L = {}'.format(projected_L))
    check_hermitian = np.allclose(np.conj(np.transpose(projected_L))-projected_L, np.zeros(projected_L.shape))
    print('projected_L is Hermitian: {}'.format(check_hermitian))
        
    # Perform eigendecomposition on the projected L matrix.
    (evals, evecs) = nla.eigh(0.5*(projected_L + np.conj(np.transpose(projected_L))))

    print('evals = {}'.format(evals))
    #print('evecs = {}'.format(evecs))

    # Find all of the positive eigenvalues that
    # come paired with a negative eigenvalue partner.
    inds_pos           = np.where(np.logical_and(np.imag(evals) < tol, np.real(evals) > tol))[0]
    inds_pos_partnered = []
    for ind_pos_ev in inds_pos:
        inds_neg_ev = np.where(np.abs(evals[ind_pos_ev] + evals) < tol)[0]
        if len(inds_neg_ev) > 0: 
            inds_pos_partnered.append(ind_pos_ev)

    print('inds_pos           = {}'.format(inds_pos))
    print('inds_pos_partnered = {}'.format(inds_pos_partnered))
            
    ladder_evals = np.real(evals[inds_pos_partnered])
    if len(inds_os_notAB_in_B) > 0:
        ladder_evecsAB = np.dot(valid_vecs, evecs[:, inds_pos_partnered])
    else:
        ladder_evecsAB = evecs[:, inds_pos_partnered]
        
    ladder_evecs                     = np.zeros((len(basisA), len(ladder_evals)), dtype=complex) 
    ladder_evecs[inds_os_AB_in_A, :] = ladder_evecsAB

    # Post-process the eigenvalues and eigenvectors.
    inds_sort = np.argsort(ladder_evals)

    evals_ladder_ops = ladder_evals[inds_sort]
    evecs_ladder_ops = ladder_evecs[:, inds_sort]

    num_ladder_ops = len(evals_ladder_ops)
    if num_ladder_ops == 0:
        if return_operators:
            evecs_ladder_ops = []
            
        return (evecs_ladder_ops, evals_ladder_ops)
    
    """
    # Perform SVD on the Liouvillian matrix.
    (left_svecsB, svals, right_svecsAH) = nla.svd(L)
    right_svecsA = np.conj(np.transpose(right_svecsAH))

    # Identify the degenerate positive singular values subspaces.
    degenerate_subspace_inds = []
    degenerate_subspace_sval = []
    visited_inds = set()
    for ind_sv in range(len(svals)):
        subspace_inds = []
        inds_deg = np.where(np.logical_and(np.abs(svals[ind_sv]-svals) < tol, np.abs(svals) > tol))[0]
        for ind_deg in inds_deg:
            if ind_deg not in visited_inds:
                subspace_inds.append(ind_deg)
                visited_inds.add(ind_deg)
        if len(subspace_inds) > 0:
            degenerate_subspace_inds.append(subspace_inds)
            degenerate_subspace_sval.append(svals[ind_sv])

    

    projector_B_to_A = np.zeros((len(basisA), len(basisB)), dtype=complex)
    
    for ind_os_b in range(len(basisB)):
        os_b = basisB.op_strings[ind_os_b]
        if os_b in basisA:
            ind_os_a                             = basisA.index(os_b)
            projector_B_to_A[ind_os_a, ind_os_b] = 1.0

    projected_L = np.dot(projector_B_to_A, L)

    # For each degenerate subspace, project onto the right
    # singular vectors with singular value s and see if
    # the eigenvalues of the projected matrix are +/- s.
    # If they are, then the projection preserved the
    # singular vectors as eigenvectors of the projected
    # matrix.
    num_ladder_ops   = 0
    evecs_ladder_ops = []
    evals_ladder_ops = []
    for (subspace_inds, subspace_sval) in zip(degenerate_subspace_inds, degenerate_subspace_sval):
        if len(subspace_inds) > 1:    
            right_vecs = right_svecsA[:, subspace_inds]
            
            subspace_L = np.dot(np.conj(np.transpose(right_vecs)), np.dot(projected_L, right_vecs))

            (evals_sub, evecs_sub) = nla.eig(subspace_L)

            inds_eval_matches_sval  = np.where(np.abs(evals_sub - subspace_sval) < tol)[0]
            inds_eval_matches_msval = np.where(np.abs(evals_sub + subspace_sval) < tol)[0]

            assert(len(inds_eval_matches_sval) == len(inds_eval_matches_msval))

            if len(inds_eval_matches_sval) > 0:
                evals_ladder_ops.append(evals_sub[inds_eval_matches_sval])
                evecs_ladder_ops.append(evecs_sub[inds_eval_matches_sval])

                num_ladder_ops += len(inds_eval_matches_sval)
                
    if num_ladder_ops == 0:
        if return_operators:
            evecs_ladder_ops = []
        else:
            evecs_ladder_ops = np.zeros((len(basisA), 0), dtype=complex)
            
        return (evecs_ladder_ops, np.zeros(0, dtype=complex))
    
    evals_ladder_ops = np.concatenate(tuple(evals_ladder_ops))
    evecs_ladder_ops = np.hstack(tuple(evecs_ladder_ops))
    """
    
    # Go through each degenerate subspace and sparsify the basis of that subspace.
    if sparsification:
        inds_sort = np.argsort(np.real(evals_ladder_ops))
        evals_ladder_ops = evals_ladder_ops[inds_sort]
        evecs_ladder_ops = evecs_ladder_ops[:, inds_sort]

        ind_ev = 0
        while ind_ev < len(evals_ladder_ops):
            inds_jump = np.where(np.abs(evals_ladder_ops[ind_ev:] - evals_ladder_ops[ind_ev]) > 1e-10)[0]

            if len(inds_jump) == 0:
                ind_jump = 0
            else:
                ind_jump = inds_jump[0]

            inds_degenerate_eval = np.arange(ind_ev, ind_ev+ind_jump)
            if ind_jump > 1:
                evecs_ladder_ops[:,inds_degenerate_eval] = sparsify(evecs_ladder_ops[:,inds_degenerate_eval])

                ind_ev += ind_jump
            else:
                ind_ev += 1
        
    if not return_operators:
        return (evecs_ladder_ops, evals_ladder_ops)
    else:
        ladder_ops = []
        if isinstance(basis, Basis):
            for ind_ev in range(num_ladder_ops):
                ladder_op = Operator(evecs_ladder_ops[:, ind_ev], basisA.op_strings)
                ladder_ops.append(ladder_op)

            #print('Ladder operators:')
            #for op in ladder_ops:
            #    print(op)
        elif isinstance(basis, list) and isinstance(basis[0], Operator):
            raise NotImplementedError('A basis made of a list of Operators is not supported yet.')
        else:
            raise ValueError('Invalid basis of type: {}'.format(type(basis)))

        return (ladder_ops, evals_ladder_ops)

def _inverse_ladder_operators(basis, ladder_operator, sparsification=True, tol=1e-12, return_operators=True):
    """Find operators :math:`\\hat{O}` in the vector space spanned by the
    OperatorStrings :math:`\\hat{\\mathcal{S}}_a` that have  
    :math:`\\hat{\\mathcal{R}} = \\sum_{a} g_a \\hat{\\mathcal{S}}_a` as
    a ladder operator.

    These operators satisfy :math:`[\\hat{O}, \\hat{R}] = (\\delta O) \\hat{R}`
    where :math:`\\delta O > 0`.

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings :math:`\\hat{\\mathcal{S}}_a` to search in.
    operator : Operator
        The Operator :math:`\\hat{\\mathcal{O}}` to be a raising operator of.
    sparsification : bool, optional
        Specifies whether to sparsify vectors during the calculations. 
        Defaults to True.
    tol : float, optional
        The numerical cutoff used to determine what eigenvectors
        are in a null space.
    return_operators : bool, optional
        Flag that specifies whether to return a list of Operators
        instead of numpy array representing vectors in the given basis. 
        Defaults to True.

    Returns
    -------
    (Operator or ndarray, list of Operators or ndarrays)
        Returns a tuple of the particular solution and the homogenous
        solutions for the problem.
        If `return_operators` is False, the tuples contain numpy arrays
        whose columns are operators in the given Basis.
        If `return_operators` is True, they contain lists of Operators.
    """
    
    basisA = basis
    
    # The Liouvillian matrix for commuting with the ladder operator.
    (L, basisB) = liouvillian_matrix(basisA, ladder_operator, return_extended_basis=True)
    
    # The right null vectors of the Liouvillian matrix.
    (left_svecs, svals, right_svecs_dag) = nla.svd(L.toarray())
    inds_null_space = np.where(np.abs(svals) < tol)[0]
    right_null_vecs = np.conj(right_svecs_dag.T)[:, inds_null_space]

    if sparsification:
        right_null_vecs = sparsify(right_null_vecs)
    
    # Check if the ladder operator can be expressed in basisB.
    check_ladder_op_in_B = True
    for (coeff, os) in ladder_operator:
        if os not in basisB:
            check_ladder_op_in_B = False
            break
    # If it is not in basisB, there are only
    # homogenous solutions (symmetries) and
    # no particular solution.
    if not check_ladder_op_in_B:
        return (None, right_null_vecs)
    
    vector_ladder_op = ladder_operator.to_vector(basisB)
    
    # The pseudo-inverse of the Liouvillian matrix.
    L_pinv = nla.pinv(L.toarray())

    # The particular solution.
    vector_inverse_op = np.dot(L_pinv, vector_ladder_op)

    # TODO: modify. Keep only the real null vectors.

    print('basisA:\n{}\nbasisB:\n{}'.format(basisA,basisB))
    print('L  = \n{}'.format(L.toarray()))
    print('L+ = \n{}'.format(L_pinv))
    print('vector_ladder_op  = \n{}'.format(vector_ladder_op))
    print('vector_inverse_op = \n{}'.format(vector_inverse_op))
    
    if not return_operators:
        vector_inverse_op /= nla.norm(vector_inverse_op)
        return (vector_inverse_op, right_null_vecs)
    else:
        inverse_op = Operator(vector_inverse_op, basisA.op_strings)
        inverse_op = inverse_op.remove_zeros(tol=tol)
        inverse_op.normalize()
        
        null_vec_ops = []
        if isinstance(basis, Basis):
            for ind_ev in range(right_null_vecs.shape[1]):
                null_vec_op = Operator(right_null_vecs[:, ind_ev], basisA.op_strings)
                null_vec_ops.append(null_vec_op)
        elif isinstance(basis, list) and isinstance(basis[0], Operator):
            raise NotImplementedError('A basis made of a list of Operators is not supported yet.')
        else:
            raise ValueError('Invalid basis of type: {}'.format(type(basis)))

        return (inverse_op, null_vec_ops)
