#!/usr/bin/env python
import warnings
import numpy as np
import numpy.linalg as nla
import scipy.sparse.linalg as ssla

from .tools import intersection
from .algebra import commutant_matrix
from .transformation import symmetry_matrix

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
    ndarray or (scipy.sparse.csc_matrix, scipy.sparse.csc_matrix)
        If `return_com_matrix` is False, returns a numpy array
        whose columns are operators in the given Basis. In particular,
        the column vectors :math:`J^{(1)},\ldots,J^{(M)}` are the 
        coefficients :math:`J_a^{(i)}` of operators 
        :math:`\hat{H}^{(i)}=\sum_{a} J_a^{(i)} \hat{h}_a` that commute 
        (or anti-commute) with the given Operator :math:`\hat{\mathcal{O}}`.
        If `return_com_matrix` is True, returns a tuple of the null
        vectors, the commutant matrix, its eigenvalues, and its eigenvectors
        as sparse scipy matrices.
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
        return (null_space, com_matrix, evals, evecs)
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
    def __init__(self, basis, num_vecs=None):
        # Input
        self.basis            = basis
        self.operation_modes  = []
        self.input_symmetries = []

        # Output
        self.superoperators   = []
        self.eigenvectors     = []
        self.eigenvalues      = []
        self.output_operators = []

        # Projected output
        #self.projected_superoperators   = []
        #self.projected_eigenvalues      = []
        #self.projected_eigenvectors     = []
        self.projected_output_operators = []

        # Specifies whether to use scipy.sparse.linalg.eigh
        # or numpy.linalg.eigh.
        self.num_vecs = num_vecs
    
    def add_symmetry(self, symmetry, operation_mode='commutator'):
        self.operation_modes.append(operation_mode)
        self.input_symmetries.append(symmetry)

    def generate(self, verbose=True):
        num_inputs = len(self.input_symmetries)

        if verbose:
            print('===== GENERATING OPERATORS =====')
        # Go through the Operators or Transformations, one by one,
        # in order, and find the null spaces of the (anti-)commutant matrix
        # or the +/-1 eigenspaces of the symmetry matrix. Save the results.
        for ind_input in range(num_inputs):
            input_symmetry = self.input_symmetries[ind_input]
            operation_mode = self.operation_modes[ind_input]

            op_mode_print = ' COMMUTE WITH '
            if operation_mode == 'anticommutator':
                op_mode_print = ' ANTICOMMUTE WITH '
            
            if isinstance(input_symmetry, Operator):
                if verbose:
                    print('{} OPERATOR {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, com_matrix, eigvals, eigvecs) = commuting_operators(basis, input_symmetry, operation_mode=operation_mode, return_com_matrix=True, num_vecs=self.num_vecs)

                self.superoperators.append(com_matrix)
                self.eigenvectors.append(eigvecs)
                self.eigenvalues.append(eigvals)
                self.output_operators.append(output_ops)

                if verbose:
                    dim_output = int(self.output_operators.shape[1])
                    print(' Generated a vector space of operators of dimension: {}'.format(dim_output))
                
            elif isinstance(input_symmetry, Transformation):
                if verbose:
                    print('{} OPERATOR {}'.format(op_mode_print, ind_input+1))
                
                (output_ops, sym_matrix, eigvals, eigvecs) = invariant_operators(basis, input_symmetry, operation_mode=operation_mode, return_sym_matrix=True, num_vecs=self.num_vecs)

                self.superoperators.append(sym_matrix)
                self.eigenvectors.append(eigvecs)
                self.eigenvalues.append(eigvals)
                self.output_operators.append(output_ops)

                if verbose:
                    dim_output = int(self.output_operators.shape[1])
                    print(' Generated a vector space of operators of dimension: {}'.format(dim_output))
            else:
                raise ValueError('Invalid input symmetry of type {}. Must be an Operator or a Transformation.'.format(type(input_symmetry)))


        if verbose:
            print('===== POST-PROCESSING: PROJECTION =====')
        # Post processing: Go through the output vector spaces
        # V_1,V_2,...,V_N and project them in order:
        # Project V_2 into V_1. Project V_3 into V_1 and V_2, etc.
        for ind_output in range(num_inputs):
            if ind_output == 0:
                # Do not project the first iteration.
                self.projected_output_operators[ind_output] = self.output_operators[ind_output]
            else:
                # Project onto the previous iteration's results.
                curr_ops   = self.output_operators[ind_output]
                prev_ops   = self.projected_output_operators[ind_output]
                
                self.projected_output_operators[ind_output] = intersection(curr_ops, prev_ops)

            if verbose:
                dim_output = int(self.output_operators.shape[1])
                print(' ({}) The projected vector space of operators has dimension: {}'.format(ind_output+1, dim_output))

        if verbose:
            print('===== POST-PROCESSING: ORTHOGONALIZATION =====')
        # Post processing: Go through the projected output
        # vector spaces in reverse order, V_N', V_{N-1}', ..., V_1', and use the
        # Gram-Schmidt procedure to orthogonalize them.
        # Orthogonalize V_{N-1}' against V_N'. Orthogonalize V_{N-2}' against V_{N-1}', etc.
        # The goal here is to find a nicer basis for each of the vector spaces.
        for ind_output in range(num_inputs-2,-1,-1):
            # Orthogonalize against the previous iteration's results.
            curr_ops   = self.projected_output_operators[ind_output]
            prev_ops   = self.projected_output_operators[ind_output+1]
                
            self.projected_output_operators[ind_output] = gram_schmidt(curr_ops, prev_ops)

            if verbose:
                dim_curr = int(curr_ops.shape[1])
                dim_prev = int(prev_ops.shape[1])
                print(' ({}) Orthogonalized a vector space of dim {} against one with dim {}'.format(ind_output+1, dim_curr, dim_prev))
        
