#!/usr/bin/env python
import warnings
import numpy as np
import numpy.linalg as nla
import scipy.sparse.linalg as ssla

from .algebra import commutant_matrix
from .transformation import symmetry_matrix

def commuting_operators(basis, operator, operation_mode='commutator', num_vecs=None, return_com_matrix=False, _sigma=None, _tol=1e-10):
    com_matrix = commutant_matrix(basis, operator, operation_mode=operation_mode)

    CDagC = (com_matrix.H).dot(com_matrix)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(CDagC.toarray())
    else:
        # Use a small negative sigma for the shift-invert method.
        if _sigma is None:
            _sigma = -1e-2
        (evals, evecs) = ssla.eigsh(CDagC, k=num_vecs, sigma=_sigma)

    inds_ns    = np.where(np.abs(evals) < _tol)[0]
    null_space = evecs[:,inds_ns]

    if num_vecs is not None and len(inds_ns) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the null space. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs))

    if return_com_matrix:
        return (null_space, com_matrix)
    else:
        return null_space

def invariant_operators(basis, transform, operation_mode='commutator', num_vecs=None, return_sym_matrix=False, _tol=1e-10):
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

    inds = np.where(np.abs(evals - sign) < _tol)[0]
    vecs = evecs[:,inds]

    if num_vecs is not None and len(inds) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the ({})-eigenvalue subspace. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs, sign))

    if return_sym_matrix:
        return (vecs, sym_matrix)
    else:
        return vecs
