#!/usr/bin/env python
import warnings
import numpy as np
import numpy.linalg as nla
import scipy.sparse.linalg as ssla

import algebra
import transformation

def commuting_operators(basis, operator, operation_mode='commutator', num_vecs=None, _sigma=None, _tol=1e-10):
    commutant_matrix = algebra.commutant_matrix(basis, operator, operation_mode=operation_mode)

    CDagC = (commutant_matrix.H).dot(commutant_matrix)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(CDagC.todense())
    else:
        # Use a small negative sigma for the shift-invert method.
        if _sigma is None:
            _sigma = -1e-2
        (evals, evecs) = ssla.eigsh(CDagC, k=num_vecs, sigma=_sigma)

    inds_ns    = np.where(np.abs(evals) < _tol)[0]
    null_space = evecs[:,inds_ns]

    if num_vecs is not None and len(inds_ns) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the null space. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs))

    return null_space

def invariant_operators(basis, transform, operation_mode='commutator', num_vecs=None, _tol=1e-10):
    symmetry_matrix = transformation.symmetry_matrix(basis, transform)
    
    if operation_mode in ['commute', 'Commute', 'symmetry', 'commutator']:
        sign = 1.0
    elif operation_mode in ['anticommute', 'Anticommute', 'antisymmetry', 'anticommutator']:
        sign = -1.0
    else:
        raise ValueError('Unknown operation_mode: {}'.format(operation_mode))

    matrix = 0.5*(symmetry_matrix + symmetry_matrix.H)

    if num_vecs is None:
        (evals, evecs) = nla.eigh(matrix.todense())
    else:
        (evals, evecs) = ssla.eigsh(matrix, k=num_vecs, sigma=sign)

    inds = np.where(np.abs(evals - sign) < _tol)[0]
    vecs = evecs[:,inds]

    if num_vecs is not None and len(inds) == num_vecs:
        warnings.warn('All {} vectors found with eigsh were in the ({})-eigenvalue subspace. Increase num_vecs to ensure that you are not missing null vectors.'.format(num_vecs, sign))
    
    return vecs
