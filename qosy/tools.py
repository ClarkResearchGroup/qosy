#!/usr/bin/env python
import numpy as np
import numpy.linalg as nla

def sort_sign(vector, tol=1e-10):
    """Stable sort the vector and return
    the sign of the permutation needed to sort it.

    Parameters
    ----------
    vector : ndarray of numbers
        The vector to sort.
    
    Returns
    -------
    (array_like, int)
        A tuple of (a new copy of) the sorted vector
        and the sign of the permutation needed to sort it.

    Examples
    --------
       >>> arr = numpy.array([1, -0.3, 141])
       >>> (sorted_arr, sign) = qosy.tools.sort_sign(arr)
       >>> sorted_arr # numpy.array([-0.3, 1, 141])
       >>> sign       # -1
    """
    
    vec   = np.copy(vector)
    
    n     = len(vec)
    swaps = 0

    for i in range(1,n):
        j = i
        while j > 0 and np.abs(vec[j-1] - vec[j]) > tol and vec[j-1] > vec[j]:
            # Swap if the elements are not identical.
            vec[j], vec[j-1] = vec[j-1], vec[j]
            swaps += 1
            j -= 1
                
    if swaps % 2 == 1:
        return (vec, -1)
    else:
        return (vec, 1)

def compare(labelsI, labelsJ):
    """Lexicographically compare two sets of labels, :math:`(i_1,\ldots,i_m)` 
    and :math:`(j_1,\ldots,j_l)`. 

    If :math:`m < l`, then :math:`(i_1,\ldots,i_m) < (j_1,\ldots,j_l)`. 
    If :math:`m = l`, then you compare :math:`i_1` and :math:`j_1`. 
    If those are equal, you compare :math:`i_2` and :math:`j_2`, and so on.

    Parameters
    ----------
    labelsI, labelsJ : lists, tuples, or ndarray of ints
        The tuples of labels to compare.

    Returns
    -------
    int
        Returns 1 if `labelsI` > `labelsJ`, 
        -1 if `labelsI` < labelsJ`, and 0 otherwise. 
    """
    
    m = len(labelsI)
    l = len(labelsJ)
    if m < l:
        return -1 # labelsI < labelsJ
    elif m > l:
        return 1  # labelsI > labelsJ
    else:
        for ind in range(m):
            iInd = labelsI[ind]
            jInd = labelsJ[ind]
            if iInd < jInd:
                return -1 # labelsI < labelsJ
            elif iInd > jInd:
                return 1  # labelsI > labelsJ
        
    return 0 # labelsI == labelsJ

def gram_schmidt(matrix, tol=1e-10):
    """Perform Gram-Schmidt decomposition.

    Parameters
    ----------
    matrix : ndarray or scipy.sparse matrix
        A matrix whose columns we want to orthogonalize
        going from left to right.

    Returns
    -------
    ndarray
        A unitary matrix whose columns are orthonormal vectors that
        form an orthonormal basis of the column space of the given matrix.
        The number of columns of the returned matrix can be smaller than
        the given matrix if the given matrix is noninvertible.
    """
    
    newMatrix = np.copy(matrix)
    n = int(matrix.shape[0])
    k = int(matrix.shape[1])

    shift = 0
    newMatrix[:,0] = matrix[:,0] / nla.norm(matrix[:,0])
    
    for indColUnshifted in range(1,k):
        indCol = indColUnshifted - shift
        newMatrix[:,indCol] = matrix[:,indColUnshifted]
        for indVec in range(indCol):
            newMatrix[:,indCol] = newMatrix[:,indCol] - np.vdot(newMatrix[:,indCol], newMatrix[:,indVec]) * newMatrix[:,indVec]

        norm = nla.norm(newMatrix[:,indCol])
        if norm > tol:
            newMatrix[:,indCol] = newMatrix[:,indCol] / norm
        else:
            # Found a column that is linearly dependent on the previously found ones.
            shift += 1
        
    return newMatrix[:,0:(k-shift)]

def intersection(A, B, tol=1e-10):
    """Find the intersection of two vector spaces. 
    
    Parameters
    ----------
    A : ndarray
        A matrix whose column vectors form 
        an orthonormal basis of vector space :math:`V`.
    B : ndarray
        Another matrix whose column vectors 
        form an orthonormal basis of vector space :math:`W`.

    Returns
    -------
    ndarray
        A matrix whose column vectors form 
        an orthonormal basis of the vector space :math:`V\cap W`.
    """
    
    dimVS = int(A.shape[0])
    if dimVS != int(B.shape[0]):
        raise ValueError('A and B do not agree in size: {} {}'.format(A.shape, B.shape))
    
    dimA  = int(A.shape[1])
    dimB  = int(B.shape[1])
    
    C = np.hstack((A, -B))

    D = np.dot(np.conj(C.T), C)

    (evals, evecs) = nla.eigh(D)
    indsNS    = np.where(np.abs(evals) < tol)[0]
    null_vecsA = evecs[0:dimA, indsNS]

    intersectionAB = np.dot(A, null_vecsA)
    dimAB          = int(intersectionAB.shape[1])
    
    for indVec in range(dimAB):
        vec = intersectionAB[:,indVec]
        intersectionAB[:,indVec] /= nla.norm(vec)
    
    return intersectionAB

