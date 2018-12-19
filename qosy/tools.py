#!/usr/bin/env python
"""
This module provides miscellaneous methods that are helpful
throughout ``qosy``.
"""

import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla

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

def maximal_cliques(adjacency_lists):
    """Find the maximal cliques of an undirected graph.

    Parameters
    ----------
    adjacency_lists : list of list of int
        Specifies the neighbors of each node in the graph.

    Returns
    -------
    list of list of int
        The maximal cliques of the graph.

    Notes
    -----
    The adjacency lists must be valid. If invalid,
    might enter an infinite recursion.
    
    """

    # Implementation of the Bron-Kerbosch algorithm described
    # on wikipedia: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    
    num_nodes = len(adjacency_lists)

    adjacency_sets = [set(adj_list) for adj_list in adjacency_lists]
    
    degeneracies = [len(adj_list) for adj_list in adjacency_lists]
    degeneracy_ordering = np.argsort(degeneracies)[::-1]

    P = set(np.arange(num_nodes))
    R = set()
    X = set()

    cliques = []

    def BronKerbosch2(R, P, X):
        #print('R={},P={},X={}'.format(R,P,X))
        if len(P) == 0 and len(X) == 0:
            cliques.append(list(R))
            return R

        PunionX = P.union(X)
        u = next(iter(PunionX))
        for v in set(P):
            if v not in adjacency_sets[u]:
                BronKerbosch2(R.union(set([v])), P.intersection(adjacency_sets[v]), X.intersection(adjacency_sets[v]))
                P.remove(v)
                X = X.union(set([v]))
            
    for v in degeneracy_ordering:
        BronKerbosch2(R.union(set([v])), P.intersection(adjacency_sets[v]), X.intersection(adjacency_sets[v]))
        P.remove(v)
        X = X.union(set([v]))

    return cliques

def argsort(mylist):
    """Returns the indices that sort a list.

    Parameters
    ----------
    list of comparable objects
        List to sort.

    Returns
    -------
    list of int
        The permutation that sorts the list.
    """

    # Based on https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python
    return sorted(range(len(mylist)), key=mylist.__getitem__)

def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function.
    """

    # From https://docs.python.org/3/howto/sorting.html#sortinghowto
    class K:
        def __init__(self, obj, *args):
            self.obj = obj
        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0
        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0
        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0
        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0
        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0
        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0
    return K

def swap(string, nameA, nameB):
    """ Swap all occurences of nameA with nameB
    and vice-versa in the string.

    Parameters
    ----------
    string : str
        The string to manipulate.
    nameA : str
        The substring to replace with `nameB`.
    nameB : str
        The substring to replace with `nameA`.

    Returns
    -------
    str
        The modified string.

    Examples
    --------
        >>> swap('A Up B Dn', 'Up', 'Dn') # 'A Dn B Up'
        >>> swap('X X A', 'X', 'Y') # 'Y Y A'
        >>> swap('1 2 3', '1', '3') # '3 2 1'
    """
    
    result = string.replace(nameA, '({})'.format(nameA))
    result = result.replace(nameB, nameA)
    result = result.replace('({})'.format(nameA), nameB)

    return result

def gram_schmidt(matrix, tol=0.0):
    """Perform Gram-Schmidt decomposition.

    Parameters
    ----------
    matrix : ndarray or scipy.sparse matrix
        A matrix whose columns we want to orthogonalize
        going from left to right.
    tol : float, optional
        Specifies the tolerance used in the algorithm for
        discarding vector which are not orthogonal. Defaults to
        zero, in which case vectors are not discarded.

    Returns
    -------
    ndarray
        A unitary matrix whose columns are orthonormal vectors that
        form an orthonormal basis of the column space of the given matrix.
        The number of columns of the returned matrix can be smaller than
        the given matrix if the given matrix is noninvertible.
    """
    
    new_matrix = np.copy(matrix)
    n = int(matrix.shape[0])
    k = int(matrix.shape[1])

    if n==0 or k==0:
        return new_matrix

    shift = 0
    new_matrix[:,0] = matrix[:,0] / nla.norm(matrix[:,0])
    
    for indColUnshifted in range(1,k):
        indCol = indColUnshifted - shift
        new_matrix[:,indCol] = matrix[:,indColUnshifted]
        for indVec in range(indCol):
            new_matrix[:,indCol] = new_matrix[:,indCol] - np.vdot(new_matrix[:,indCol], new_matrix[:,indVec]) * new_matrix[:,indVec]

        norm = nla.norm(new_matrix[:,indCol])
        
        if norm > tol:
            new_matrix[:,indCol] = new_matrix[:,indCol] / norm
        else:
            # Found a column that is linearly dependent on the previously found ones.
            shift += 1
        
    return new_matrix[:,0:(k-shift)]

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

def sparsify(vectors, orthogonalize=True, tol=1e-12):
    """Heuristically compute a sparsified representation 
    of the vectors, i.e., a new basis of sparser vectors 
    that span the same vector space.

    Parameters
    ----------
    vectors : ndarray or scipy.sparse.csc_matrix
        The vectors to sparsify.
    orthogonalize : bool, optional
        Specifies whether to make the sparsified vectors orthogonal.
        In general, orthogonalization makes the vectors denser.
        Defaults to True.
    tol : float, optional
        The tolerance with which numbers are considered zero.

    Returns
    -------
    ndarray
        A matrix whose columns are the sparsified vectors.
    """

    num_vectors = int(vectors.shape[1])
    
    if num_vectors <= 1:
        return vectors
    
    # Perform an LU decomposition, which
    # will do a reduced-row echelon form
    # decomposition. The L matrix corresponds
    # to a new set of vectors that are
    # sparser than the original (but, are
    # not guaranteed to be optimally sparse).
    (vectors_rre, _) = sla.lu(vectors, permute_l=True)

    if orthogonalize:
        vectors_rre = gram_schmidt(vectors_rre[:, ::-1])
    else:
        vectors_rre = vectors_rre[:, ::-1]
        
    # Normalize the vectors.
    for ind_vec in range(num_vectors):
        vectors_rre[:, ind_vec] /= nla.norm(vectors_rre[:, ind_vec])
    
    return vectors_rre
