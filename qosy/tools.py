#!/usr/bin/env python
"""
This module provides miscellaneous methods that are helpful
throughout ``qosy``.
"""

import re
import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.linalg as sla

def sign(permutation):
    """Compute the sign of a permutation.

    Parameters
    ----------
    permutation : ndarray of int
        The permutation whose sign to compute.

    Returns
    -------
    int
        +1 or -1, depending on whether the
        permutation is even or odd.

    Examples
    --------
        >>> sign(np.array([1,0,2],dtype=int)) # -1
    """

    # Identify the cycles and count their lengths.
    # Even (odd) length cycles are made of an odd (even)
    # number of transpositions, so contribute a -1 (+1).
    visited = set()
    cycle_length = 0
    sign = 1
    
    for ind in range(len(permutation)):
        if ind not in visited:
            visited.add(ind)
            cycle_length = 1
            ind_cycle = permutation[ind]
            while ind_cycle not in visited:
                cycle_length += 1
                visited.add(ind_cycle)
                ind_cycle = permutation[ind_cycle]

            if cycle_length % 2 == 0:
                sign *= -1
                
    return sign

def sort_sign(vector, tol=1e-10, method='insertionsort'):
    """Stable sort the vector and return
    the sign of the permutation needed to sort it.

    Parameters
    ----------
    vector : list or ndarray of numbers
        The vector to sort.
    method : str, optional
        The sorting method used, 'insertionsort'
        or 'mergesort'. Defaults to 'insertionsort'.
    
    Returns
    -------
    (ndarray, int)
        A tuple of (a new copy of) the sorted vector
        and the sign of the permutation needed to sort it.

    Examples
    --------
       >>> arr = numpy.array([1, -0.3, 141])
       >>> (sorted_arr, sign) = qosy.tools.sort_sign(arr)
       >>> sorted_arr # numpy.array([-0.3, 1, 141])
       >>> sign       # -1
    """

    if method == 'mergesort':
        # Using numpy's mergesort.
        vec = np.array(vector)

        permutation = np.argsort(vec, kind='mergesort')
        sign_perm   = sign(permutation)

        return (vec[permutation], sign_perm)
    elif method == 'insertionsort':
        # Using insertion sort.
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
    else:
        raise ValueError('Invalid method: {}'.format(method))

def compare(labelsI, labelsJ):
    """Lexicographically compare two sets of labels, :math:`(i_1,\\ldots,i_m)` 
    and :math:`(j_1,\\ldots,j_l)`. 

    If :math:`m < l`, then :math:`(i_1,\\ldots,i_m) < (j_1,\\ldots,j_l)`. 
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

def connected_components(adjacency_lists, mode=None):
    """Find the connected components of an undirected graph.

    Parameters
    ----------
    adjacency_lists : list of list of int
        Specifies the neighbors of each node in the graph.

    mode : str, optional
        Specifies whether to do a breadth-first-search ('BFS')
        or depth-first-search ('DFS'). Default is 'DFS'.

    Returns
    -------
    list of list of int
        The connected components of the graph.
    
    """

    if mode is None:
        mode = 'DFS'

    num_nodes = len(adjacency_lists)
    
    visited  = set()
    connected_components = []
    for node in range(num_nodes):
        if node not in visited:
            to_visit            = [node]
            to_visit_set        = set(to_visit)
            connected_component = []
            
            while len(to_visit) > 0:
                if mode == 'DFS':
                    curr_node = to_visit.pop()
                elif mode == 'BFS':
                    curr_node = to_visit.pop(0)
                else:
                    raise ValueError('Invalid mode: {}'.format(mode))
                to_visit_set.remove(curr_node)
                
                visited.add(curr_node)
                connected_component.append(curr_node)
                
                for neighbor in adjacency_lists[curr_node]:
                    if (neighbor not in visited) and (neighbor not in to_visit_set):
                        to_visit.append(neighbor)
                        to_visit_set.add(neighbor)
                
            if len(connected_component) > 0:
                connected_components.append(connected_component)

    return connected_components

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

def argsort(mylist, comp=None):
    """Returns the indices that sort a list.

    Parameters
    ----------
    mylist : list of objects
        List to sort.
    comp : function, optional
        A comparison function used
        to compare two objects in the list.
        Defaults to None.

    Returns
    -------
    list of int
        The permutation that sorts the list.
    """

    # Based on https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python

    if comp is None:
        return sorted(range(len(mylist)), key=mylist.__getitem__)
    else:
        return sorted(range(len(mylist)), key=cmp_to_key(comp))

def remove_duplicates(objects, equiv=None, tol=1e-12):
    """Remove duplicate objects from
    a list of objects.

    Parameters
    ----------
    objects : list of objects
        A list of objects, 
        with possible duplicates.
    equiv : function, optional
        A function for checking equality
        of objects. Defaults to numpy.allclose
        for numpy arrays.
    tol : float, optional
        The tolerance within which to
        consider numpy arrays
        identical. Defaults to 1e-12.
    
    Returns
    -------
    list of objects
        The list with duplicates removed.
    """

    # Based on answer from https://stackoverflow.com/questions/27751072/removing-duplicates-from-a-list-of-numpy-arrays

    if equiv is None:
        def _equiv(objA, objB):
            return np.allclose(objA, objB, atol=tol)
        equiv = _equiv
    
    uniques = []
    for obj in objects:
        if not any(equiv(obj, unique_obj) for unique_obj in uniques):
            uniques.append(obj)
    
    return uniques

def swap(string, nameA, nameB):
    """ Swap all occurances of nameA with nameB
    and vice-versa in the string.

    Parameters
    ----------
    string : str
        The string to modify.
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
        >>> qosy.swap('A Up B Dn', 'Up', 'Dn') # 'A Dn B Up'
        >>> qosy.swap('X X A', 'X', 'Y') # 'Y Y A'
        >>> qosy.swap('1 2 3', '1', '3') # '3 2 1'
    """
    
    result = string.replace(nameA, '({})'.format(nameA))
    result = result.replace(nameB, nameA)
    result = result.replace('({})'.format(nameA), nameB)

    return result

def replace(string, substitutions):
    """ Perform many string replacements
    all at once.

    Parameters
    ----------
    string : str
        The string to modify.
    substitutions : dict of str to str
        The string replacements to perform.

    Returns
    -------
    str
        The modified string.

    Examples
    --------
        >>> qosy.tools.replace('ABC', {'A':'AB', 'B':'D', 'C':'AC'}) # 'ABDAC'
    """
    
    # From https://gist.github.com/carlsmith/b2e6ba538ca6f58689b4c18f46fef11c
    
    substrings = sorted(substitutions, key=len, reverse=True)
    regex      = re.compile('|'.join(map(re.escape, substrings)))

    return regex.sub(lambda match: substitutions[match.group(0)], string)

def compose_permutations(permA, permB):
    """Compose two permutations.

    Parameters
    ----------
    permA : list or ndarray of int
        First permutation.
    permB : list or ndarray of int
        Second permutation.

    Returns
    -------
    ndarray of int
        Composition of the two permutations.
    """

    pA = np.array(permA, dtype=int)
    pB = np.array(permB, dtype=int)
    
    return pA[pB]

def gram_schmidt(matrix, tol=0.0):
    """Perform Gram-Schmidt decomposition.

    Parameters
    ----------
    matrix : ndarray or scipy.sparse.csc_matrix
        A matrix whose columns we want to orthogonalize
        going from left to right.
    tol : float, optional
        Specifies the tolerance used in the algorithm for
        discarding vector which are not orthogonal. Defaults to
        zero, in which case vectors are not discarded.

    Returns
    -------
    ndarray or scipy.sparse.csc_matrix
        A unitary matrix whose columns are orthonormal vectors that
        form an orthonormal basis of the column space of the given matrix.
        The number of columns of the returned matrix can be smaller than
        the given matrix if the given matrix is noninvertible.
    """
    
    n = int(matrix.shape[0])
    k = int(matrix.shape[1])

    if n==0 or k==0:
        return matrix.copy()

    # Use different functions for
    # dot products and norms for numpy
    # arrays and scipy sparse matrices.
    if isinstance(matrix, np.ndarray):
        new_matrix = np.zeros(dtype=complex,shape=matrix.shape)
        
        def _norm(vec):
            return nla.norm(vec)
        def _vdot(vec1, vec2):
            return np.dot(np.conj(vec1), vec2)
    else:
        new_matrix = ss.lil_matrix(matrix.shape, dtype=complex)
        
        def _norm(vec):
            return np.sqrt(np.abs((vec.H).dot(vec)[0,0]))
        def _vdot(vec1, vec2):
            return (vec1.H).dot(vec2)[0,0]
        
    shift = 0
    new_matrix[:,0] = matrix[:,0] / _norm(matrix[:,0])
    
    for ind_col_unshifted in range(1,k):
        ind_col = ind_col_unshifted - shift
        new_matrix[:,ind_col] = matrix[:,ind_col_unshifted]
        for ind_vec in range(ind_col):
            new_matrix[:,ind_col] -= _vdot(new_matrix[:,ind_vec], new_matrix[:,ind_col]) * new_matrix[:,ind_vec]

        norm = _norm(new_matrix[:,ind_col])
        
        if norm > tol:
            new_matrix[:,ind_col] = new_matrix[:,ind_col] / norm
        else:
            # Found a column that is linearly dependent on the previously found ones.
            shift += 1

    if isinstance(matrix, np.ndarray):
        return new_matrix[:,0:(k-shift)]
    else:
        return new_matrix[:,0:(k-shift)].tocsc()

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
        an orthonormal basis of the vector space :math:`V\\cap W`.
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

    Examples
    --------
        >>> vecs = np.array([[1.0/np.sqrt(2), 1.0],[1.0/np.sqrt(2), 0.0]])
        >>> qosy.sparsify(vecs) # [[1,0],[0,1]]
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

    # Reorder the vectors by sparsity.
    sparsity = [np.sum(np.abs(vectors_rre[:, ind_vec]) > tol) for ind_vec in range(num_vectors)]
    inds_sort = np.argsort(sparsity)
    vectors_rre = vectors_rre[:, inds_sort]
        
    return vectors_rre

# TODO: test, document
def project_out_nullspace(matrix, tol=1e-14):
    # Idea: Repeatedly compute the null space of
    # the input (non-invertible) square matrix and
    # project out against it to make a smaller square matrix.
    # Do this until there is either no matrix or
    # a square invertible matrix.
    
    n = int(matrix.shape[0])
    m = int(matrix.shape[1])
    
    assert(m == len(inds_basisA))
    assert(n == len(inds_basisB))
    
    (left_svecs, svals, right_svecsH) = nla.svd(matrix)

    inds_zero_svals = np.where(np.abs(svals) < tol)[0]
    
    # TODO: work out details
    raise NotImplementedError('Not finished yet.')
