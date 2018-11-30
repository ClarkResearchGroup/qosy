#!/usr/bin/env python
import numpy as np

def sort_sign(vector, tol=1e-10):
    """Stable sorts the vector and returns
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

def argsort(my_list):
    """Finds the permutation of the indices
    that sorts the list.

    Parameter
    ---------
    my_list : list
        The list to sort.

    Returns
    -------
    list of int
        The permutation that sorts the list.
    """

    return sorted(range(len(my_list)), key=my_list.__getitem__)

def compare(labelsI, labelsJ):
    """Lexicographically compares two sets of labels, (i_1,\ldots,i_m) 
    and (j_1,\ldots,j_l). 

    If m < l, then (i_1,\ldots,i_m) < (j_1,\ldots,j_l). 
    If m = l, then you compare i_1 and j_1. If those are equal, you 
    compare i_2 and j_2, and so on.

    Parameters
    ----------
    labelsI, labelsJ : lists, tuples, or ndarray of ints
        The sets of labels to compare.

    Returns
    -------
    int
        Returns 1 if labelsI > labelsJ, 
        -1 if labelsI < labelsJ, and 0 otherwise. 
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
