#!/usr/bin/env python
import numpy as np

def sort_sign(vector, tol=1e-10):
    """Stable sorts the vector and returns
    the sign of the permutation needed to sort it.

    Parameters
    ----------
    vector : array_like of numbers
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
