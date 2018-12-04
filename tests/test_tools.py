from .context import qosy as qy

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

def test_sort_sign():
    arr                 = np.array([-1,3,1])
    expected_sorted_arr = np.sort(arr)
    expected_sign       = -1

    (sorted_arr, sign) = qy.tools.sort_sign(arr)

    assert(np.allclose(sorted_arr, expected_sorted_arr))
    assert(sign == expected_sign)
    
    arr                 = [-1, 3, 5., 1]
    expected_sorted_arr = np.sort(arr)
    expected_sign       = 1

    (sorted_arr, sign) = qy.tools.sort_sign(arr)

    assert(np.allclose(np.array(sorted_arr), expected_sorted_arr))
    assert(sign == expected_sign)

def test_compare():
    assert(qy.tools.compare((0,1), (0,)) > 0)
    assert(qy.tools.compare((0,1), (0,1)) == 0)
    assert(qy.tools.compare((0,1), (1,0)) < 0)
    
def test_gram_schmidt():
    # Do a simple check by hand.
    matrix = np.array([[1., 1.,  2.],\
                       [0., 1., -2.]])

    vecs = qy.tools.gram_schmidt(matrix)
    expected_vecs = np.array([[1., 0.],\
                              [0., 1.]])

    assert(np.allclose(vecs, expected_vecs))
    
    matrix = np.array([[1., 1.,  2.],\
                       [1., 0., -2.]])

    vecs = qy.tools.gram_schmidt(matrix)
    expected_vecs = 1./np.sqrt(2.)*np.array([[1.,  1.],\
                                             [1., -1.]])

    assert(np.allclose(vecs, expected_vecs))
    
    n = 10
    m = 5

    # For random matrices, check that the
    # orthogonalized vectors still span
    # the same space.
    num_trials = 50
    np.random.seed(42)
    for ind_trial in range(num_trials):
        random_matrix = np.random.rand(n,m)
        vecs          = qy.tools.gram_schmidt(random_matrix)
        
        np.allclose(np.dot(np.conj(vecs).T, vecs), np.eye(m))

        # Check that every column vector of random_matrix
        # has non-zero overlap with the vectors of vecs.
        overlaps = np.dot(np.conj(random_matrix).T, vecs)
        all_rows_non_zero = True
        for ind_row in range(m):
            if np.allclose(overlaps[ind_row,:], 0.0):
                all_rows_non_zero = False
                break
        
        assert(all_rows_non_zero)
    
def test_intersection():
    # Some simple checks by hand.

    vecs1 = np.array([[1., 0.],\
                      [0., 1.]])
    vecs2 = np.array([[1.],\
                      [1.]])

    vecs12          = qy.tools.intersection(vecs1, vecs2)
    expected_vecs12 = np.copy(vecs2)/np.sqrt(2.0)

    assert(np.allclose(np.abs(vecs12), expected_vecs12))

    vecs1 = np.array([[1., 0.],\
                      [0., 1.],\
                      [0., 0.]])
    vecs2 = np.array([[1./np.sqrt(2), 0.],\
                      [1./np.sqrt(2), 0.],\
                      [0.,            1.]])

    vecs12          = qy.tools.intersection(vecs1, vecs2)
    expected_vecs12 = np.array([[1./np.sqrt(2)],\
                                [1./np.sqrt(2)],\
                                [0.]])

    assert(np.allclose(np.abs(vecs12), expected_vecs12))
