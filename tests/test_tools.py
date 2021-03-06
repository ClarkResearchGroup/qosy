from .context import qosy as qy

import numpy as np
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

def test_sign():
    perm = np.array([0,1,2,3],dtype=int)
    expected_sign = 1
    sign = qy.tools.sign(perm)

    assert(sign == expected_sign)

    perm = np.array([1,0,2,3],dtype=int)
    expected_sign = -1
    sign = qy.tools.sign(perm)

    assert(sign == expected_sign)

    perm = np.array([1,0,3,2],dtype=int)
    expected_sign = 1
    sign = qy.tools.sign(perm)

    assert(sign == expected_sign)

    perm = np.array([2,4,1,3,0,5],dtype=int)
    expected_sign = -1
    sign = qy.tools.sign(perm)

    assert(sign == expected_sign)
    
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

def test_sort_sign_mergesort():
    np.random.seed(42)
    num_trials = 100
    arr_length = 100

    for ind_trial in range(num_trials):
        arr = 2.0*np.random.rand(arr_length) - 1.0

        (sorted_arr1, sign1) = qy.tools.sort_sign(arr, method='insertionsort')
        (sorted_arr2, sign2) = qy.tools.sort_sign(arr, method='mergesort')
        
        assert(np.allclose(sorted_arr1, sorted_arr2))
        assert(np.isclose(sign1, sign2))
        
def test_compare():
    assert(qy.tools.compare((0,1), (0,)) > 0)
    assert(qy.tools.compare((0,1), (0,1)) == 0)
    assert(qy.tools.compare((0,1), (1,0)) < 0)

def test_swap():
    assert(qy.tools.swap('A B C', 'A', 'B') == 'B A C')
    assert(qy.tools.swap('1 2 3', '1', '4') == '4 2 3')
    assert(qy.tools.swap('Up Up Dn', 'Up', 'Dn') == 'Dn Dn Up')
    assert(qy.tools.swap('Up Up Dn', 'Dn', 'Up') == 'Dn Dn Up')
    assert(qy.tools.swap('1 2 3', 'X', 'Y') == '1 2 3')

def test_replace():
    assert(qy.tools.replace('ABC', {'A':'AB', 'B':'D', 'C':'AC'}) == 'ABDAC')
    assert(qy.tools.replace(' 0 100 10 1 110', {' 0':'_{0}', ' 1' : '_{1}', ' 10' : '_{10}', ' 100' : '_{100}', ' 110' : '_{110}'}) == '_{0}_{100}_{10}_{1}_{110}')
    
def test_maximal_cliques():
    # Toy graph on https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    adjacency_lists = [[1,4], [0,2,4], [1,3], [2,4,5], [0,1,3], [3]]
    
    maximal_cliques = qy.tools.maximal_cliques(adjacency_lists)

    maximal_cliques_set = set([tuple(clique) for clique in maximal_cliques])
    
    expected_maximal_cliques_set = set([(0,1,4), (1,2), (2,3), (3,4), (3,5)])
    
    assert(maximal_cliques_set == expected_maximal_cliques_set)

def test_connected_components():
    # Graph with two connected components, [0,1,2,3] and [4,5]
    adjacency_lists = [[1,2], [0,2,3], [0,1], [1], [5], [4]]
    
    c_components = qy.tools.connected_components(adjacency_lists, mode='BFS')
    
    expected_c_comps = set([(0,1,2,3), (4,5)])
    assert(len(c_components) == 2)
    for c_comp in c_components:
        c_comp_tuple = tuple(np.sort(c_comp))
        assert(c_comp_tuple in expected_c_comps)
        
    c_components = qy.tools.connected_components(adjacency_lists, mode='DFS')
    
    expected_c_comps = set([(0,1,2,3), (4,5)])
    assert(len(c_components) == 2)
    for c_comp in c_components:
        c_comp_tuple = tuple(np.sort(c_comp))
        assert(c_comp_tuple in expected_c_comps)
        
def test_gram_schmidt():
    # Do a simple check by hand.
    matrix = np.array([[1., 1.,  2.],\
                       [0., 1., -2.]])

    vecs = qy.tools.gram_schmidt(matrix, tol=1e-12)
    expected_vecs = np.array([[1., 0.],\
                              [0., 1.]])

    assert(np.allclose(vecs, expected_vecs))
    
    matrix = np.array([[1., 1.,  2.],\
                       [1., 0., -2.]])

    vecs = qy.tools.gram_schmidt(matrix, tol=1e-12)
    expected_vecs = 1./np.sqrt(2.)*np.array([[1.,  1.],\
                                             [1., -1.]])

    assert(np.allclose(vecs, expected_vecs))

    matrix = np.array([[1., 1.,  2.],\
                       [1., 0., -2.]])

    vecs = qy.tools.gram_schmidt(ss.csc_matrix(matrix), tol=1e-12)
    expected_vecs = 1./np.sqrt(2.)*np.array([[1.,  1.],\
                                             [1., -1.]])

    assert(np.allclose(vecs.toarray(), expected_vecs))
    
    # Do a simple complex number check by hand.
    matrix = np.array([[1.,  1.],\
                       [1.j, 1.]])

    vecs = qy.tools.gram_schmidt(matrix, tol=1e-12)
    expected_vecs = 1./np.sqrt(2.)*np.array([[1.,  1.],\
                                             [1j, -1j]])
    
    # First vector agrees
    assert(np.allclose(vecs[:,0], expected_vecs[:,0]))
    # Second vector agrees up to a phase.
    overlap = np.vdot(vecs[:,1], expected_vecs[:,1])
    assert(np.isclose(np.abs(overlap), 1.0))
    
    n = 10
    m = 5

    # For random real matrices, check that the
    # orthogonalized vectors still span
    # the same space.
    num_trials = 50
    np.random.seed(42)
    for ind_trial in range(num_trials):
        random_matrix = 2.0*np.random.rand(n,m)-1.0
        vecs          = qy.tools.gram_schmidt(random_matrix)

        # Check that the vectors are orthonormal.
        assert(np.allclose(np.dot(np.conj(vecs).T, vecs), np.eye(m)))

        # Check that every column vector of random_matrix
        # has non-zero overlap with the vectors of vecs.
        overlaps = np.dot(np.conj(random_matrix).T, vecs)
        all_rows_non_zero = True
        for ind_row in range(m):
            if np.allclose(overlaps[ind_row,:], 0.0):
                all_rows_non_zero = False
                break
        
        assert(all_rows_non_zero)

    # For random complex matrices, check that the
    # orthogonalized vectors still span
    # the same space.
    num_trials = 50
    np.random.seed(42)
    for ind_trial in range(num_trials):
        random_matrix = (2.0*np.random.rand(n,m)-1.0) + 1j*(2.0*np.random.rand(n,m)-1.0)
        vecs          = qy.tools.gram_schmidt(random_matrix)

        # Check that the vectors are orthonormal.
        assert(np.allclose(np.dot(np.conj(vecs).T, vecs), np.eye(m, dtype=complex)))
        
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
