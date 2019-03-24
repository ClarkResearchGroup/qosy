from .context import qosy as qy
from .helper import _random_op_string

import numpy as np
import numpy.linalg as nla
import scipy.special as sps
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

def test_to_matrix():
    I = np.array([[1., 0.], [0., 1.]], dtype=complex)
    X = np.array([[0., 1.], [1., 0.]], dtype=complex)
    Y = np.array([[0., -1j], [1j, 0.]], dtype=complex)
    Z = np.array([[1., 0.], [0., -1.]], dtype=complex)
    
    # Check one-site operators, X, Y, Z. 
    op_string = qy.opstring('X 0')
    matrix_os = qy.to_matrix(op_string, 1).toarray()

    expected_matrix_os = X
    assert(np.allclose(matrix_os, expected_matrix_os))
    
    op_string = qy.opstring('Y 0')
    matrix_os = qy.to_matrix(op_string, 1).toarray()

    expected_matrix_os = Y
    assert(np.allclose(matrix_os, expected_matrix_os))

    op_string = qy.opstring('Z 0')
    matrix_os = qy.to_matrix(op_string, 1).toarray()

    expected_matrix_os = Z

    assert(np.allclose(matrix_os, expected_matrix_os))

    # Random two, three, and four-site checks
    num_trials = 10
    
    max_num_orbitals        = 2
    possible_orbital_labels = np.arange(max_num_orbitals)

    np.random.seed(42)
    for ind_trial in range(num_trials):
        # Create a random operator string, put it into an Operator.
        os = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Pauli')

        expected_matrix = np.ones((1,1), dtype=complex)
        for orb_label in range(max_num_orbitals):
            if orb_label in os.orbital_labels:
                ind_orb = list(os.orbital_labels).index(orb_label)
                orb_op  = os.orbital_operators[ind_orb]
                if orb_op == 'X':
                    expected_matrix = np.kron(expected_matrix, X)
                elif orb_op == 'Y':
                    expected_matrix = np.kron(expected_matrix, Y)
                elif orb_op == 'Z':
                    expected_matrix = np.kron(expected_matrix, Z)
                else:
                    raise ValueError('Invalid orbital operator: {}'.format(orb_op))
            else:
                expected_matrix = np.kron(expected_matrix, I)

        matrix = qy.to_matrix(os, max_num_orbitals).toarray()

        assert(np.allclose(matrix, expected_matrix))
        
    # Simple by hand test for Operators that are sums of OperatorStrings.
    op_string1 = qy.opstring('X 0 Y 2 Z 3')
    matrix_os1 = qy.to_matrix(op_string1, 4).toarray()
    op_string2 = qy.opstring('X 2')
    matrix_os2 = qy.to_matrix(op_string2, 4).toarray()
    
    expected_matrix_os1 = np.kron(X, np.kron(I, np.kron(Y, Z)))
    expected_matrix_os2 = np.kron(I, np.kron(I, np.kron(X, I)))
    
    op = qy.Operator([0.2, -0.3], [op_string1, op_string2])
    matrix_op = qy.to_matrix(op, 4).toarray()
    expected_matrix_op = 0.2*expected_matrix_os1 - 0.3*expected_matrix_os2
    
    assert(np.allclose(matrix_op, expected_matrix_op))

def test_apply_transformation():
    num_orbitals = 2
    labels = np.arange(num_orbitals)
    
    permutation    = np.array([1,0], dtype=int)
    transformation = qy.label_permutation(permutation)

    # [00, 01, 10, 11] -> [00, 10, 01, 11]
    vector     = np.array([0.1, 0.2j, -0.3, -0.4j], dtype=complex)
    new_vector = qy.apply_transformation(transformation, vector, num_orbitals)

    expected_new_vector = np.array([0.1, -0.3, 0.2j, -0.4j], dtype=complex)
    assert(np.allclose(new_vector, expected_new_vector))
    
    num_orbitals = 3
    labels = np.arange(num_orbitals)
    
    permutation    = np.array([1,2,0], dtype=int)
    transformation = qy.label_permutation(permutation)

    # [000, 001, 010, 011, 100, 101, 110, 111] = [0,1,2,3,4,5,6,7]
    # -> [000, 010, 100, 110, 001, 011, 101, 111] = [0,2,4,6,1,3,5,7]
    vector     = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=complex)
    new_vector = qy.apply_transformation(transformation, vector, num_orbitals)

    expected_new_vector = np.array([0.1,0.3,0.5,0.7,0.2,0.4,0.6,0.8], dtype=complex)
    assert(np.allclose(new_vector, expected_new_vector))

def test_reduced_density_matrix():
    # 00,01,10,11
    # A = region (orbital labels) remaining after trace
    # B = region to trace out
    num_orbitals = 2
    A = [0]
    B = [1]
    a = 1.0/np.sqrt(2.0)
    b = 1j/np.sqrt(2.0)
    vector_AB = np.array([a, 0, 0, b], dtype=complex)
    
    rho_A = qy.reduced_density_matrix(vector_AB, A, num_orbitals)
    
    expected_rho_A = np.array([[np.abs(a)**2, 0], [0, np.abs(b)**2]], dtype=complex)

    assert(np.allclose(rho_A, expected_rho_A))

    rho_B = qy.reduced_density_matrix(vector_AB, B, num_orbitals)
    
    expected_rho_B = np.array([[np.abs(a)**2, 0], [0, np.abs(b)**2]], dtype=complex)
    
    assert(np.allclose(rho_B, expected_rho_B))

    # 000,001,010,011,100,101,110,111
    num_orbitals = 3
    A = [1,2]
    B = [0]
    a = 1.0/np.sqrt(3.0)
    b = 1j/np.sqrt(3.0)
    c = -1.0/np.sqrt(3.0)
    vector_AB = np.array([a, 0, b, 0, 0, 0, 0, c], dtype=complex)
    
    rho_A = qy.reduced_density_matrix(vector_AB, A, num_orbitals)
    
    expected_rho_A      = np.zeros((4,4), dtype=complex)
    expected_rho_A[0,0] = np.abs(a)**2
    expected_rho_A[0,2] = a*np.conj(b)
    expected_rho_A[2,0] = b*np.conj(a)
    expected_rho_A[2,2] = np.abs(b)**2
    expected_rho_A[3,3] = np.abs(c)**2
    
    assert(np.allclose(rho_A, expected_rho_A))

def test_renyi_entropy():
    num_orbitals = 2
    rho = np.zeros((2**num_orbitals,2**num_orbitals), dtype=complex)
    rho[0,0] = 1
    
    entropy = qy.renyi_entropy(rho)
    expected_entropy = 0.0

    assert(np.isclose(entropy, expected_entropy))

    rho = np.diag(1.0/(2**num_orbitals)*np.ones(2**num_orbitals,dtype=complex))
    
    entropy = qy.renyi_entropy(rho)
    expected_entropy = num_orbitals * np.log(2)

    assert(np.isclose(entropy, expected_entropy))
    
def test_diagonalize_transverse_ising():
    L = 5
    lamb = 1.0

    coeffs     = []
    op_strings = []
    for site in range(L-1):
        coeffs.append(-1.0)
        op_strings.append(qy.opstring('Z {} Z {}'.format(site, site+1)))

    for site in range(L):
        coeffs.append(-lamb)
        op_strings.append(qy.opstring('X {}'.format(site)))

    hamiltonian = qy.Operator(coeffs, op_strings)

    # Full diagonalization
    (evals1, evecs1) = qy.diagonalize(hamiltonian, L)
    # Lanczos
    (evals2, evecs2) = qy.diagonalize(hamiltonian, L, num_vecs=1)
    gs_energy1 = evals1[0] / L
    gs_energy2 = evals2[0] / L

    assert(np.isclose(gs_energy1, gs_energy2))
    
    # Exact ground state energy for finite open transverse-field Ising chain
    # at lambda = h = J = 1.
    # From: https://dmrg101-tutorial.readthedocs.io/en/latest/tfim.html
    # but they have a typo: they meant Pauli matrices but said S_i = \sigma_i/2.
    expected_gs_energy = (1.0 - 1.0/np.sin(np.pi/(2.0*(2.0*L+1.0))))/L
    
    assert(np.isclose(gs_energy1, expected_gs_energy))

def test_diagonalize_majumdar_ghosh():
    L = 6
    lamb = 1.0
    
    coeffs     = []
    op_strings = []
    for site in range(L):
        for d_site in [1,2]:
            sitep = (site + d_site) % L
            s1 = np.minimum(site, sitep)
            s2 = np.maximum(site, sitep)
            for orb_op in ['X', 'Y', 'Z']:
                if d_site == 1:
                    coeffs.append(1.0*0.25)
                else:
                    coeffs.append(0.5*0.25)
                op_strings.append(qy.opstring('{} {} {} {}'.format(orb_op, s1, orb_op, s2)))
    
    hamiltonian = qy.Operator(coeffs, op_strings)
    
    # Full diagonalization
    (evals1, evecs1) = qy.diagonalize(hamiltonian, L)
    # Lanczos
    (evals2, evecs2) = qy.diagonalize(hamiltonian, L, num_vecs=3)
    gs_energy1 = evals1[0] / L
    gs_energy2 = evals2[0] / L
    
    assert(np.isclose(gs_energy1, gs_energy2))
    
    # Should be a doubly degenerate ground state
    inds_gs1 = np.where(np.abs(evals1-evals1[0]) < 1e-14)[0]
    inds_gs2 = np.where(np.abs(evals2-evals2[0]) < 1e-14)[0]
    num_gs1 = len(inds_gs1)
    num_gs2 = len(inds_gs2)

    assert(num_gs1 == num_gs2)
    assert(num_gs1 == 2)
    
    # Exact ground state energy per site
    expected_gs_energy = -1.5*0.25
    
    assert(np.isclose(gs_energy1, expected_gs_energy))
    
