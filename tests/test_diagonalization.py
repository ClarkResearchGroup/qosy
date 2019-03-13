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

def test_diagonalize():
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
