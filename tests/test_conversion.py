from .context import qosy as qy
from .helper import _random_op_string
import numpy as np
import numpy.linalg as nla
import scipy.sparse as ss
import scipy.sparse.linalg as ssla

def test_fermion_to_majorana_conversion():
    # Test some examples by hand.

    # Check identities: a_j = c_j + c^\dagger_j, b_j = i c^\dagger_j - i c_j
    majorana_operator          = qy.convert(qy.opstring('CDag 1'), 'Majorana')
    expected_majorana_operator = qy.Operator(np.array([1.0]), [qy.opstring('A 1')])
    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)
    
    majorana_operator          = qy.convert(qy.opstring('1j CDag 1'), 'Majorana')
    expected_majorana_operator = qy.Operator(np.array([1.0]), [qy.opstring('B 1')])
    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)
    
    # (1) c_1^\dagger c_1
    #     = 1/2*(I - d_1)
    
    # Including the identity
    fermion_string    = qy.opstring('CDag 1 C 1')
    majorana_operator = qy.convert(fermion_string, 'Majorana', include_identity=True)

    expected_coeffs = np.array([0.5, -0.5])
    expected_majorana_strings = [qy.opstring('I', 'Majorana'), \
                                 qy.opstring('D 1')]

    expected_majorana_operator = qy.Operator(expected_coeffs, expected_majorana_strings)

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)

    # Excluding the identity
    fermion_string    = qy.opstring('CDag 1 C 1')
    majorana_operator = qy.convert(fermion_string, 'Majorana', include_identity=False)

    expected_coeffs = np.array([-0.5])
    expected_majorana_strings = [qy.opstring('D 1')]

    expected_majorana_operator = qy.Operator(expected_coeffs, expected_majorana_strings)

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)
    
    # (2) c_1^\dagger c_2^\dagger c_3 + H.c.
    #     = 1/4 * (-i a_1 b_2 a_3 - i b_1 a_2 a_3 - i a_1 a_2 b_3 - i b_1 b_2 b_3) 

    fermion_string    = qy.opstring('CDag 1 CDag 2 C 3')
    majorana_operator = qy.convert(fermion_string, 'Majorana')

    expected_coeffs = np.array([-0.25, -0.25, 0.25, -0.25])
    expected_majorana_strings = [qy.opstring('1j A 1 B 2 A 3'), \
                                 qy.opstring('1j B 1 A 2 A 3'), \
                                 qy.opstring('1j A 1 A 2 B 3'), \
                                 qy.opstring('1j B 1 B 2 B 3')]

    expected_majorana_operator = qy.Operator(expected_coeffs, expected_majorana_strings)

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)

    # (3) i c_1^\dagger c_2^\dagger c_1 + H.c.
    #       = 1/2 * (-b_2 + d_1 a_2)

    fermion_string    = qy.opstring('1j CDag 1 CDag 2 C 1')
    majorana_operator = qy.convert(fermion_string, 'Majorana')

    expected_coeffs = np.array([-0.5, 0.5])
    expected_majorana_strings = [qy.opstring('B 2'), \
                                 qy.opstring('D 1 B 2')]

    expected_majorana_operator = qy.Operator(expected_coeffs, expected_majorana_strings)

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)
    

def test_majorana_to_fermion_conversion():
    # Test some examples by hand

    # Check identities: a_j = c_j + c^\dagger_j, b_j = i c^\dagger_j - i c_j
    fermion_operator          = qy.convert(qy.opstring('A 1'), 'Fermion')
    expected_fermion_operator = qy.Operator(np.array([1.0]), [qy.opstring('CDag 1')])
    assert(np.allclose((expected_fermion_operator - fermion_operator).coeffs, 0.0))

    fermion_operator          = qy.convert(qy.opstring('B 1'), 'Fermion')
    expected_fermion_operator = qy.Operator(np.array([1.0]), [qy.opstring('1j CDag 1')])
    assert(np.allclose((expected_fermion_operator - fermion_operator).coeffs, 0.0))
    
    # (1) i a_1 a_2 
    #       = i c^\dagger_1 c^\dagger_2 + i c^\dagger_1 c_2 + H.c.
    majorana_string  = qy.opstring('1j A 1 A 2')
    fermion_operator = qy.convert(majorana_string, 'Fermion')

    expected_coeffs = np.array([1.0, -1.0])
    expected_fermion_strings = [qy.opstring('1j CDag 1 CDag 2'), \
                                qy.opstring('1j CDag 2 C 1')]

    expected_fermion_operator = qy.Operator(expected_coeffs, expected_fermion_strings)

    assert(np.allclose((expected_fermion_operator - fermion_operator).coeffs, 0.0))
    
    # (2) d_1 d_2 
    #       = -I - 2 c^\dagger_1 c_2 - 2 c^\dagger_2 c_2 + 4 c^\dagger_1 c^\dagger_1 c^\dagger_2 c_2 c_1

    # Include identity
    majorana_string  = qy.opstring('D 1 D 2')
    fermion_operator = qy.convert(majorana_string, 'Fermion', include_identity=True)
    
    print(fermion_operator)
    
    expected_coeffs = np.array([1.0, -2.0, -2.0, 4.0])
    expected_fermion_strings = [qy.opstring('I', 'Fermion'), \
                                qy.opstring('CDag 1 C 1'), \
                                qy.opstring('CDag 2 C 2'), \
                                qy.opstring('CDag 1 CDag 2 C 2 C 1')]

    expected_fermion_operator = qy.Operator(expected_coeffs, expected_fermion_strings)

    assert(np.allclose((expected_fermion_operator - fermion_operator).coeffs, 0.0))

    # Exclude identity
    majorana_string  = qy.opstring('D 1 D 2')
    fermion_operator = qy.convert(majorana_string, 'Fermion', include_identity=False)
    
    expected_coeffs = np.array([-2.0, -2.0, 4.0])
    expected_fermion_strings = [qy.opstring('CDag 1 C 1'), \
                                qy.opstring('CDag 2 C 2'), \
                                qy.opstring('CDag 1 CDag 2 C 2 C 1')]
    
    expected_fermion_operator = qy.Operator(expected_coeffs, expected_fermion_strings)
    
    assert(np.allclose((expected_fermion_operator - fermion_operator).coeffs, 0.0))

def test_pauli_to_majorana_conversion():
    # Test some examples by hand

    # Check identities: a_0 = X_0, b_0 = Y_0, D_0 = Z_0
    majorana_operator          = qy.convert(qy.opstring('X 0'), 'Majorana')
    expected_majorana_operator = qy.Operator([1.0], [qy.opstring('A 0')])

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-16)

    majorana_operator          = qy.convert(qy.opstring('Y 0'), 'Majorana')
    expected_majorana_operator = qy.Operator([1.0], [qy.opstring('B 0')])
    
    assert((expected_majorana_operator - majorana_operator).norm() < 1e-16)

    majorana_operator          = qy.convert(qy.opstring('Z 0'), 'Majorana')
    expected_majorana_operator = qy.Operator([1.0], [qy.opstring('D 0')])
    
    assert((expected_majorana_operator - majorana_operator).norm() < 1e-16)

    # Check a two-site example: i a_0 b_1 = i (X_0)(Z_0 Y_1) = Y_0 Y_1
    majorana_operator          = qy.convert(qy.opstring('Y 0 Y 1'), 'Majorana')
    expected_majorana_operator = qy.Operator([1.0], [qy.opstring('A 0 B 1')])
    
    assert((expected_majorana_operator - majorana_operator).norm() < 1e-16)

def test_majorana_to_pauli_conversion():
    # Test some examples by hand

    # Check identities: a_0 = X_0, b_0 = Y_0, D_0 = Z_0
    pauli_operator          = qy.convert(qy.opstring('A 0'), 'Pauli')
    expected_pauli_operator = qy.Operator([1.0], [qy.opstring('X 0')])

    assert((expected_pauli_operator - pauli_operator).norm() < 1e-16)

    pauli_operator          = qy.convert(qy.opstring('B 0'), 'Pauli')
    expected_pauli_operator = qy.Operator([1.0], [qy.opstring('Y 0')])
    
    assert((expected_pauli_operator - pauli_operator).norm() < 1e-16)

    pauli_operator          = qy.convert(qy.opstring('D 0'), 'Pauli')
    expected_pauli_operator = qy.Operator([1.0], [qy.opstring('Z 0')])
    
    assert((expected_pauli_operator - pauli_operator).norm() < 1e-16)

    # Check a two-site example: i a_0 b_1 = i (X_0)(Z_0 Y_1) = Y_0 Y_1
    pauli_operator          = qy.convert(qy.opstring('A 0 B 1'), 'Pauli')
    expected_pauli_operator = qy.Operator([1.0], [qy.opstring('Y 0 Y 1')])
    
    assert((expected_pauli_operator - pauli_operator).norm() < 1e-16)
    
    
def test_conversions_are_consistent():
    # Create random operator strings, convert them to other operator strings,
    # and convert them back. You should always get what you started with.
    num_trials = 10
    
    max_num_orbitals        = 4
    possible_orbital_labels = np.arange(max_num_orbitals)

    op_types = ['Pauli', 'Majorana', 'Fermion']
    
    np.random.seed(42)
    for ind_trial in range(num_trials):
        for op_type1 in op_types:
            # Create a random operator string, put it into an Operator.
            os1 = _random_op_string(max_num_orbitals, possible_orbital_labels, op_type1)
            op1 = qy.Operator([1.0], [os1])
            for op_type2 in op_types:
                # Convert from op_type1 to op_type2 and back.
                op2 = qy.convert(op1, op_type2)
                op3 = qy.convert(op2, op_type1)

                # Check that you get the original operator before conversion.
                assert((op1 - op3).norm() < 1e-16)

def test_conversion_matrix():
    # Use complete cluster bases of 4^n-1 Majorana
    # and Fermion strings.
    num_orbitals   = 4
    orbital_labels = np.arange(num_orbitals)
    basisA = qy.cluster_basis(2, [1,2], 'Majorana')
    basisB = qy.cluster_basis(2, [1,2], 'Fermion')
    
    B = qy.conversion_matrix(basisA, basisB).toarray()

    # Check that the conversion matrix is invertible.
    Binv = nla.inv(B)
    assert(nla.norm(np.dot(B,Binv) - np.eye(len(basisA), dtype=complex)) < 1e-16)
    assert(nla.norm(np.dot(Binv,B) - np.eye(len(basisA), dtype=complex)) < 1e-16)

    # Check that running the conversion in the other
    # direction reproduces the inverse matrix.
    Binv2 = qy.conversion_matrix(basisB, basisA).toarray()
    assert(nla.norm(Binv - Binv2) < 1e-16)
