from .context import qosy as qy
from helper import _random_op_string
import numpy as np
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
    assert((expected_fermion_operator - fermion_operator).norm() < 1e-12)

    fermion_operator          = qy.convert(qy.opstring('B 1'), 'Fermion')
    expected_fermion_operator = qy.Operator(np.array([1.0]), [qy.opstring('1j CDag 1')])
    assert((expected_fermion_operator - fermion_operator).norm() < 1e-12)
    
    # (1) i a_1 a_2 
    #       = i c^\dagger_1 c^\dagger_2 + i c^\dagger_1 c_2 + H.c.
    majorana_string  = qy.opstring('1j A 1 A 2')
    fermion_operator = qy.convert(majorana_string, 'Fermion')

    expected_coeffs = np.array([1.0, -1.0])
    expected_fermion_strings = [qy.opstring('1j CDag 1 CDag 2'), \
                                qy.opstring('1j CDag 2 C 1')]

    expected_fermion_operator = qy.Operator(expected_coeffs, expected_fermion_strings)

    assert((expected_fermion_operator - fermion_operator).norm() < 1e-12)
    
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

    assert((expected_fermion_operator - fermion_operator).norm() < 1e-12)

    # Exclude identity
    majorana_string  = qy.opstring('D 1 D 2')
    fermion_operator = qy.convert(majorana_string, 'Fermion', include_identity=False)
    
    expected_coeffs = np.array([-2.0, -2.0, 4.0])
    expected_fermion_strings = [qy.opstring('CDag 1 C 1'), \
                                qy.opstring('CDag 2 C 2'), \
                                qy.opstring('CDag 1 CDag 2 C 2 C 1')]
    
    expected_fermion_operator = qy.Operator(expected_coeffs, expected_fermion_strings)
    
    assert((expected_fermion_operator - fermion_operator).norm() < 1e-12)

def test_conversions_are_consistent():
    # Create random Fermion strings, convert them to Majorana strings,
    # and convert them back to Fermion strings. You should always get
    # what you started with.
    num_trials = 50
    
    max_num_orbitals        = 4
    possible_orbital_labels = np.arange(max_num_orbitals)

    np.random.seed(42)
    for ind_trial in range(num_trials):
        # Create a random Fermion string, put it into an Operator.
        fermion_string1   = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Fermion')
        fermion_operator1 = qy.Operator(np.array([1.0]), [fermion_string1])

        # Convert the Fermion operator to a Majorana operator.
        majorana_operator = qy.convert(fermion_operator1, 'Majorana')
        # Convert the Majorana operator to a Fermion operator.
        fermion_operator2 = qy.convert(majorana_operator, 'Fermion')

        # Check that you get the original operator before conversion.
        assert((fermion_operator1 - fermion_operator2).norm() < 1e-12)
    
    # Create random Majorana strings, convert them to Fermion strings,
    # and convert them back to Majorana strings. You should always get
    # what you started with.
    num_trials = 50
    
    max_num_orbitals        = 4
    possible_orbital_labels = np.arange(max_num_orbitals)

    np.random.seed(42)
    for ind_trial in range(num_trials):
        # Create a random Majorana string, put it into an Operator.
        majorana_string1   = _random_op_string(max_num_orbitals, possible_orbital_labels, 'Majorana')
        majorana_operator1 = qy.Operator(np.array([1.0]), [majorana_string1])

        # Convert the Majorana operator to a Fermion operator.
        fermion_operator   = qy.convert(majorana_operator1, 'Fermion')
        # Convert the Fermion operator to a Majorana operator.
        majorana_operator2 = qy.convert(fermion_operator, 'Majorana')

        # Check that you get the original operator before conversion.
        assert((majorana_operator1 - majorana_operator2).norm() < 1e-12)

def test_conversion_matrix():
    # Use complete cluster bases of 4^n-1 Majorana
    # and Fermion strings.
    num_orbitals   = 4
    orbital_labels = np.arange(num_orbitals)
    basisA = qy.cluster_basis(2, [1,2], 'Majorana')
    basisB = qy.cluster_basis(2, [1,2], 'Fermion')

    print(basisA)
    print(basisB)
    
    B = qy.conversion_matrix(basisA, basisB)

    # Check that the conversion matrix is invertible.
    Binv = ssla.inv(B)
    assert(ssla.norm(B*Binv - ss.eye(len(basisA), dtype=complex, format='csc')) < 1e-12)
    assert(ssla.norm(Binv*B - ss.eye(len(basisA), dtype=complex, format='csc')) < 1e-12)

    # Check that running the conversion in the other
    # direction reproduces the inverse matrix.
    Binv2 = qy.conversion_matrix(basisB, basisA)
    assert(ssla.norm(Binv - Binv2) < 1e-12)
