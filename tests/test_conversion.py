from .context import qosy as qy
from helper import _random_op_string
import numpy as np

def test_fermion_to_majorana_conversion():
    # Test some examples by hand:
    # c_1^\dagger c_2^\dagger c_2 c_1    =  
    # c_1^\dagger c_2^\dagger c_3 + H.c. = 1/4 * (-i a_1 b_2 a_3 - i b_1 a_2 a_3 - i a_1 a_2 b_3 - i b_1 b_2 b_3) 

    fermion_string    = qy.opstring('CDag 1 CDag 2 C 3')
    majorana_operator = qy.convert(fermion_string, 'Majorana')

    expected_coeffs = np.array([-0.25, -0.25, 0.25, -0.25])
    expected_majorana_strings = [qy.opstring('1j A 1 B 2 A 3'), \
                                 qy.opstring('1j B 1 A 2 A 3'), \
                                 qy.opstring('1j A 1 A 2 B 3'), \
                                 qy.opstring('1j B 1 B 2 B 3')]

    expected_majorana_operator = qy.Operator(expected_coeffs, expected_majorana_strings)

    print(expected_majorana_operator)
    print(majorana_operator)

    assert((expected_majorana_operator - majorana_operator).norm() < 1e-12)

"""
# TODO
def test_majorana_to_fermion_convesion():
    # Test some examples by hand
    
    pass

# TODO
def test_conversions_are_consistent():

    num_trials = 50
    
    max_num_orbitals        = 5
    possible_orbital_labels = np.arange(max_num_orbitals)

    np.random.seed(42)
    for ind_trial in range(num_trials):
        majorana_string = random_op_string(max_num_orbitals, possible_orbital_labels, 'Majorana')
        
        (coeffs, fermion_strings) = qy.convert(majorana_string, 'Fermion')
        
        majorana_string2
"""
