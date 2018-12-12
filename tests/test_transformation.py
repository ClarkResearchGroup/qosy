from .context import qosy as qy
import numpy as np

def test_time_reversal():
    # Test some examples by hand.
    
    op_string = qy.opstring('A 1')
    T         = qy.time_reversal()
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)
    
    op_string = qy.opstring('B 1')
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)
    
    op_string = qy.opstring('A 1 B 2 D 3 A 4 A 5')
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)
    
    op_string = qy.opstring('1j A 1 A 2')
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('X 1 Y 2')
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('X 1 Y 2 Z 3')
    operator  = T.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

def test_particle_hole():
    # Test some examples by hand.
    
    op_string = qy.opstring('A 1')
    C         = qy.particle_hole()
    operator  = C.apply(op_string)

    expected_operator = qy.Operator([1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('B 1')
    operator  = C.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('D 1')
    operator  = C.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('1j A 1 D 2 A 3')
    operator  = C.apply(op_string)

    expected_operator = qy.Operator([-1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('1j A 1 D 2 B 3')
    operator  = C.apply(op_string)

    expected_operator = qy.Operator([1.0], [op_string])

    assert((operator - expected_operator).norm() < 1e-12)

def test_permutation():
    permutation = np.array([1,2,3,4,0], dtype=int)
    P           = qy.label_permutation(permutation)
    
    # Test some examples by hand.

    op_string = qy.opstring('A 1')
    operator  = P.apply(op_string)

    expected_operator = qy.Operator([1.0], [qy.opstring('A 2')])

    assert((operator - expected_operator).norm() < 1e-12)
    
    op_string = qy.opstring('A 4')
    operator  = P.apply(op_string)

    expected_operator = qy.Operator([1.0], [qy.opstring('A 0')])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('A 1 B 4')
    operator  = P.apply(op_string)

    expected_operator = qy.Operator([-1.0], [qy.opstring('B 0 A 2')])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('X 1')
    operator  = P.apply(op_string)

    expected_operator = qy.Operator([1.0], [qy.opstring('X 2')])

    assert((operator - expected_operator).norm() < 1e-12)

    op_string = qy.opstring('Y 1 Z 4')
    operator  = P.apply(op_string)

    expected_operator = qy.Operator([1.0], [qy.opstring('Z 0 Y 2')])

    assert((operator - expected_operator).norm() < 1e-12)
    
def test_transformation_product():
    
    permutation = np.array([1,2,3,4,0], dtype=int)

    # Permutation
    P = qy.label_permutation(permutation)
    # Charge-conjugation/particle-hole
    C = qy.particle_hole()
    # Time-reversal
    T = qy.time_reversal()

    # Chiral S = T*C
    S = T * C

    # h_a = i a_1 b_2 d_3 b_4
    op_string = qy.opstring('1j A 1 B 2 D 3 B 4')
    # C h_a C^{-1}            = i (a_1)(-b_2)(-d_3)(-b_4)
    #                         = - i a_1 b_2 d_3 b_4
    # T (C h_a C^{-1}) T^{-1} = -T (i a_1 b_2 d_3 b_4) T^{-1}
    #                         = -(-i)(a_1)(-b_2)(d_3)(-b_4)
    #                         = i a_1 b_2 d_3 b_4
    operator  = S.apply(op_string)

    expected_operator = qy.Operator([1.0], [qy.opstring('1j A 1 B 2 D 3 B 4')])

    assert((operator - expected_operator).norm() < 1e-12)

    TP = T * P
    # P h_a P^{-1}            = i a_2 b_3 d_4 b_0 = i b_0 a_2 b_3 d_4
    # T (P h_a P^{-1}) T^{-1} = (-i) (-b_0) (a_2) (-b_3) (d_4)
    #                         = -i b_0 a_2 b_3 d_4
    operator = TP.apply(op_string)
    
    expected_operator = qy.Operator([-1.0], [qy.opstring('1j B 0 A 2 B 3 D 4')])

    assert((operator - expected_operator).norm() < 1e-12)
    
    translation_perm = permutation
    reflection_perm  = np.array([4,3,2,1,0], dtype=int)

    # Translates 5-orbital chain to the right.
    P = qy.label_permutation(translation_perm)
    # Reflects about center (3rd orbital) of chain.
    R = qy.label_permutation(reflection_perm)

    RP = R * P
    # P h_a P^{-1}            = i a_2 b_3 d_4 b_0 = i b_0 a_2 b_3 d_4
    # R (P h_a P^{-1}) R^{-1} = i b_4 a_2 b_1 d_0 = -i d_0 b_1 a_2 b_4
    operator          = RP.apply(op_string)
    expected_operator = qy.Operator([-1.0], [qy.opstring('1j D 0 B 1 A 2 B 4')])

    assert((operator - expected_operator).norm() < 1e-12)
    
    PR = P * R
    # R h_a R^{-1}            = i a_3 b_2 d_1 b_0  = -i b_0 d_1 b_2 a_3
    # P (R h_a R^{-1}) P^{-1} = -i b_1 d_2 b_3 a_4
    #                         != R (P h_a P^{-1}) R^{-1}
    operator          = PR.apply(op_string)
    expected_operator = qy.Operator([-1.0], [qy.opstring('1j B 1 D 2 B 3 A 4')])

    assert((operator - expected_operator).norm() < 1e-12)
    
def test_symmetry_matrix_simple():
    # Test the symmetry matrix using a Basis of OperatorStrings.
    
    op_strings = [qy.opstring('X 0 Y 1'), qy.opstring('X 0'), qy.opstring('X 1'), qy.opstring('X 0 X 1')]
    basis      = qy.Basis(op_strings)

    # Swap labels 0 and 1
    permutation = np.array([1,0], dtype=int)
    P = qy.label_permutation(permutation)

    sym_matrix          = qy.symmetry_matrix(basis, P).toarray()
    expected_sym_matrix = np.array([[0, 0, 0, 0], \
                                    [0, 0, 1, 0], \
                                    [0, 1, 0, 0], \
                                    [0, 0, 0, 1]],dtype=complex)

    assert(np.linalg.norm(sym_matrix - expected_sym_matrix) < 1e-12)

def test_symmetry_matrix_operators_simple():
    # Test the symmetry matrix using a list of Operators as a "basis."

    operator1 = qy.Operator([1,1],  [qy.opstring('X 0'), qy.opstring('X 1')])
    operator2 = qy.Operator([1],    [qy.opstring('X 0 X 1')])
    operator3 = qy.Operator([1,-1], [qy.opstring('Y 0'), qy.opstring('Y 1')])
    operator4 = qy.Operator([1],    [qy.opstring('Z 0')])
    operator5 = qy.Operator([1],    [qy.opstring('Z 1')])
    
    operators = [operator1, operator2, operator3, operator4, operator5]

    # Swap labels 0 and 1
    permutation = np.array([1,0], dtype=int)
    P = qy.label_permutation(permutation)

    sym_matrix          = qy.symmetry_matrix(operators, P).toarray()
    expected_sym_matrix = np.array([[1, 0,  0, 0, 0], \
                                    [0, 1,  0, 0, 0], \
                                    [0, 0, -1, 0, 0], \
                                    [0, 0,  0, 0, 1], \
                                    [0, 0,  0, 1, 0]],dtype=complex)

    print(sym_matrix)
    print(expected_sym_matrix)
    assert(np.linalg.norm(sym_matrix - expected_sym_matrix) < 1e-12)
