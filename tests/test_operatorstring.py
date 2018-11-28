from .context import qosy as qy
import numpy as np

def test_operator_string():
    os1 = qy.OperatorString(['X', 'Y', 'Z'], [1, 2, 4], 'Pauli')
    expected_name1 = '1.0 X 1 Y 2 Z 4 '

    assert(str(os1) == os1.name and os1.name == expected_name1)

    os2 = qy.OperatorString(['A', 'B', 'D', 'D'], [1, 3, 5, 6], 'Majorana')
    expected_name2 = '1j A 1 B 3 D 5 D 6 '

    assert(str(os2) == os2.name and os2.name == expected_name2)
    
    os3 = qy.OperatorString(['CDag', 'CDag', 'C'], [1, 2, 3], 'Fermion')
    expected_name3 = '1.0 CDag 1 CDag 2 C 3 '

    assert(str(os3) == os3.name and os3.name == expected_name3)
    
    os4 = qy.OperatorString(['CDag', 'CDag', 'C'], [1, 2, 3], 'Fermion')

    assert(os3 == os4)
    assert(hash(os3) == hash(os4))

    os5 = qy.OperatorString(['X', 'Y', 'Z'], [1,2,4], 'Pauli')

    assert(os1 == os5)

    assert(os1 != os2)

    os5a = qy.opstring('X 1 Y 2 Z 4')
    os5b = qy.opstring('1.0 X 1 Y 2 Z 4')
    os5c = qy.opstring('1 X 1 Y 2 Z 4')

    assert(os5 == os5a)
    assert(os5a == os5b)
    assert(os5b == os5c)

    identityA = qy.OperatorString([], [], 'Pauli')
    identityB = qy.opstring('1', 'Pauli')
    identityC = qy.opstring('I', 'Pauli')

    assert(identityA == identityB)
    assert(identityB == identityC)
