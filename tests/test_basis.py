#!/usr/bin/env python
from .context import qosy as qy
from qosy.basis import Basis, Operator
from qosy.operatorstring import OperatorString
import numpy as np

def test_cluster_basis():
    k = 2
    cluster_labels = [1,2]
    basis = qy.cluster_basis(k, cluster_labels, 'Pauli')

    expected_op_strings =  \
    [OperatorString(['X'], [1], 'Pauli'), OperatorString(['X'], [2], 'Pauli'), \
     OperatorString(['Y'], [1], 'Pauli'), OperatorString(['Y'], [2], 'Pauli'), \
     OperatorString(['Z'], [1], 'Pauli'), OperatorString(['Z'], [2], 'Pauli'), \
     OperatorString(['X','X'], [1,2], 'Pauli'), OperatorString(['X','Y'], [1,2], 'Pauli'), \
     OperatorString(['X','Z'], [1,2], 'Pauli'), OperatorString(['Y','Y'], [1,2], 'Pauli'), \
     OperatorString(['Y','Z'], [1,2], 'Pauli'), OperatorString(['Z','Z'], [1,2], 'Pauli'), \
     OperatorString(['Y','X'], [1,2], 'Pauli'), \
     OperatorString(['Z','X'], [1,2], 'Pauli'), \
     OperatorString(['Z','Y'], [1,2], 'Pauli'), \
    ]

    assert(basis.op_strings[0] == expected_op_strings[0])
    assert(set(basis.op_strings) == set(expected_op_strings))

    k = 2
    cluster_labels = [1,2]
    basis = qy.cluster_basis(k, cluster_labels, 'Majorana')

    expected_op_strings =  \
    [OperatorString(['A'], [1], 'Majorana'), OperatorString(['A'], [2], 'Majorana'), \
     OperatorString(['B'], [1], 'Majorana'), OperatorString(['B'], [2], 'Majorana'), \
     OperatorString(['D'], [1], 'Majorana'), OperatorString(['D'], [2], 'Majorana'), \
     OperatorString(['A','A'], [1,2], 'Majorana'), OperatorString(['A','B'], [1,2], 'Majorana'), \
     OperatorString(['A','D'], [1,2], 'Majorana'), OperatorString(['B','B'], [1,2], 'Majorana'), \
     OperatorString(['B','D'], [1,2], 'Majorana'), OperatorString(['D','D'], [1,2], 'Majorana'), \
     OperatorString(['B','A'], [1,2], 'Majorana'), \
     OperatorString(['D','A'], [1,2], 'Majorana'), \
     OperatorString(['D','B'], [1,2], 'Majorana'), \
    ]

    assert(set(basis.op_strings) == set(expected_op_strings))

    k = 2
    cluster_labels = [1,2]
    basis = qy.cluster_basis(k, cluster_labels, 'Fermion')

    expected_op_strings = \
    [qy.opstring('CDag 1 C 1'), qy.opstring('CDag 2 C 2'), qy.opstring('CDag 1 CDag 2 C 2 C 1'),
     qy.opstring('CDag 1'), qy.opstring('1j CDag 1'),
     qy.opstring('CDag 2'), qy.opstring('1j CDag 2'),
     qy.opstring('CDag 2 C 1'), qy.opstring('1j CDag 2 C 1'),
     qy.opstring('CDag 1 CDag 2'), qy.opstring('1j CDag 1 CDag 2'),
     qy.opstring('CDag 1 CDag 2 C 1'), qy.opstring('1j CDag 1 CDag 2 C 1'),
     qy.opstring('CDag 1 CDag 2 C 2'), qy.opstring('1j CDag 1 CDag 2 C 2')]

    assert(set(basis.op_strings) == set(expected_op_strings))
    
def test_basis_addition():
    k = 2
    cluster_labelsA = [1,2]
    cluster_labelsB = [2,3]
    cluster_labelsC = [3,4]

    basisA = qy.cluster_basis(k, cluster_labelsA, 'Majorana')
    basisB = qy.cluster_basis(k, cluster_labelsB, 'Majorana')
    basisC = qy.cluster_basis(k, cluster_labelsC, 'Majorana')

    basisAB1 = basisA + basisB
    basisAB2 = Basis()
    basisAB2 += basisA
    basisAB2 += basisB
    
    basisAC1 = basisA + basisC
    basisAC2 = Basis()
    basisAC2 += basisA
    basisAC2 += basisC

    assert(basisAB1 == basisAB2)
    assert(basisAC1 == basisAC2)

def test_operator_addition():
    heisenberg_bond = Operator(np.array([1.,1.,1.]), [qy.opstring('X 1 X 2'), qy.opstring('Y 1 Y 2'), qy.opstring('Z 1 Z 2')])

    XX_bond = Operator(np.array([1.,1.]), [qy.opstring('X 1 X 2'), qy.opstring('Y 1 Y 2')])
    
    sum_bonds1 = heisenberg_bond + XX_bond
    sum_bonds2 = Operator(op_type='Pauli')
    sum_bonds2 += heisenberg_bond
    sum_bonds2 += XX_bond
    sum_bonds3 = Operator(np.array([2.,2.,1.]), [qy.opstring('X 1 X 2'), qy.opstring('Y 1 Y 2'), qy.opstring('Z 1 Z 2')])

    assert(sum_bonds1 == sum_bonds2)
    assert(sum_bonds2 == sum_bonds3)
    
    sum_bonds4 = heisenberg_bond - XX_bond
    sum_bonds4 = sum_bonds4.remove_zeros()
    sum_bonds5 = heisenberg_bond + (-1.0*XX_bond)
    sum_bonds5 = sum_bonds5.remove_zeros()
    sum_bonds6 = Operator(np.array([1.]), [qy.opstring('Z 1 Z 2')])

    assert(sum_bonds4 == sum_bonds5)
    assert(sum_bonds5 == sum_bonds6)
    
def test_distance_basis():
    N = 3

    # A periodic chain with one orbital per unit cell.
    chain_lattice = qy.lattice.chain(N, periodic=True)

    # When R == N-1, then the cluster basis and distance basis should be the same.
    k = 2
    R = N-1
    distance_basis = qy.basis.distance_basis(chain_lattice, k, R, 'Pauli')

    cluster_basis  = qy.basis.cluster_basis(k, np.arange(len(chain_lattice)), 'Pauli')

    assert(set(distance_basis.op_strings) == set(cluster_basis.op_strings))

    N = 8
    
    # An open chain with two orbitals per unit cell.
    chain_lattice = qy.lattice.chain(N, ['A', 'B'], periodic=False)
    
    k = 2
    R = 1
    distance_basis = qy.basis.distance_basis(chain_lattice, k, R, 'Pauli')
    
    intercell_clusters = [[site,site+1,site+2,site+3] for site in np.arange(0,2*N,2) if site+3 < 2*N]
    expected_basis = Basis()
    for cluster in intercell_clusters:
        expected_basis += qy.basis.cluster_basis(k, cluster, 'Pauli')


    print(distance_basis)
    print(expected_basis)
    assert(set(distance_basis.op_strings) == set(expected_basis.op_strings))
