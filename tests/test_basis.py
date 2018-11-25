#!/usr/bin/env python
from .context import qosy as qy
from qosy.operator import OperatorString
import numpy as np

def test_cluster_basis():
    k = 2
    cluster_labels = [1,2]
    basis = qy.basis.cluster_basis(k, cluster_labels, 'Pauli')

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
    basis = qy.basis.cluster_basis(k, cluster_labels, 'Majorana')

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

    
def test_distance_basis():
    N = 3
    chain_lattice = qy.lattice.chain(N, boundary=('Open',))

    # When 2*R+1 == N, then the cluster basis and distance basis should be the same.
    k = 2
    R = 1
    distance_basis = qy.basis.distance_basis(chain_lattice, k, R, 'Pauli')

    print(distance_basis)

    print(chain_lattice.labels)

    cluster_basis  = qy.basis.cluster_basis(k, chain_lattice.labels, 'Pauli')

    print(cluster_basis)
    
    assert(set(distance_basis.op_strings) == set(cluster_basis.op_strings))
    
