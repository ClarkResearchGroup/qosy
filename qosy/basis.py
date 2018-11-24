#!/usr/bin/env python
import numpy as np
import itertools as it

from .operator import OperatorString
from .lattice import Lattice

class Basis:
    def __init__(self, op_strings):
        self.op_strings = op_strings
        self.indices    = dict()
        
        for ind_os in range(len(self.op_strings)):
            op_string = self.op_strings[ind_os]
            self.indices[op_string] = ind_os

    def __getitem__(self, key):
        if type(key) is OperatorString:
            return self.indices[key]
        else:
            return self.op_strings[key]

def cluster_basis(k, cluster_labels, op_type, include_identity=False):
    # The labels of the cluster in sorted order.
    cluster = np.sort(np.copy(cluster_labels))
    
    op_strings = []

    if include_identity:
        op_strings.append(OperatorString([], [], op_type)) # Identity operator
    
    max_num_operators = np.minimum(len(cluster), k)

    if op_type == 'Pauli' or op_type == 'Majorana':
        orbital_ops = ['X', 'Y', 'Z']
        if op_type == 'Majorana':
            orbital_ops = ['A', 'B', 'C']
        
        for num_operators in np.arange(1,max_num_operators+1):
            possible_ops    = list(it.product(orbital_ops, repeat=num_operators))
            possible_labels = list(it.combinations(cluster, num_operators))
            
            for labels in possible_labels:
                for ops in possible_ops:
                    op_string = OperatorString(ops, labels, op_type)
                    op_strings.append(op_string)
    elif op_type == 'Fermion':
        #op_strings.append('1 ') # The identity operator
        
        for num_operators_forward in np.arange(1,max_num_operators+1):
            possible_labels_forward = list(it.combinations(cluster, num_operators_forward))
            
            # Operator of type 1: c^\dagger_{i_1} ... c^\dagger_{i_m} c_{i_m} ... c_{i_1}
            for labels in possible_labels_forward:
                labels_forward  = np.copy(labels)
                labels_backward = np.copy(labels[::-1])

                ops    = ['CDag ']*len(labels_forward) + ['C ']*len(labels_backward)
                labels = labels_forward + labels_backward

                op_string = OperatorString(ops, labels, op_type)
                
                op_strings.append(op_string)

            # Operator of type 2 and 3: i^s * c^\dagger_{i_1} ... c^\dagger_{i_m} c_{j_l} ... c_{j_1} + H.c.
            for ind_labels_forward in range(len(possible_labels_forward)):
                labels_forward = possible_labels_forward[ind_labels_forward]
                
                for num_operators_backward in np.arange(0,num_operators_forward+1):
                    possible_labels_backward = list(it.combinations(cluster, num_operators_backward))
                    
                    max_ind_labels_backward = len(possible_labels_backward)
                    if num_operators_backward == num_operators_forward:
                        max_ind_labels_backward = ind_labels_forward
                
                    possible_prefactors = [1, 1j]
                    for prefactor in possible_prefactors:
                        for ind_labels_backward in range(max_ind_labels_backward):
                            labels_backward = possible_labels_backward[ind_labels_backward]

                            ops    = ['CDag ']*len(labels_forward) + ['C ']*len(labels_backward)
                            labels = labels_forward + labels_backward

                            op_string = OperatorString(ops, labels, op_type, prefactor)
                
                            op_strings.append(op_string)
    else:
        raise ValueError('Unknown operator string type: {}'.format(op_type))
    
    return Basis(op_strings)
    
def distance_basis(lattice, k, R, op_type, tol=1e-10):
    cluster_labels = []
    
    num_positions = len(lattice.positions)
    distances = np.zeros((num_positions, num_positions))
    for ind1 in range(num_positions):
        pos1 = lattice.positions[:,ind1]
        for ind2 in range(ind1+1,num_positions):
            pos2 = lattice.positions[:,ind2]
            
            distances[ind1,ind2] = lattice.distance(pos1, pos2)
            distances[ind2,ind1] = distances[ind1,ind2]
            
    cluster_labels = [[lattice.labels[ind2] for ind2 in range(num_positions) if np.abs(distances[ind1,ind2]-R) <= tol] for ind1 in range(num_positions)]
    
    return construct_cluster_basis(k, cluster_labels, op_type)
