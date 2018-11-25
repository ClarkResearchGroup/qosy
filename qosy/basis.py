#!/usr/bin/env python
import numpy as np
import itertools as it
import copy

from .operator import OperatorString, Operator
from .lattice import Lattice

class Basis:
    def __init__(self, op_strings=[]):
        self.op_strings = op_strings
        self.indices    = dict()
        
        for ind_os in range(len(self.op_strings)):
            op_string = self.op_strings[ind_os]
            self.indices[op_string] = ind_os

        if len(self.indices) != len(self.op_strings):
            raise ValueError('Tried to create a basis with multiple copies of the same operator string. There were {} operator strings, but only {} uniques ones.'.format(len(self.op_strings),len(self.indices)))

    def index(self, op_string):
        return self.indices[op_string]

    def __iter__(self):
        return iter(self.op_strings)

    def __getitem__(self, key):
        return self.op_strings[key]

    def __len__(self):
        return len(self.op_strings)

    def __str__(self):
        list_strings = []
        for ind_os in range(len(self.op_strings)):
            os = self.op_strings[ind_os]
            list_strings += [str(os), '\n']

        result = ''.join(list_strings)
        return result

    def __contains__(self, item):
        return item in self.indices

    def __add__(self, other):
        if isinstance(other, Basis):
            new_op_strings = [os for os in other.op_strings if os not in self]
        elif isinstance(other, OperatorString):
            new_op_strings = [other]
        elif isinstance(other, Operator):
            new_op_strings = [other.op_strings]
        else:
            raise TypeError('Cannot add object of type {} to basis.'.format(type(other)))

        return Basis(self.op_strings + new_op_strings)

def cluster_basis(k, cluster_labels, op_type, include_identity=False):
    # The labels of the cluster in sorted order.
    cluster = copy.deepcopy(cluster_labels)
    cluster.sort()
    
    op_strings = []

    if include_identity:
        op_strings.append(OperatorString([], [], op_type)) # Identity operator
    
    max_num_operators = np.minimum(len(cluster), k)

    if op_type == 'Pauli' or op_type == 'Majorana':
        orbital_ops = ['X', 'Y', 'Z']
        if op_type == 'Majorana':
            orbital_ops = ['A', 'B', 'D']
        
        for num_operators in np.arange(1,max_num_operators+1):
            possible_ops    = list(it.product(orbital_ops, repeat=num_operators))
            possible_labels = list(it.combinations(cluster, num_operators))
            
            for labels in possible_labels:
                for ops in possible_ops:
                    op_string = OperatorString(list(ops), list(labels), op_type)
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
    num_positions = int(lattice.positions.shape[1])
    distances = np.zeros((num_positions, num_positions))
    for ind1 in range(num_positions):
        pos1 = lattice.positions[:,ind1]
        for ind2 in range(ind1+1,num_positions):
            pos2 = lattice.positions[:,ind2]
            
            distances[ind1,ind2] = lattice.distance(pos1, pos2)
            distances[ind2,ind1] = distances[ind1,ind2]
            
    clusters = [[lattice.labels[ind2] for ind2 in range(num_positions) if distances[ind1,ind2] <= R+tol] for ind1 in range(num_positions)]

    total_basis = Basis()
    for cluster_labels in clusters:
        total_basis += cluster_basis(k, cluster_labels, op_type)

    return total_basis
