#!/usr/bin/env python
from .context import qosy as qy
import numpy as np

def _random_op_string(max_num_orbitals, possible_orbital_labels, op_type):
    # Creates a random operator string made of up to max_num_orbital
    # orbitals with labels drawn from possible_orbital_labels.
    
    if op_type == 'Pauli':
        ops = qy.PAULI_OPS
    elif op_type == 'Majorana':
        ops = qy.MAJORANA_OPS
    else:
        raise NotImplementedError('Cannot create random op_type: {}'.format(op_type))

    num_orbitals = np.random.randint(max_num_orbitals)+1

    orbital_operators = np.random.choice(ops, num_orbitals)
    orbital_labels    = np.random.permutation(possible_orbital_labels)[0:num_orbitals]
    orbital_labels    = np.sort(orbital_labels)

    return qy.OperatorString(orbital_operators, orbital_labels, op_type)
