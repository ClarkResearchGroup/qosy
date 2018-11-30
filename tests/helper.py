#!/usr/bin/env python
from .context import qosy as qy
import itertools as it
import numpy as np

def _random_op_string(max_num_orbitals, possible_orbital_labels, op_type):
    # Creates a random operator string made of up to max_num_orbital
    # orbitals with labels drawn from possible_orbital_labels.
    
    if op_type == 'Pauli':
        ops = qy.PAULI_OPS
    elif op_type == 'Majorana':
        ops = qy.MAJORANA_OPS
    elif op_type != 'Fermion':
        raise NotImplementedError('Cannot create random op_type: {}'.format(op_type))

    prefactor = 1.0
    
    num_orbitals = np.random.randint(1, max_num_orbitals+1)

    if op_type in ['Pauli', 'Majorana']:
        orbital_operators = np.random.choice(ops, num_orbitals)
        orbital_labels    = np.random.permutation(possible_orbital_labels)[0:num_orbitals]
        orbital_labels    = np.sort(orbital_labels)
    elif op_type == 'Fermion':
        # Choose one of the three types of Fermion string operators.
        fermion_type = np.random.choice([1,2,3])
        if fermion_type == 3:
            prefactor = 1j

        # Randomly choose the c^\dagger_{i_1}...c^\dagger_{i_m} indices.
        combs1    = list(it.combinations(possible_orbital_labels, num_orbitals))
        ind_comb1 = np.random.randint(0,len(combs1))
        comb1     = combs1[ind_comb1]

        if fermion_type == 1:
            comb2 = comb1[::-1]
        else:
            num_orbitals2 = np.random.randint(0, num_orbitals)
            
            # Randomly choose the c_{j_l}...c_{j_1} indices.
            combs2    = list(it.combinations(possible_orbital_labels, num_orbitals2))
            ind_comb2 = np.random.randint(0,len(combs2))
            comb2     = combs2[ind_comb2]
            comb2     = comb2[::-1]
                
        orbital_operators = ['CDag']*len(comb1) + ['C']*len(comb2)
        orbital_labels    = list(comb1) + list(comb2)

    return qy.OperatorString(orbital_operators, orbital_labels, op_type, prefactor=prefactor)

def createRandomFermionHamiltonian(numTerms, maxK, sites):
    randomFermionHam = []
    for indTerm in range(numTerms):
        opName = ''
        fermionType = np.random.choice([1,2,3])
        if fermionType == 3:
            opName += 'I '
        else:
            opName += '1 '
            
        k1       = np.random.randint(1,maxK+1)
        combs1   = list(it.combinations(sites,k1))
        indComb1 = np.random.randint(0,len(combs1))
        comb1    = combs1[indComb1]
        
        if fermionType == 1:
            for site in comb1:
                opName += 'CDag {} '.format(site)
            for site in comb1[::-1]:
                opName += 'C {} '.format(site)
        else:
            k2 = np.random.randint(0,k1)
            combs2   = list(it.combinations(sites,k2))
            indComb2 = np.random.randint(0,len(combs2))
            comb2    = combs2[indComb2]

            for site in comb1:
                opName += 'CDag {} '.format(site)
            for site in comb2[::-1]:
                opName += 'C {} '.format(site)

        coeff = np.random.randn()
            
        randomFermionHam.append((coeff, opName))

    randomFermionHam = coi.clean(randomFermionHam)

    return randomFermionHam
