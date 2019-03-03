#!/usr/bin/env python

# The names of the orbital operators for each type of operator string.
PAULI_OPS    = ('X', 'Y', 'Z')
MAJORANA_OPS = ('A', 'B', 'D')
FERMION_OPS  = ('CDag', 'C')
VALID_OPS    = PAULI_OPS + MAJORANA_OPS + FERMION_OPS

# Precomputed dictionaries that stores
# the rule for taking products of Pauli
# matrices and Majorana fermions.
PRODUCT_DICT = dict()
for (opA, opB, opC) in [PAULI_OPS, MAJORANA_OPS]:
    PRODUCT_DICT[(opA,opB)] = (1.0j, opC)
    PRODUCT_DICT[(opB,opA)] = (-1.0j, opC)
    
    PRODUCT_DICT[(opA,opC)] = (-1.0j, opB)
    PRODUCT_DICT[(opC,opA)] = (1.0j, opB)
    
    PRODUCT_DICT[(opB,opC)] = (1.0j, opA)
    PRODUCT_DICT[(opC,opB)] = (-1.0j, opA)
