#!/usr/bin/env python
"""
This module provides functions for symmetrizing Bases and Lattices.

"""

import numpy as np
import copy

from .lattice import UnitCell, Lattice
from .basis import Operator
from .transformation import label_permutation
from .tools import remove_duplicates, compose_permutations

def _symmetrize_opstring(op_string, symmetry_group):
    """From an operator string h_a and a symmetry group G,
    computes a symmetrized operator \\sum_{g \\in G} g h_a.
    """
    
    symmetrized_op = Operator([], [], op_string.op_type)
    for g in symmetry_group:
        symmetrized_op += g.apply(op_string)

    return symmetrized_op

def symmetrize_basis(basis, group_generators, tol=1e-12):
    """Symmetrize a basis so that its basis
    vectors are invariant under the given
    discrete symmetries.

    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings to symmetrize.
    group_generators : list of Transformations
        The generators of the desired discrete 
        symmetry group.
    tol : float, optional
        The tolerance used to check if an
        Operator is a zero Operator. Default is 1e-12.

    Returns
    -------
    list of Operators
        The symmetrized basis, represented as a 
        list of Operators. Each Operator obeys 
        the given space group symmetries.

    Notes
    -----
    As this is currently implemented, the Transformations
    must be Transformations that permute the labels
    of orbitals, such as point group and space group
    symmetries.
    """

    # Construct the symmetry group G from its generators.
    num_orbitals = len(group_generators[0].info)
    I = label_permutation(np.arange(num_orbitals)) # Identity transformation
    G = [I]
    G_set = set(G)
    newG = G + group_generators
    newG_set = set([tuple(g.info) for g in newG])
    
    while len(newG) != len(G):
        G = newG
        G_set = newG_set
        
        # Generate new group elements
        for gen in group_generators:
            for g in G:
                perm       = compose_permutations(g.info, gen.info)
                perm_tuple = tuple(perm)
                if perm_tuple not in newG_set:
                    newG_set.add(perm_tuple)
                    perm_transformation = label_permutation(perm)
                    newG.append(perm_transformation)
        
    # The symmetrized basis will be a list of Operators
    # that obey the given discrete group symmetries.
    symmetrized_basis  = []
    visited_op_strings = set()
    for op_string in basis:
        if op_string in visited_op_strings:
            continue
        
        symmetrized_op = _symmetrize_opstring(op_string, G)

        for (coeff, visited_op_string) in symmetrized_op:
            visited_op_strings.add(visited_op_string)
            
        symmetrized_op.remove_zeros()

        # Ignore zero operators.
        if symmetrized_op.norm() > tol:
            symmetrized_op.normalize()
            symmetrized_basis.append(symmetrized_op)
        
    return symmetrized_basis

# TODO: test
def symmetrize_lattice(lattice, point_group_generators, num_expansions=1, tol=1e-12):
    """Construct a lattice with a
    new, larger unit cell that is 
    invariant under the given
    point group symmetries.

    Parameters
    ----------
    lattice : Lattice
        The lattice to symmetrize.
    point_group_generators : list of ndarray
        The discrete point group symmetry
        transformations that generator the 
        symmetry group, represented by matrices.
    num_expansions : int, optional
        The number of times to attempt to
        enlarge the unit cell. Defaults to 1.
    tol : float, optional
        The tolerance within which to consider
        positions equivalent. Defaults to 1e-12.

    Returns
    -------
    Lattice
        The symmetrized lattice with an
        enlarged unit cell with the desired 
        symmetries.

    Notes
    -----
    This function performs the simplest
    possible expansion of the original unit
    cell. Namely, it expands each lattice
    vector. This will miss certain unit
    cells that also obey the given point
    group.
    """

    # The original lattice's unit cell
    unit_cell = lattice.unit_cell
    
    # The spatial dimension
    dim = lattice.dim

    # Construct the point group G from its generators.
    G = [np.eye(dim)]
    newG = G + point_group_generators
    while len(newG) != len(G):
        G = newG
        
        # Generate new group elements, potentially with duplicates
        newG = [np.dot(g,gen) for g in G for gen in point_group_generators]
        # Remove the duplicates
        newG = remove_duplicates(newG)

    # Store the original lattice.
    orig_lattice         = copy.deepcopy(lattice)
    orig_lattice_vectors = copy.deepcopy(orig_lattice.unit_cell.lattice_vectors)
        
    # Build the new unit cell by symmetrizing
    # the old unit cell.
    old_lattice = copy.deepcopy(lattice)

    # Expand and symmetrize the unit cell.
    for ind_expansion in range(num_expansions):
        # Expand the lattice in the direction of each
        # lattice vector.
        for ind_lv in range(len(orig_lattice_vectors)):
            delta = orig_lattice_vectors[ind_lv]

            # Expand the lattice in the direction of delta.
            new_lattice = _expand_lattice(old_lattice, ind_lv, delta)
        
        # Symmetrize the atom positions for the new
        # unit cell.
        new_lattice = _symmetrize_atoms(new_lattice, G, tol=tol)

        # Remove the duplicate atoms equivalent to
        # one another by lattice translations
        # by the new lattice vectors.
        new_lattice = _remove_duplicate_atoms(new_lattice, tol=tol)

        # Update the old lattice.
        old_lattice = new_lattice
    
    return new_lattice
