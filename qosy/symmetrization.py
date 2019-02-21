#!/usr/bin/env python
"""
This module provides functions for symmetrizing Bases and Lattices.

"""

import numpy as np
import numpy.linalg as nla
import copy

from .lattice import UnitCell, Lattice
from .basis import Operator
from .transformation import label_permutation
from .tools import remove_duplicates, compose_permutations, argsort

def _symmetrize_opstring(op_string, symmetry_group):
    """Helper function to `symmetrize_basis`. 
    From an operator string h_a and a symmetry group G,
    computes  a symmetrized operator \\sum_{g \\in G} g h_a.
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
            
        symmetrized_op = symmetrized_op.remove_zeros()

        # Ignore zero operators.
        if symmetrized_op.norm() > tol:
            symmetrized_op.normalize()
            symmetrized_basis.append(symmetrized_op)
        
    return symmetrized_basis

def _symmetrize_atoms(lattice, G, tol=1e-12):
    # Helper function to `symmetrize_lattice`. Constructs
    # a new lattice with all atoms (and their orbitals)
    # symmetrized by the symmetry group transformations.

    sym_atom_info = list(zip(lattice.unit_cell.atom_positions, lattice.unit_cell.atom_orbitals))

    result = []
    for (pos, orbs) in sym_atom_info:
        for g in G:
            new_pos = np.dot(g, pos)

            result.append((new_pos, orbs))

    sym_atom_info = result

    # Sort the atom positions
    # by their distance from the origin.
    # Break ties by considering the distance
    # from the origin along each axis.
    distances = [(nla.norm(pos),)+tuple(np.abs(pos))+tuple(pos) for (pos, _) in sym_atom_info]
    def _comp(ind1, ind2):
        tup1 = distances[ind1]
        tup2 = distances[ind2]

        # Ignore equal entries.
        comparison_vec = np.abs(np.array(tup1) - np.array(tup2)) < tol
        index_tup = 0
        while comparison_vec[index_tup] and index_tup < len(tup1)-1:
            index_tup += 1

        # Use the last unequal entry for comparison.
        if comparison_vec[index_tup]:
            return 0
        else:
            return tup1[index_tup] - tup2[index_tup]

    inds_sort = argsort(distances, comp=_comp)
    sym_atom_info = [sym_atom_info[ind] for ind in inds_sort]

    # Construct the unit cell.
    new_unit_cell = UnitCell(lattice.unit_cell.lattice_vectors)
    for (pos, orbs) in sym_atom_info:
        new_unit_cell.add_atom(pos, orbs)

    # Construct the lattice.
    new_lattice = Lattice(new_unit_cell, (1,)*lattice.dim, periodic_boundaries=lattice.periodic_boundaries)

    return new_lattice

def _expand_lattice(lattice, ind_lattice_vector, delta):
    # Helper function to `symmetrize_lattice`. Expands
    # the lattice in a given lattice vector
    # direction.

    # First, extend the specified lattice vector
    # by the given amount.
    lattice.unit_cell.lattice_vectors[ind_lattice_vector] += delta

    # Second, enlarge the unit cell by shifting all
    # atoms in the unit cell by this amount as well.
    new_atom_pos  = list(lattice.unit_cell.atom_positions) \
                    + [delta + pos for pos in lattice.unit_cell.atom_positions]
    new_atom_orbs = list(lattice.unit_cell.atom_orbitals)*2

    new_unit_cell = UnitCell(lattice.unit_cell.lattice_vectors,
                             atom_positions=new_atom_pos,
                             atom_orbitals=new_atom_orbs)

    new_lattice = Lattice(new_unit_cell, (1,)*lattice.dim, periodic_boundaries=lattice.periodic_boundaries)

    return new_lattice

def _remove_duplicate_atoms(lattice, tol=1e-12):
    # Helper function for `symmetrize_lattice`.
    # Returns a new lattice with the
    # duplicate atoms from the lattice
    # removed.

    sym_atom_info = list(zip(lattice.unit_cell.atom_positions, lattice.unit_cell.atom_orbitals))

    # Function for comparing equality of atoms.
    # Compares their positions and orbitals in
    # the lattice.
    def _equiv(infoA, infoB):
        (posA, orbsA) = infoA
        (posB, orbsB) = infoB
        return lattice.distance(posA, posB) < tol and orbsA == orbsB

    # Remove duplicate atoms.
    sym_atom_info = remove_duplicates(sym_atom_info, equiv=_equiv)

    # Construct the unit cell.
    new_unit_cell = UnitCell(lattice.unit_cell.lattice_vectors)
    for (pos, orbs) in sym_atom_info:
        new_unit_cell.add_atom(pos, orbs)

    # Construct the lattice.
    new_lattice = Lattice(new_unit_cell, (1,)*lattice.dim, periodic_boundaries=lattice.periodic_boundaries)

    return new_lattice

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
