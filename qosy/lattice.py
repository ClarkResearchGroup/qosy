#!/usr/bin/env python
"""
This module defines Unit Cell and Lattice classes that
conveniently handle the indexing of orbitals in a crystalline 
lattice of atoms.
"""

import copy
import itertools as it
import numpy as np
import numpy.linalg as nla

from .tools import argsort, remove_duplicates

class UnitCell:
    """A UnitCell object represents a unit cell of atoms
    with orbitals on the atoms.

    Attributes
    ----------
    lattice_vectors : list of ndarray
        The primitive lattice vectors that indicate the
        size of the unit cell. The vectors are stored in
        the columns of this array.
    atom_positions : list of ndarray
        The positions of the atoms in the UnitCell.
    atom_orbitals : list of list of hashable
        The list of orbital names for each atom.
    """
    
    def __init__(self, lattice_vectors, atom_positions=None, atom_orbitals=None):
        self.lattice_vectors = lattice_vectors

        if atom_positions is None:
            self.atom_positions = []
        else:
            self.atom_positions = atom_positions
            
        if atom_orbitals is None:
            self.atom_orbitals = []
        else:
            self.atom_orbitals = atom_orbitals

    def add_atom(self, position, orbitals=None):
        """Add an atom to the UnitCell.

        Parameters
        ----------
        position : ndarray, (d,)
            The position of the atom in the UnitCell.
        orbitals : list of hashable, optional
            The names of the orbitals associated with the atom.
            By default creates a single orbital with name ''.
        """
        
        self.atom_positions.append(position)
        
        if orbitals is None:
            self.atom_orbitals.append([''])
        else:
            self.atom_orbitals.append(orbitals)

class Lattice:
    """A Lattice object represents a lattice of orbitals 
    attached to atoms that is tiled by translated copies 
    of a d-dimensional unit cell.
        
    Attributes
    ----------
    unit_cell : UnitCell
        The UnitCell of atoms that tiles the Lattice.
    num_cells : list, array, or tuple of int, (d,)
        The number of unit cells to tile in each of 
        the d directions of the lattice vectors.
    periodic_boundaries : list, array, or tuple of bool, (d,)
        The boundary conditions in each of the d lattice vector
        directions. True means periodic, False means open.
    dim : int
        The dimension d of the Lattice.
    precision : int
        The precision with which to consider atomic coordinates
        equivalent.
    """
    
    def __init__(self, unit_cell, num_cells, periodic_boundaries=None, precision=8):
        self.unit_cell = unit_cell
        self.num_cells = num_cells

        self.precision = precision
        self.tol       = 10.0**(-precision)

        # A list of tuples of the positions, orbital names, and unit cell coordinates
        # of each orbital:
        # Ex: [(array([0,0]), 'A', (0,0)), (array([0,0]), 'B', (0,0)), (array([2,0]), 'A', (1,0)), ...]
        self._orbitals = []
        
        # Dictionary that maps tuples of position (as strings) and orbital names to
        # the index in the self._orbitals list.
        # Ex: self._indices_orbitals[('array([2 0])', 'A')] = 2
        self._indices_orbitals = dict()

        ranges       = tuple([range(num_cell) for num_cell in num_cells])
        coords_cells = list(it.product(*ranges))

        self.dim = len(num_cells)

        if periodic_boundaries is None:
            self.periodic_boundaries = np.zeros(self.dim, dtype=bool)
        else:
            self.periodic_boundaries = np.array(periodic_boundaries, dtype=bool)
            
        for coords_cell in coords_cells:
            for (pos_atom, orbitals_atom) in zip(self.unit_cell.atom_positions, self.unit_cell.atom_orbitals):
                pos = np.zeros(self.dim)
                for ind_vec in range(self.dim):
                    pos += coords_cell[ind_vec] * unit_cell.lattice_vectors[ind_vec]
                pos += pos_atom

                # Make sure the zeros are exactly zero (no floating point roundoff).
                pos      = np.array([x if np.abs(x) > self.tol else 0.0 for x in pos])
                pos_name = np.array2string(pos.flatten(), precision=self.precision)

                for orbital_name in orbitals_atom:
                    ind_orbital = len(self._orbitals)
                    self._orbitals.append((pos, orbital_name, coords_cell))
                    self._indices_orbitals[(pos_name, orbital_name)] = ind_orbital
                    
        # These vectors translate one periodic edge of a lattice
        # to another. They are used to check for wrapping of points
        # around the periodic edges.
        self._boundary_vectors = [np.zeros(self.dim)]
        for ind_vec in range(self.dim):
            if self.periodic_boundaries[ind_vec]:
                previous_bvecs = list(self._boundary_vectors)
                boundary_vec   = self.num_cells[ind_vec] * self.unit_cell.lattice_vectors[ind_vec]

                self._boundary_vectors = previous_bvecs \
                                         + [boundary_vec + pbv for pbv in previous_bvecs] \
                                         + [-boundary_vec + pbv for pbv in previous_bvecs]
                    
    def index(self, position, orbital_name=''):
        """Return the index of the orbital whose atom is located
        at the given position in the Lattice.

        Parameters
        ----------
        position : ndarray, (d,)
            The coordinate of the atom associated with the orbital.
            If the lattice is periodic in any direction and `position`
            is outside of the periodic boundaries of the lattice, it 
            will be wrapped back into the lattice.
        orbital_name : hashable, optional
            The name of the orbital.

        Returns
        -------
        int
            The index of the orbital in the lattice.
        """
        
        # Translate by the boundary vectors associated with
        # the periodic edges. The first boundary vector is the
        # zero vector, which corresponds to no translation.
        for boundary_vector in self._boundary_vectors:
            # Shift the position.
            pos      = position + boundary_vector
            # Make sure the zeros are exactly zero (no floating point roundoff).
            pos      = np.array([x if np.abs(x) > self.tol else 0.0 for x in pos])
            
            pos_name = np.array2string(pos.flatten(), precision=self.precision)
            
            if (pos_name, orbital_name) in self._indices_orbitals:
                return self._indices_orbitals[(pos_name, orbital_name)] 
            
        return -1
        
    def distance(self, position1, position2):
        """Compute the minimum distance between
        two positions in the given lattice.

        Parameters
        ----------
        position1 : ndarray or int
            The spatial position in the lattice
            or the index of the orbital whose postiion
            to use.
        position2 : ndarray or int
            The spatial position in the lattice
            or the index of the orbital whose postiion
            to use.

        Returns
        -------
            The closest distance between the two positions.

        Notes
        -----
        At least one of the two positions needs to be within
        the boundaries of the lattice for the periodic
        boundaries to be computed correctly.
        """

        if type(position1) is int:
            (pos1, _, _) = self._orbitals[position1]
        else:
            pos1 = position1
            
        if type(position2) is int:
            (pos2, _, _) = self._orbitals[position2]
        else:
            pos2 = position2
            
        mirror_distances = [np.linalg.norm((pos1+bv)-pos2) for bv in self._boundary_vectors]
        return np.min(mirror_distances)
    
    def __iter__(self):
        """Return an iterator over the positions, the names,
        and the unit cell coordinates of the orbitals.

        Returns
        -------
        iterator over tuples of ndarray and hashable

        Examples
        --------
        To collect a list of positions of all orbitals named `'B'`:
            >>> positionsB = [position for (position, orbital_name, cell_coords) in lattice if orbital_name == 'B']
        """
        
        return iter(self._orbitals)

    def __len__(self):
        """Return the number of orbitals in the Lattice.
        """
        
        return len(self._orbitals)
                            
    def __str__(self):
        """Convert Lattice to a python string representation.

        Returns
        -------
        str
            String representation of the Lattice.

        Examples
        --------
            >>> print(lattice)
        """

        list_strings = []
        for (position, orbital_name, cell_coords) in self._orbitals:
            list_strings += [str(position), ' ', str(cell_coords), ' ', str(orbital_name), '\n']

        result = ''.join(list_strings)
        return result

def cubic(d, num_cells, periodic_boundaries=None, orbital_names=None):
    """Construct a :math:`d`-dimensional cubic lattice
    with one atom per unit cell.

    Parameters
    ----------
    d : int
        The spatial dimension :math:`d` of the cubic lattice.
    num_cells : tuple of int, (d,)
        The number of unit cells to repeat in each dimension.
    periodic_boundaries : tuple of bool, (d,), optional
        Specifies whether the boundary conditions are periodic
        or open in all spatial directions. Default is False (open)
        in all directions.
    orbital_names, list of hashable, (d,), optional
        The names of the orbitals on each atom. Defaults
        to a single orbital per atom with name ''.

    Returns
    -------
    Lattice
        A :math:`d`-dimensional cubic lattice.

    Examples
    --------
    To create a 4x4x4 periodic 3D cubic lattice
        >>> qosy.lattice.cubic(3, (4,4,4), periodic=(True,True,True))
    """

    if periodic_boundaries is None:
        periodic_boundaries = (False,)*d

    if orbital_names is None:
        orbital_names = ['']
    
    # Lattice spacing
    a  = 1.0
    # Lattice vectors
    lattice_vectors = []
    for ind_dim in range(d):
        lattice_vec = np.zeros(d)
        lattice_vec[ind_dim] = a
        lattice_vectors.append(lattice_vec)
    
    # Construct the unit cell: a single atom with many orbitals.
    unit_cell = UnitCell(lattice_vectors)
    unit_cell.add_atom(np.zeros(d), orbitals=orbital_names)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, num_cells, periodic_boundaries=periodic_boundaries)

    return lattice

def square(N1, N2, periodic_boundaries=None, orbital_names=None):
    """Construct a 2D square lattice.

    Parameters
    ----------
    N1 : int
        The number of unit cells in the 
        directions of the :math:`a_1` lattice vector.
    N2 : int
        The number of unit cells in the 
        directions of the :math:`a_2` lattice vector.
    periodic_boundaries : (bool, bool), optional
        The periodic boundary conditions in 
        the directions of the lattice vectors. 
        Defaults to (False, False).
    orbital_names : list of hashable
        The names of the orbitals in each unit cell if there
        is more than one orbital per unit cell. Default is
        one (unnamed) orbital per unit cell.

    Returns
    -------
    Lattice
        A 2D square lattice.

    Examples
    --------
    To create a 4x4 toroidal square lattice
        >>> qosy.lattice.square(4, 4, periodic_boundaries=(True,True))
    """
    
    return cubic(2, (N1,N2), periodic_boundaries=periodic_boundaries, orbital_names=orbital_names)

def chain(N, periodic=False, orbital_names=None):
    """Construct a 1D chain lattice.

    Parameters
    ----------
    N : int
        The number of unit cells in the chain.
    periodic : bool, optional
        Specifies whether the boundary condition is periodic
        rather than open. Default is False (open).
    orbital_names : list of hashable
        The names of the orbitals in each unit cell if there
        is more than one orbital per unit cell. Default is
        one (unnamed) orbital per unit cell.

    Returns
    -------
    Lattice
        A 1D chain lattice.

    Examples
    --------
    To create a twelve-site periodic chain
        >>> qosy.lattice.chain(12, periodic=True)
    """

    return cubic(1, (N,), periodic_boundaries=(periodic,), orbital_names=orbital_names)

def kagome(N1, N2, periodic_boundaries=None, orbital_names=None):
    """Construct a 2D Kagome lattice.

    Parameters
    ----------
    N1 : int
        The number of unit cells in the 
        directions of the :math:`a_1` lattice vector.
    N2 : int
        The number of unit cells in the 
        directions of the :math:`a_2` lattice vector.
    periodic_boundaries : (bool, bool), optional
        The periodic boundary conditions in 
        the directions of the lattice vectors. 
        Defaults to (False, False).
    orbital_names : list of hashable
        The names of the orbitals in each unit cell if there
        is more than one orbital per unit cell. Default is
        one (unnamed) orbital per unit cell.

    Returns
    -------
    Lattice
        A Lattice object representation 
        of the Kagome lattice.

    Examples
    --------
    To create a 4 x 3 Kagome lattice on a cylinder geometry
        >>> qosy.lattice.kagome(4, 3, (True, False))
    """

    if periodic_boundaries is None:
        periodic_boundaries = (False, False)

    # Lattice spacing
    a  = 1.0
    # Lattice vectors
    a1 = a * np.array([1.0, 0.0])
    a2 = a * np.array([1.0/2.0, np.sqrt(3.0)/2.0])
    lattice_vectors = [a1, a2]
    
    # positions of the three sites in the unit cell.
    r1 = np.zeros(2)
    r2 = a1 / 2.0
    r3 = a2 / 2.0
    positions = [r1,r2,r3]

    # Construct the unit cell.
    unit_cell = UnitCell(lattice_vectors)
    for atom_position in positions:
        unit_cell.add_atom(atom_position, orbitals=orbital_names)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, (N1,N2), periodic_boundaries=periodic_boundaries)

    return lattice
