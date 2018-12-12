#!/usr/bin/env python
"""
This module defines Unit Cell and Lattice classes that
conveniently handle the indexing of orbitals in a crystalline 
lattice of atoms.
"""

import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
                
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
    precision : float
        The precision with which to consider atomic coordinates
        equivalent.
    """
    
    def __init__(self, unit_cell, num_cells, periodic_boundaries=None, precision=3):
        self.unit_cell = unit_cell
        self.num_cells = num_cells

        self.precision = precision

        # A list of tuples of the positions, orbital names, and unit cell coordinates
        # of each orbital:
        # Ex: [(array([0,0]), 'A', (0,0)), (array([0,0]), 'B', (0,0)), (array([2,0]), 'A', (1,0)), ...]
        self._orbitals = []
        
        # Dictionary that maps tuples of position (as strings) and orbital names to
        # the index in the self._orbitals list.
        # Ex: self._indices_orbitals[('array([2 0])', 'A')] = 2
        self._indices_orbitals = dict()
        
        # Dictionary that maps 

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
            pos      = position + boundary_vector
            pos_name = np.array2string(pos.flatten(), precision=self.precision)
            
            if (pos_name, orbital_name) in self._indices_orbitals:
                return self._indices_orbitals[(pos_name, orbital_name)] 
            
        return -1
        
    def distance(self, index1, index2):
        """Compute the minimum distance between two orbitals with
        the given indices.

        Parameters
        ----------
        index1 : int
            The index of the first orbital.
        index2 : int
            The index of the second orbital.

        Returns
        -------
            The closest distance between the first and second orbital.
        """

        (pos1, orbital_name1, cell_coords1) = self._orbitals[index1]
        (pos2, orbital_name2, cell_coords2) = self._orbitals[index2]
        
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
        for (position, orbital_names) in self._orbitals:
            list_strings += [str(position)]
            for orbital_name in orbital_names:
                list_strings += [' ', str(orbital_name)]
            list_strings += ['\n']

        result = ''.join(list_strings)
        return result

def plot(lattice, with_labels=False):
    """Plot the lattice represented by the 
    Lattice object. Plots the unit cell that 
    tiles the lattice in a separate color.

    Parameters
    ----------
    lattice : Lattice
        The lattice to plot.
    with_labels : bool, optional
        Annotate each lattice point with its 
        label if True. False by default.
    
    Examples
    --------
       >>> qosy.plot(lattice)
       >>> qosy.show()
    """
    
    if lattice.dim == 1:
        # Plot the first unit cell separately.
        x_uc = []
        y_uc = []
        for position in lattice.unit_cell.atom_positions:
            x_uc.append(position[0])
            y_uc.append(0.0)

        plt.plot(x_uc, y_uc, 'rs', markeredgecolor='r', markersize=10, alpha=0.5)

        xs = []
        ys = []
        names = []
        label = 0
        for (position, orbital_name, cell_coords) in lattice:
            xs.append(position[0])
            ys.append(0.0)
            names.append('{}'.format(label))
            label += 1
            
        plt.plot(xs, ys, 'ko', markeredgecolor='k', markersize=8)

        if with_labels:
            for (x,y,name) in zip(xs,ys,names):
                plt.annotate(name, xy=(x,y), color='b')

        plt.xlim(np.min(xs)-1, np.max(xs)+1)
        plt.ylim(-1,1)

        plt.xlabel('$x$')
        plt.axis('equal')
                
    elif lattice.dim == 2:
        # Plot the first unit cell separately.
        x_uc = []
        y_uc = []
        for position in lattice.unit_cell.atom_positions:
            x_uc.append(position[0])
            y_uc.append(position[1])
    
        plt.plot(x_uc, y_uc, 'rs', markeredgecolor='r', markersize=10, alpha=0.5)
                
        xs = []
        ys = []
        names = []
        label = 0
        for (position, orbital_name, cell_coords) in lattice:
            xs.append(position[0])
            ys.append(position[1])
            names.append('{}'.format(label))
            label += 1
            
        plt.plot(xs, ys, 'ko', markeredgecolor='k', markersize=8)

        if with_labels:
            for (x,y,name) in zip(xs,ys,names):
                plt.annotate(name, xy=(x,y), color='b')
        
        plt.xlim(np.min(xs)-1, np.max(xs)+1)
        plt.xlim(np.min(ys)-1, np.max(ys)+1)


        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.axis('equal')
        
    elif lattice.dim == 3:
        fig = plt.gcf()
        ax  = fig.add_subplot(111, projection='3d')
        
        # Plot the first unit cell separately.
        x_uc = []
        y_uc = []
        z_uc = []
        for position in lattice.unit_cell.atom_positions:
            x_uc.append(position[0])
            y_uc.append(position[1])
            z_uc.append(position[2])

        ax.scatter(x_uc, y_uc, z_uc, c='r', marker='s')

        xs = []
        ys = []
        zs = []
        names = []
        label = 0
        for (position, orbital_name, cell_coords) in lattice:
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])
            names.append(str(label))
            label += 1
            
        ax.scatter(xs, ys, zs, c='k', marker='o')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        
        if with_labels:
            label = 0
            for (pos_atom, orbitals_atom) in zip(lattice.unit_cell.atom_positions, lattice.unit_cell.atom_orbitals):
                labels_atom = []
                for orbital in orbitals_atom:
                    labels_atom.append(label)
                    label += 1
                
                ax.text(pos_atom[0], pos_atom[1], pos_atom[2], str(labels_atom), size=20, zorder=1)
    else:
        raise ValueError('Cannot plot a {}-dimensional lattice.'.format(lattice.dim))
    
def show():
    """Display a figure.
    """

    plt.show()


def chain(N, orbital_names=None, periodic=False):
    """Construct a 1D chain lattice.

    Parameters
    ----------
    N : int
        The number of unit cells in the chain.
    orbital_names : list of hashable
        The names of the orbitals in each unit cell if there
        is more than one orbital per unit cell. Default is
        one (unnamed) orbital per unit cell.
    periodic : bool, optional
        Specifies whether the boundary condition is periodic
        rather than open. Default is False (open).

    Returns
    -------
    Lattice
        A 1D chain lattice.

    Examples
    --------
    To create a twelve site periodic chain
        >>> qosy.lattice.chain(12, periodic=True)
    """
    
    # Lattice spacing
    a  = 1.0
    # Lattice vector
    a1 = a * np.array([1.0])
    lattice_vectors = [a1]

    # Construct the unit cell: a single atom with many orbitals.
    unit_cell = UnitCell(lattice_vectors)
    unit_cell.add_atom(np.zeros(1), orbital_names)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, (N,), periodic_boundaries=(periodic,))

    return lattice
    
def kagome(N1, N2, periodic_boundaries=None):
    """Construct a 2D Kagome lattice.

    Parameters
    ----------
    N1,N2 : int
        The number of unit cells in the 
        directions of the lattice vectors.
    periodic_boundaries : (bool, bool), optional
        The periodic boundary conditions in 
        the directions of the lattice vectors. 
        Defaults to (False, False).

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
        unit_cell.add_atom(atom_position)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, (N1,N2), periodic_boundaries=periodic_boundaries)

    return lattice

# TODO
#def symmetrize(unit_cell, space_group_symmetries):
    # Construct a new, larger unit cell with the given
    # space group symmetries.

    
