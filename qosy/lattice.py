#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

class UnitCell:
    def __init__(self, labels, positions=None, lattice_vectors=None):
        """Construct a UnitCell object that represents a unit 
        cell of orbitals that can tile a lattice.
        
        Parameters
        ----------
        labels : array_like of hashable items, (N,)
            The labels of the N orbitals in the unit cell.
        positions : numpy array, (N,d), optional
            The d-dimensional coordinates of the N orbitals 
            in real space. Defaults to equally spaced positions 
            in d=1 dimension.
        lattice_vectors : numpy array, (d,d), optional
            Array whose columns are the bravais lattice vectors 
            that define the size of the unit cell. Defaults to 
            a d x d identity matrix.

        """
        self.labels          = labels
        self.positions       = positions
        self.lattice_vectors = lattice_vectors

        if self.positions is None:
            self.positions = np.zeros((len(self.labels),1))
            self.positions[:,0] = np.arange(len(self.labels))
        if self.lattice_vectors is None:
            self.lattice_vectors = np.eye(int(self.positions.shape[1]))
        
class Lattice:
    """A Lattice object represents a lattice tiled 
    by translated copies of a d-dimensional unit cell.
        
    Attributes
    ----------
    unit_cell : UnitCell object
        The unit cell that tiles the lattice.
    num_cells : tuple of int, (d,)
        The number of unit cells to tile in each of 
        the d directions of the lattice vectors.
        Defaults to all 1.
    boundary : tuple of str, (d,)
        The boundary conditions ('Open' or 'Periodic') 
        in each of the d directions of the lattice vectors. 
        Defaults to all 'Open'.
    labels : array_like of hashable items, (N,)
        The labels of the N orbitals in the lattice.
    positions : numpy array, (d,N)
        The d-dimensional position vectors of the N orbitals.
    """
    
    def __init__(self, unit_cell, num_cells=None, boundary=None):
        """Construct a Lattice object that represents a lattice tiled 
        by translated copies of a unit cell.
        
        Parameters
        ----------
        unit_cell : UnitCell object
            The unit cell that tiles the lattice.
        num_cells : tuple of int, (d,), optional
            The number of unit cells to tile in each of 
            the d directions of the lattice vectors.
            Defaults to all 1.
        boundary : tuple of str, (d,), optional
            The boundary conditions ('Open' or 'Periodic') 
            in each of the d directions of the lattice vectors. 
            Defaults to all 'Open'.
        """
        
        self.unit_cell = unit_cell
        self.num_cells = num_cells
        self.boundary  = boundary

        if self.num_cells is None:
            self.num_cells = tuple([1]*len(num_cells))
        
        if self.boundary is None:
            self.boundary = tuple(['Open']*len(num_cells))

        if len(boundary) != len(num_cells):
            raise ValueError('The size of boundary and num_cells in the lattice are inconsistent: {} and {}'.format(len(boundary), len(num_cells)))

        # Number of orbitals in lattice.
        N = len(self.unit_cell.labels)*np.prod(self.num_cells)
        # Dimensionality of lattice vectors.
        d = int(self.unit_cell.positions.shape[0])
        
        self.labels    = []
        self.positions = np.zeros((d,N))

        ind_pos = 0
        for ind_uc in range(len(self.unit_cell.labels)):
            label_uc    = self.unit_cell.labels[ind_uc]
            position_uc = self.unit_cell.positions[:,ind_uc]

            if type(label_uc) is not tuple:
                label_uc = (label_uc,)
            
            if len(self.num_cells) == 1:
                for ind1 in range(self.num_cells[0]):
                    label = label_uc + (ind1,)
                    pos   = position_uc + ind1*self.unit_cell.lattice_vectors[:,0]
                    self.labels.append(label)
                    self.positions[:,ind_pos] = pos
                    ind_pos += 1
                    
            elif len(self.num_cells) == 2:
                for ind1 in range(self.num_cells[0]):
                    for ind2 in range(self.num_cells[1]):
                        label = label_uc + (ind1,ind2)
                        pos   = position_uc + ind1*self.unit_cell.lattice_vectors[:,0] + ind2*self.unit_cell.lattice_vectors[:,1]
                        self.labels.append(label)
                        self.positions[:,ind_pos] = pos
                        ind_pos += 1
                        
            elif len(self.num_cells) == 3:
                for ind1 in range(self.num_cells[0]):
                    for ind2 in range(self.num_cells[1]):
                        for ind3 in range(self.num_cells[2]):
                            label = label_uc + (ind1,ind2,ind3)
                            pos   = position_uc + ind1*self.unit_cell.lattice_vectors[:,0] + ind2*self.unit_cell.lattice_vectors[:,1] + ind3*self.unit_cell.lattice_vectors[:,2]
                            self.labels.append(label)
                            self.positions[:,ind_pos] = pos
                            ind_pos += 1

    def distance(self, pos1, pos2):
        # TODO: implement mirror distance convention
        # Assumes open boundaries
        return np.linalg.norm(pos1-pos2)
                            
    def __str__(self):
        """Convert Lattice object to a string representation of the lattice.

        Returns
        -------
        str
            String representation of lattice.
        """

        list_strings = []
        for ind_label in range(len(self.labels)):
            label = self.labels[ind_label]
            pos   = self.positions[:,ind_label]
            list_strings += [str(label), ' ', str(pos), '\n']

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
    
    """
    
    # Plot the first unit cell's positions.
    x_uc = lattice.unit_cell.positions[0,:]
    y_uc = lattice.unit_cell.positions[1,:]
    plt.plot(x_uc, y_uc, 'rs', markeredgecolor='r', markersize=10)

    # Plot all of the lattice positions.
    x_l = lattice.positions[0,:]
    y_l = lattice.positions[1,:]
    plt.plot(x_l, y_l, 'ko', markeredgecolor='k', markersize=8)

    # Plot the labels of the lattice positions.
    if with_labels:
        # TODO:
        raise NotImplementedError('Plotting the labels is not implemented yet.')

    plt.axis('equal')
    
def show():
    """Display a figure.
    """

    plt.show()


def chain(N, boundary=('Open',)):
    """Construct a 1D chain lattice.

    Parameters
    ----------
    N : int
        The number of unit cells in the chain.
    boundary : (str,), optional
        The boundary condition ('Open' or 'Periodic').
        Defaults to ('Open',).
    """
    
    # Lattice spacing
    a  = 1.0
    # Lattice vector
    a1 = a * np.array([[1.0]])
    lattice_vectors = a1
    
    # The single orbital's label in the unit cell.
    labels = [1]
    # The single orbital's position in the unit cell.
    positions = np.zeros((1,1))
    
    # Construct the unit cell.
    unit_cell = UnitCell(labels, positions, lattice_vectors)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, (N,), boundary=boundary)

    return lattice
    
def kagome(N1, N2, boundary=('Open','Open')):
    """Construct a 2D Kagome lattice.

    Parameters
    ----------
    N1,N2 : int
        The number of unit cells in the 
        directions of the lattice vectors.
    boundary : (str, str), optional
        The boundary conditions ('Open' or 
        'Periodic') in the directions of 
        the lattice vectors. Defaults to 
        ('Open', 'Open').

    Returns
    -------
    Lattice
        A Lattice object representation 
        of the Kagome lattice.
    """

    # Lattice spacing
    a  = 1.0
    # Lattice vectors
    a1 = a * np.array([[1.0], [0.0]])
    a2 = a * np.array([[1.0/2.0], [np.sqrt(3.0)/2.0]])
    lattice_vectors = np.hstack((a1,a2))
    
    # Labels of the three sites in the unit cell.
    labels = [1,2,3]
    # positions of the three sites in the unit cell.
    r1 = np.zeros((2,1))
    r2 = a1 / 2.0
    r3 = a2 / 2.0
    positions = np.hstack((r1,r2,r3))

    # Construct the unit cell.
    unit_cell = UnitCell(labels, positions, lattice_vectors)
    
    # Construct the lattice.
    lattice = Lattice(unit_cell, (N1,N2), boundary=boundary)

    return lattice

def decorate(lattice, extra_labels):
    """Decorate a lattice's orbitals with extra labels.

    Parameters
    ----------
    lattice : Lattice object
        The lattice to decorate.
    extra_labels : array_like of hashable, (l,)
        The extra labels to assign to each orbital.

    Returns
    -------
    Lattice
        A new lattice with l times more labels.
    """

    result = Lattice(lattice.unit_cell, lattice.num_cells, lattice.boundary)

    num_extra_labels = len(extra_labels)
    
    new_labels    = []
    new_positions = np.zeros((int(lattice.positions.shape[0]), int(lattice.positions.shape[1])*num_extra_labels))
    ind_new_label = 0
    for ind_label in range(len(lattice.labels)):
        label = lattice.labels[ind_label]
        pos   = lattice.positions[:,ind_label]
        for ind_extra_label in range(num_extra_labels):
            extra_label = extra_labels[ind_extra_label]
            if type(extra_label) is not tuple:
                extra_label = (extra_label,)
            
            new_labels.append(label + extra_label)
            new_positions[:,ind_new_label] = pos
            
            ind_new_label += 1

    result.labels    = new_labels
    result.positions = new_positions

    return result
