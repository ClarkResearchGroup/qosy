#!/usr/bin/env python
"""
This module provides some tools for visualizing
a lattice and interactions on a lattice. Also,
printing tools are provided.
"""

import numpy as np
import scipy.spatial as ssp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .basis import Operator
from .conversion import convert
from .tools import replace

def relabel_orbitals(string, lattice):
    """Relabel the operator string orbitals 
    referenced in the given python string to 
    the coordinates of the Lattice.

    Parameters
    ----------
    string : str
        A string that contains orbital labels.
    lattice : Lattice
        A lattice containing those orbitals.

    Returns
    -------
    str
        A new string with the orbital labels
        replaced by those orbitals' coordinates
        on the lattice.
    """

    num_orbitals = len(lattice)

    substitutions = dict()
    for orbital_label in np.arange(num_orbitals-1,-1,-1):
        (lattice_pos, orb_name, unit_cell_coords) = lattice._orbitals[orbital_label]

        """
        # Compute the position of the atom within
        # the unit cell.
        unit_cell_pos = lattice_pos
        for ind_dim in range(lattice.dim):
            unit_cell_pos -= unit_cell_coords[ind_dim] * lattice.unit_cell.lattice_vectors[ind_dim]
        """

        # Compute the index of the atom within the
        # unit cell.
        unit_cell_num_orbitals  = num_orbitals // np.prod(lattice.num_cells)
        unit_cell_orbital_index = orbital_label % unit_cell_num_orbitals

        unit_cell_atom_index = 0
        index1 = 0
        for ind_atom in range(len(lattice.unit_cell.atom_orbitals)-1):
            num_orbitals_on_atom = len(lattice.unit_cell.atom_orbitals[ind_atom])
            index2 = index1 + num_orbitals_on_atom
            
            if index1 <= unit_cell_orbital_index and unit_cell_orbital_index < index2:
                break

            index1 = index2
            unit_cell_atom_index += 1

        # The information of the orbital with
        # respect to the lattice:
        # 1) its unit cell coordinate if there is more than
        # one unit cell in the lattice.
        orbital_info = '{'
        if np.prod(lattice.num_cells) != 1:
            orbital_info += str(unit_cell_coords)+'; '

        # 2) The index of the atom in the unit cell.
        orbital_info += str(unit_cell_atom_index) #np.array2string(unit_cell_pos,precision=3) 

        # 3) Any extra orbital information (like "Up" or "Dn"
        # for spin-1/2 orbitals).
        if orb_name !='':
            orbital_info += '; '+orb_name+'}'
        else:
            orbital_info += '}'
        
        substitutions[' '+str(orbital_label)] = '_'+str(orbital_info)
    
    result = replace(string, substitutions)
    
    return result
        
def print_vectors(basis, vectors, lattice=None, convert_to=None, norm_order=None):
    """Print in human-readable form the vectors
    as Operators in the given Basis.
    
    Parameters
    ----------
    basis : Basis
        The Basis of OperatorStrings that the vector
        is represented in.
    vectors : ndarray
        The vectors to print.
    lattice : Lattice, optional
        If provided, convert orbital labels to 
        lattice coordinate labels. Defaults to None.
    convert_to : str, optional
        If not None, convert OperatorStrings to the given type
        before printing. Defaults to None.
    norm_order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Normalizes the vector to have a norm of this order 
        (see `numpy.norm`). Defaults to None, which is the 
        :math:`\\ell_2`-norm. Another useful norm is the `inf` 
        order norm.
    """
    
    if len(basis) != int(vectors.shape[0]):
        raise ValueError('Vectors are not of the right size {} to be in the basis of dimension {}'.format(vectors.shape, len(basis)))

    num_vecs = int(vectors.shape[1])
    
    operators = [Operator(vectors[:,ind_vec], basis.op_strings) for ind_vec in range(num_vecs)]

    if convert_to is not None:
        operators = [convert(op, convert_to) for op in operators]
    
    ind_vec = 1
    for operator in operators:
        cleaned_operator = operator.remove_zeros()
        cleaned_operator.normalize(order=norm_order)
        print('vector {} = '.format(ind_vec))

        output_string = str(cleaned_operator)
        if lattice is not None:
            output_string = relabel_orbitals(output_string, lattice)
        
        print(output_string)
        
        ind_vec += 1

def print_operators(operators, lattice=None, convert_to=None, norm_order=None):
    """Print in human-readable form a list of Operators.
    
    Parameters
    ----------
    operators : list of Operators
        The Operators to print.
    lattice : Lattice, optional
        If provided, convert orbital labels to 
        lattice coordinate labels. Defaults to None.
    convert_to : str, optional
        If not None, convert OperatorStrings to the given type
        before printing. Defaults to None.
    norm_order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Normalizes the vector to have a norm of this order 
        (see `numpy.norm`). Defaults to None, which is the 
        :math:`\\ell_2`-norm. Another useful norm is the `inf` 
        order norm.
    """
    
    if convert_to is not None:
        operators_to_print = [convert(op, convert_to) for op in operators]
    else:
        operators_to_print = operators
    
    ind_vec = 1
    for operator in operators_to_print:
        cleaned_operator = operator.remove_zeros()
        cleaned_operator.normalize(order=norm_order)

        print('operator {} = '.format(ind_vec))

        output_string = str(cleaned_operator)
        if lattice is not None:
            output_string = relabel_orbitals(output_string, lattice)
        
        print(output_string)

        ind_vec += 1

def plot(lattice, with_labels=False, with_lattice_vectors=True, with_wigner_seitz=True):
    """Plot the lattice represented by the 
    Lattice object. Plots the unit cell that 
    tiles the lattice in a separate color.

    Parameters
    ----------
    lattice : Lattice
        The lattice to plot.
    with_labels : bool, optional
        Specifies whether to annotate 
        each orbital with its 
        integer label. False by default.
    with_lattice_vectors : bool, optional
        Specifies whether to plot the lattice 
        vectors. True by default. 
    with_wigner_seitz : bool, optional
        Specifies whether to depict 
        the Wigner-Seitz unit cell boundary.
  
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

        if with_lattice_vectors:
            for v in lattice.unit_cell.lattice_vectors:
                plt.plot([0.0, v[0]], [0.0, v[1]], 'r.')
                plt.arrow(0.0, 0.0, v[0], v[1], color='r', head_width=0.1, length_includes_head=True)

        xmin = np.min(xs + [v[0] for v in lattice.unit_cell.lattice_vectors])
        xmax = np.max(xs + [v[0] for v in lattice.unit_cell.lattice_vectors])
        
        plt.xlim(xmin-1, xmax+1)
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

        if with_lattice_vectors:
            for v in lattice.unit_cell.lattice_vectors:
                plt.plot([0.0, v[0]], [0.0, v[1]], 'r.')
                plt.arrow(0.0, 0.0, v[0], v[1], color='r', head_width=0.1, length_includes_head=True)

        if with_wigner_seitz:
            points = []
            for n1 in [-1,0,1]:
                for n2 in [-1,0,1]:
                    points.append(n1*lattice.unit_cell.lattice_vectors[0] + \
                                  n2*lattice.unit_cell.lattice_vectors[1])

            points = np.array(points)
            vor = ssp.Voronoi(points)

            # Plot the first non-trivial Voronoi region.
            region_indices = []
            for region in vor.regions:
                if len(region) > 0 and -1 not in region:
                    region_indices = region + [region[0]]
                    
            for ind_r in range(len(region_indices)-1):
                ind1 = region_indices[ind_r]
                ind2 = region_indices[ind_r+1]
                plt.plot([vor.vertices[ind1][0], vor.vertices[ind2][0]], \
                         [vor.vertices[ind1][1], vor.vertices[ind2][1]], 'r-')

        xmin = np.min(xs + [v[0] for v in lattice.unit_cell.lattice_vectors])
        xmax = np.max(xs + [v[0] for v in lattice.unit_cell.lattice_vectors])

        ymin = np.min(ys + [v[1] for v in lattice.unit_cell.lattice_vectors])
        ymax = np.max(ys + [v[1] for v in lattice.unit_cell.lattice_vectors])

        plt.xlim(xmin-1, xmax+1)
        plt.ylim(ymin-1, ymax+1)

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
