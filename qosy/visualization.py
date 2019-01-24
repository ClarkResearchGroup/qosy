#!/usr/bin/env python
"""
This module provides some tools for visualizing
a lattice and interactions on a lattice. Also,
printing tools are provided.
"""

import warnings
import numpy as np
import numpy.linalg as nla
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
        
def print_vectors(basis, vectors, lattice=None, keywords=None, convert_to=None, norm_order=None, tol=1e-10):
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
    keywords : list of str, optional
        If provided (along with lattice), only prints
        OperatorStrings whose string representation
        include the given keywords. Defaults to None.
    convert_to : str, optional
        If not None, convert OperatorStrings to the given type
        before printing. Defaults to None.
    norm_order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Normalizes the vector to have a norm of this order 
        (see `numpy.norm`). Defaults to None, which is the 
        :math:`\\ell_2`-norm. Another useful norm is the `inf` 
        order norm.
    tol : float, optional
        The tolerance within which to consider zero (and
        not print them). Defaults to 1e-10.
    """
    
    if len(basis) != int(vectors.shape[0]):
        raise ValueError('Vectors are not of the right size {} to be in the basis of dimension {}'.format(vectors.shape, len(basis)))

    num_vecs = int(vectors.shape[1])
    
    operators = [Operator(vectors[:,ind_vec], basis.op_strings) for ind_vec in range(num_vecs)]

    if convert_to is not None:
        operators = [convert(op, convert_to) for op in operators]
    
    ind_vec = 1
    for operator in operators:
        cleaned_operator = operator.remove_zeros(tol=tol)
        cleaned_operator.normalize(order=norm_order)
        print('vector {} = '.format(ind_vec))

        output_string = str(cleaned_operator)
        if lattice is not None:
            output_string = relabel_orbitals(output_string, lattice)

            # Filter out all the OperatorStrings
            # that contain the given keywords.
            if keywords is not None:
                new_output_string = ''
                for line in output_string.split('\n'):
                    for keyword in keywords:
                        if keyword in line:
                            new_output_string += line+'\n'
                            break

                output_string = new_output_string
            
        print(output_string)
        
        ind_vec += 1

def print_operators(operators, lattice=None, keywords=None, convert_to=None, norm_order=None, tol=1e-10):
    """Print in human-readable form a list of Operators.
    
    Parameters
    ----------
    operators : list of Operators
        The Operators to print.
    lattice : Lattice, optional
        If provided, convert orbital labels to 
        lattice coordinate labels. Defaults to None.
    keywords : list of str, optional
        If provided (along with lattice), only prints
        OperatorStrings whose string representation
        include the given keywords. Defaults to None.
    convert_to : str, optional
        If not None, convert OperatorStrings to the given type
        before printing. Defaults to None.
    norm_order : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Normalizes the vector to have a norm of this order 
        (see `numpy.norm`). Defaults to None, which is the 
        :math:`\\ell_2`-norm. Another useful norm is the `inf` 
        order norm.
    tol : float, optional
        The tolerance within which to consider zero (and
        not print them). Defaults to 1e-10.
    """
    
    if convert_to is not None:
        operators_to_print = [convert(op, convert_to) for op in operators]
    else:
        operators_to_print = operators
    
    ind_vec = 1
    for operator in operators_to_print:
        cleaned_operator = operator.remove_zeros(tol=tol)
        cleaned_operator.normalize(order=norm_order)

        print('operator {} = '.format(ind_vec))

        output_string = str(cleaned_operator)
        if lattice is not None:
            output_string = relabel_orbitals(output_string, lattice)

            # Filter out all the OperatorStrings
            # that contain the given keywords.
            if keywords is not None:
                new_output_string = ''
                for line in output_string.split('\n'):
                    for keyword in keywords:
                        if keyword in line:
                            new_output_string += line+'\n'
                            break

                output_string = new_output_string
            
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
                plt.plot([0.0, v[0]], [0.0, 0.0], 'r.')
                plt.arrow(0.0, 0.0, v[0], 0.0, color='r', head_width=0.1, length_includes_head=True)

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

# TODO: document
def plot_opstring(op_string, lattice, distance_cutoff=None, weight=1.0, marker_size=20):
    
    
    if op_string.op_type == 'Pauli':
        markercolors = {'X':'r', 'Y':'g', 'Z':'b'}
    elif op_string.op_type == 'Majorana':
        markercolors = {'A':'r', 'B':'g', 'D':'b'}
    elif op_string.op_type == 'Fermion':
        markercolors = {'CDag':'r', 'C':'b'}
        
    max_width = 3.0
    
    if np.imag(weight) > 1e-10:
        warnings.warn('Cannot plot an operator string with an imaginary coefficient. Taking real part.')
    
    weight = np.real(weight)

    # 1D or 2D plots
    if lattice.dim == 1 or lattice.dim == 2:
        if len(op_string.orbital_operators) == 1:
            orb_op    = op_string.orbital_operators[0]
            orb_label = op_string.orbital_labels[0]
            (pos, orb_name, cell_coord) = lattice._orbitals[orb_label]

            if lattice.dim == 1:
                pos = np.array([pos[0], 0.0])
        
            plt.plot([pos[0]], [pos[1]], color=markercolors[orb_op], markeredgecolor=markercolors[orb_op], alpha=0.5*np.abs(weight), markersize=marker_size)

            if orb_name != '':
                plt.annotate(orb_name, xy=(pos[0],pos[1]), color='k')

        elif len(op_string.orbital_operators) == 2:
            orb_op1    = op_string.orbital_operators[0]
            orb_label1 = op_string.orbital_labels[0]
            (pos1, orb_name1, cell_coord1) = lattice._orbitals[orb_label1]

            orb_op2    = op_string.orbital_operators[1]
            orb_label2 = op_string.orbital_labels[1]
            (pos2, orb_name2, cell_coord2) = lattice._orbitals[orb_label2]

            if lattice.dim == 1:
                pos1 = np.array([pos1[0], 0.0])
                pos2 = np.array([pos2[0], 0.0])

            # TODO: finish
            if distance_cutoff is not None:
                if nla.norm(pos1-pos2) > distance_cutoff:
                    return

            # TODO: flag to plot only bonds involving a particular site

            plt.plot([pos1[0]], [pos1[1]], color=markercolors[orb_op1], markeredgecolor=markercolors[orb_op1], alpha=0.5*np.abs(weight), markersize=marker_size)
            if orb_name1 != '':
                plt.annotate(orb_name1, xy=(pos1[0],pos1[1]), color='k')
            
            plt.plot([pos2[0]], [pos2[1]], color=markercolors[orb_op1], markeredgecolor=markercolors[orb_op2], alpha=0.5*np.abs(weight), markersize=marker_size)
            if orb_name2 != '':
                plt.annotate(orb_name2, xy=(pos2[0],pos2[1]), color='k')

            if weight > 0.0:
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'b-', linewidth=max_width*np.abs(weight), alpha=0.5*np.abs(weight))
            else:
                plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 'r-', linewidth=max_width*np.abs(weight), alpha=0.5*np.abs(weight))
        else:
            raise NotImplementedError('No supported visualization for operators on more than two sites yet.')
    # 3D plot
    elif lattice.dim == 3:
        fig = plt.gcf()
        ax  = plt.gca()
        if ax.name != '3d':
            ax = fig.add_subplot(111, projection='3d')
        
        if len(op_string.orbital_operators) == 1:
            orb_op    = op_string.orbital_operators[0]
            orb_label = op_string.orbital_labels[0]
            (pos, orb_name, cell_coord) = lattice._orbitals[orb_label]
        
            ax.scatter([pos[0]], [pos[1]], zs=[pos[2]], c=markercolors[orb_op], alpha=0.5*np.abs(weight), s=marker_size)

            if orb_name != '':
                ax.text(pos[0], pos[1], pos[2], orb_name, size=20, zorder=1)

        elif len(op_string.orbital_operators) == 2:
            orb_op1    = op_string.orbital_operators[0]
            orb_label1 = op_string.orbital_labels[0]
            (pos1, orb_name1, cell_coord1) = lattice._orbitals[orb_label1]

            orb_op2    = op_string.orbital_operators[1]
            orb_label2 = op_string.orbital_labels[1]
            (pos2, orb_name2, cell_coord2) = lattice._orbitals[orb_label2]

            if distance_cutoff is not None:
                if nla.norm(pos1-pos2) > distance_cutoff:
                    return
            
            ax.scatter([pos1[0]], [pos1[1]], zs=[pos1[2]], marker='o', c=markercolors[orb_op1], alpha=0.5*np.abs(weight), s=marker_size)
            if orb_name1 != '':
                ax.text(pos1[0], pos1[1], pos1[2], orb_name1, size=20, zorder=1)

            ax.scatter([pos2[0]], [pos2[1]], zs=[pos2[2]], marker='o', c=markercolors[orb_op1], alpha=0.5*np.abs(weight), s=marker_size)
            if orb_name2 != '':
                ax.text(pos2[0], pos2[1], pos2[2], orb_name2, size=20, zorder=1)

            if weight > 0.0:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], zs=[pos1[2], pos2[2]], color='b', linewidth=max_width*np.abs(weight), alpha=0.5*np.abs(weight))
            else:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], zs=[pos1[2], pos2[2]], color='r', linewidth=max_width*np.abs(weight), alpha=0.5*np.abs(weight))
        else:
            raise NotImplementedError('Not finished yet.')  
    else:
        raise ValueError('Cannot plot lattice of dimension {}'.format(lattice.dim))
            
def plot_operator(operator, lattice, distance_cutoff=None, marker_size=20):
    """Plot a visual representation of the
    Operator on a Lattice.

    Parameters
    ----------
    operator : Operator
        The Operator to plot.
    lattice : Lattice
        The lattice to plot the Operator on.
  
    Examples
    --------
    To visualized the operator with the lattice in the background:
       >>> qosy.plot(lattice)
       >>> qosy.plot(operator, lattice)
       >>> qosy.show()
    """

    max_coeff = np.max(np.abs(operator.coeffs))
    
    for (coeff, op_string) in operator:
        plot_opstring(op_string, lattice, distance_cutoff=distance_cutoff, weight=(coeff/max_coeff), marker_size=marker_size)
    
def show():
    """Display a figure.
    """

    plt.show()
