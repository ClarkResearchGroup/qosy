#!/usr/bin/env python
"""
This is a script for finding and plotting
non-interacting BdG Hamiltonians that commute with
two different example of "non-Majorana" zero modes.

These results are described in more detail in Section III of

E. Chertkov, B. Villalonga, and B. K. Clark, "Engineering Topological Models with a General-Purpose Symmetry-to-Hamiltonian Approach," arXiv:1910.10165 (2019).
"""

import numpy as np
import numpy.linalg as nla

import matplotlib.pyplot as plt
from plot_tools import save_my_fig, plot_2d_zero_mode

import qosy as qy
    
threshold = 1e-12

def find_zero_modes(hamiltonian, num_orbitals, num_vecs=None):
    """Find zero modes that are linear combinations of :math:`\\hat{a}_j`
    and :math:`\\hat{b}_j` operators that commute with the given Hamiltonian.
    
    Parameters
    ----------
    hamiltonian : Operator
       The Hamiltonian to commute with.
    num_orbitals : int
       The number of orbitals in the system.
    num_vecs : int, optional
       If not None, then Lanczos with this many vectors
       is used instead of full diagonalization to find
       the zero modes. Default is None.
    """
    
    basis = qy.Basis()
    for orb in range(num_orbitals):
        basis += qy.opstring('A {}'.format(orb))
        basis += qy.opstring('B {}'.format(orb))
        basis += qy.opstring('D {}'.format(orb))

    gen = qy.SymmetricOperatorGenerator(basis)
    gen.add_symmetry(qy.convert(hamiltonian,'Majorana'),num_vecs=num_vecs)
    gen.generate(verbose=False)

    zero_modes = gen.projected_output_operators[-1]
    print(gen.projected_eigenvalues[-1])
    
    return zero_modes

def print_info(hamiltonian, zero_modes, expected_zero_modes):
    """Print the given operators for easy comparison.
    """
    
    print('========================')
    print('Hamiltonian:')
    print(qy.convert(hamiltonian,'Fermion'))
    print('Expected zero modes:')
    qy.print_operators(expected_zero_modes)
    
    print('Found zero modes:')
    qy.print_operators(zero_modes)

def two_site_ham1(ai, bi, aj, bj, i, j):
    # Two-site Hamiltonian that commutes with
    #     \alpha_i a_i + \alpha_j a_j
    # and \beta_i b_i + \beta_j b_j

    # Also works
    """
    coeffs    = [ai*bj+aj*bi,
                 ai*bj-aj*bi,
                 -2.0*aj*bj,
                 -2.0*ai*bi]
    """

    coeffs     = [1.0 + (aj/ai)/(bj/bi),
                  1.0 - (aj/ai)/(bj/bi),
                  -2.0*(aj/ai),
                  -2.0*(bi/bj)]

    op_strings = [qy.opstring('CDag {} C {}'.format(i,j)),
                  qy.opstring('CDag {} CDag {}'.format(i,j)),
                  qy.opstring('CDag {} C {}'.format(i,i)),
                  qy.opstring('CDag {} C {}'.format(j,j))]

    norm = (1.0 + (aj/ai)/(bj/bi))

    result = qy.Operator(np.array(coeffs)/norm, op_strings)
    #print(result)
    
    return result

# Specify the parameters defining the zero modes.
L1 = 30
L2 = 30
xA = np.array([1.0/4.0*L1, 1.0/4.0*L2])
xB = np.array([3.0/4.0*L1, 1.0/4.0*L2])
xC = np.array([3.0/4.0*L1, 3.0/4.0*L2])
xD = np.array([1.0/4.0*L1, 3.0/4.0*L2])
center = np.array([(L1-1.0)/2.0, (L2-1.0)/2.0])
R1  = 0.0
R2  = L2/4.0 #10.0
sigma = L2/10.0 #3.0
t  = 1.0
noise = 1e-3
sigma1 = sigma/1.2
sigma2 = sigma/2.0

# The 2D square lattice vectors.
a1 = np.array([1.0, 0.0])
a2 = np.array([0.0, 1.0])

(Xs, Ys) = np.meshgrid(np.arange(L1), np.arange(L2))

def create_zero_modes(zm_type):
    # Create a double_ring or double_gaussian zero mode.
    #
    # Return the \alpha_i, \beta_j parameters defining
    # the zero mode as well as the chemical potential
    # and pairing parameters defining the Hamiltonian
    # that commutes with these zero modes.
    
    alphas = np.zeros((L1,L2))
    betas  = np.zeros((L1,L2))
    mus    = np.zeros((L1,L2))
    
    for ind1 in range(L1):
        for ind2 in range(L2):
            x = ind1*a1 + ind2*a2
            
            # Double ring Gaussian MZMs
            if zm_type == 'double_ring':
                alphas[ind1,ind2] = np.exp(-((nla.norm(x-center) - R1)/(np.sqrt(2)*sigma1))**2.0) #+ noise*np.random.rand()
                betas[ind1,ind2]  = np.exp(-((nla.norm(x-center) - R2)/(np.sqrt(2)*sigma2))**2.0) #+ noise*np.random.rand()
            
            # Split double-Gaussian MZMs
            elif zm_type == 'double_gaussian':
                alphas[ind1,ind2] = np.exp(-(nla.norm(x-xA)/(np.sqrt(2)*sigma))**2.0) + np.exp(-(nla.norm(x-xC)/(np.sqrt(2)*sigma))**2.0)
                betas[ind1,ind2]  = np.exp(-(nla.norm(x-xB)/(np.sqrt(2)*sigma))**2.0) + np.exp(-(nla.norm(x-xD)/(np.sqrt(2)*sigma))**2.0)
            else:
                raise ValueError('Invalid zero mode type: {}'.format(zm_type))
                
    hamiltonian = qy.Operator([], [], 'Fermion')
    for ind1 in range(L1):
        for ind2 in range(L2):
            orb  = ind2*L1 + ind1
            i    = orb
            ai   = alphas[ind1,ind2]
            bi   = betas[ind1,ind2]
        
            if ind1 < L1-1:
                orb1 = ind2*L1 + (ind1+1)
                j    = orb1
                aj   = alphas[ind1+1,ind2]
                bj   = betas[ind1+1,ind2]
                hamiltonian += two_site_ham1(ai, bi, aj, bj, i, j)
                
            if ind2 < L2-1:
                orb2 = (ind2+1)*L1 + ind1
                j    = orb2
                aj   = alphas[ind1,ind2+1]
                bj   = betas[ind1,ind2+1]
                hamiltonian += two_site_ham1(ai, bi, aj, bj, i, j)
                
    mus = np.zeros((L1,L2))
    D1  = np.zeros((L1,L2))
    D2  = np.zeros((L1,L2))
    for (coeff, os) in hamiltonian:
        if len(os.orbital_operators) == 2:
            if os.orbital_operators[0] == 'CDag' and os.orbital_operators[1] == 'CDag':
                i = os.orbital_labels[0]
                j = os.orbital_labels[1]
                
                x1 = i % L1
                y1 = i // L1
                
                x2 = j % L1
                y2 = j // L1
                
                #print('(x1,y1)=({},{}), (x2,y2)=({},{})'.format(x1,y1,x2,y2))
            
                if x2 == x1+1 and y2 == y1:
                    D1[x1,y1] += np.real(coeff)
                elif x2 == x1 and y2 == y1+1:
                    D2[x1,y1] += np.real(coeff)
                else:
                    raise ValueError('Invalid term in Hamiltonian: {}'.format(os))
            elif os.orbital_operators[0] == 'CDag' and os.orbital_operators[1] == 'C':
                i = os.orbital_labels[0]
                j = os.orbital_labels[1]
                x1 = i % L1
                y1 = i // L1
            
                if i == j:
                    mus[x1,y1] += np.real(coeff)
            else:
                raise ValueError('Invalid term in Hamiltonian: {}'.format(os))

    Dabs = np.sqrt(np.abs(D1)**2.0 + np.abs(D2)**2.0)
    
    Dangle = np.arctan2(D2, D1)/np.pi

    return (alphas, betas, mus, D1, D2, Dabs, Dangle)
    
"""
# This is slow, but can be uncommented out and ran to
# verify that the Hamiltonian was built correctly.

# Check visually that the zero modes are correct
zero_modes = find_zero_modes(hamiltonian, L1*L2, num_vecs=4)
plot_2d_zero_mode(zero_modes[0], L1, L2)
plt.suptitle('Real zero mode 1')
plot_2d_zero_mode(zero_modes[1], L1, L2)
plt.suptitle('Real zero mode 2')
"""

# Print out the Hamiltonian.
#print(':::::::::::::::::')
#print('Test Hamiltonian:')
#qy.print_operators([hamiltonian], convert_to='Fermion')
#print('Zero modes:')
#qy.print_operators(zero_modes, norm_order=np.inf)

# Plot the zero modes, their Hamiltonians, and the save the plots to files.
for zm_type in ['double_ring', 'double_gaussian']:
    (alphas, betas, mus, D1, D2, Dabs, Dangle) = create_zero_modes(zm_type)

    Dmax = np.max(Dabs)
    print('For zero mode {}, the maximum pairing Dmax = {}'.format(zm_type, Dmax))

    plot_2d_zero_mode((alphas,betas), L1, L2, origin='lower')
    save_my_fig(zm_type+'_zero_modes_2d.pdf')

    plt.figure()

    plt.imshow(mus, origin='lower', cmap=plt.get_cmap('afmhot'))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xticks([0, L1//2, L1])
    plt.yticks([0, L2//2, L2])
    
    plt.colorbar()
    if zm_type == 'double_gaussian':
        plt.clim(-3.0, -6.0)
    elif zm_type == 'double_ring':
        plt.clim(-3.0, -23.0)
    plt.title('$\\mu_{\\mathbf{x}}/t$')

    save_my_fig(zm_type+'_zero_modes_chemical_potentials_2d.pdf')
    
    plt.figure()

    plt.quiver(Xs, Ys, D1, D2, Dangle,cmap=plt.get_cmap('hsv'))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.xticks([0, L1//2, L1])
    plt.yticks([0, L2//2, L2])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    
    #plt.colorbar()
    plt.clim(-1.0, 1.0)

    # Background heatmap showing the amplitude
    #plt.imshow(Dabs, cmap=plt.get_cmap('Oranges'), alpha=0.4)
    #plt.colorbar()

    plt.title('$\\Delta_{\\mathbf{x},\\mathbf{x}+\\hat{\\mathbf{\delta}}}/t$')

    save_my_fig(zm_type+'_zero_modes_pairing_2d.pdf')

plt.show()

