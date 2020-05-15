#!/usr/bin/env python
"""
This is a script for finding and plotting a
non-interacting BdG Hamiltonian that commutes with
two Majorana zero modes (MZMs) shaped according to 
an image of Ettore Majorana.

By changing the input image, one can make Hamiltonians
that commute with MZMs distributed according to a different
desired shape.

These results are described in more detail in Section III of

E. Chertkov, B. Villalonga, and B. K. Clark, "Engineering Topological Models with a General-Purpose Symmetry-to-Hamiltonian Approach," arXiv:1910.10165 (2019).

The citation for the image of Ettore Majorana is

Unknown author (Mondadori Publishers), Ettore Majorana, 1930s. Accessed July 2, 2019 from: http://commons.wikimedia.org 

Note
----
To run this script, you need to use python2 because of `scipy.misc`.
Otherwise, you can modify that line to read in an image file with
a different package.
"""

import numpy as np
import numpy.linalg as nla
import scipy.misc as sm

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

# Read in the grayscale image file (can use other package if necessary).
image = sm.imread('188px-Ettore_Majorana.jpg').astype(float)

# Rescale the image pixels
image /= np.max(image)
image = 1.0 - image

# The area over which the zero modes are defined
# is twice as tall and wide as the original image.
L1 = 2*image.shape[0]
L2 = 2*image.shape[1]
print('L1 = {}, L2 = {}'.format(L1,L2))

#plt.imshow(image, cmap=plt.get_cmap('Greens'))
#plt.colorbar()
#plt.show()

t  = 1.0

(Xs, Ys) = np.meshgrid(np.arange(L1), np.arange(L2))

# Define the $\alpha_j$ and $\beta_j$ of the zero modes.
alphas = 1e-5*np.ones((L1,L2))
betas  = 1e-5*np.ones((L1,L2))

# The zero modes are made from the image put
# into the bottom left and top right corners
# of a square lattice.
for ind1 in range(L1//2):
    for ind2 in range(L2//2):
        alphas[ind1,ind2]             += image[ind1,ind2]
        betas[L1//2+ind1, L2//2+ind2] += image[ind1, ind2]


# Build the Hamiltonian from the two-site
# bond operator Hamiltonian and keep
# track of the coefficients.
coeffs_dict = dict()

print('Building Hamiltonian.')
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
            h_ij = two_site_ham1(ai, bi, aj, bj, i, j)

            for (coeff, os) in h_ij:
                if os in coeffs_dict:
                    coeffs_dict[os] += coeff
                else:
                    coeffs_dict[os] = coeff
            
            #hamiltonian += h_ij
            
        if ind2 < L2-1:
            orb2 = (ind2+1)*L1 + ind1
            j    = orb2
            aj   = alphas[ind1,ind2+1]
            bj   = betas[ind1,ind2+1]
            h_ij = two_site_ham1(ai, bi, aj, bj, i, j)

            for (coeff, os) in h_ij:
                if os in coeffs_dict:
                    coeffs_dict[os] += coeff
                else:
                    coeffs_dict[os] = coeff
            
            #hamiltonian += h_ij

coeffs     = []
op_strings = []
for os in coeffs_dict:
    coeffs.append(coeffs_dict[os])
    op_strings.append(os)

hamiltonian = qy.Operator(coeffs, op_strings, 'Fermion')
print('Built Hamiltonian.')

"""
# This is slow, but can be uncommented out and ran to
# verify that the Hamiltonian was built correctly.

# Check that the Hamiltonian has a (pair of) zero modes.
print('Diagonalizing Hamiltonian.')
(bdg_energies, U, V) = qy.diagonalize_bdg(hamiltonian, L1*L2, sigma=0.0, num_vecs=4)
print('Diagonalized Hamiltonian.')

print('bdg_energies = {}'.format(bdg_energies))
inds_zm = np.where(np.abs(bdg_energies) < 1e-12)[0]
num_zm  = len(inds_zm)
print('Number of zero modes: {}'.format(num_zm))
"""

# Store the chemical potential and pairing information
# from the Hamiltonian in separate arrays.
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

# The magnitude of the pairing.
Dabs = np.sqrt(np.abs(D1)**2.0 + np.abs(D2)**2.0)
Dabs /= np.max(Dabs)

# The phase angle of the pairing.
Dangle = np.arctan2(D2, D1)/np.pi

"""
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

# Plot the zero modes.
plot_2d_zero_mode((alphas, betas), L1, L2, origin='upper')

# Save the image to a file.
save_my_fig('image_zero_modes_2d.pdf')

plt.figure()

# Plot the chemical potential.
plt.imshow(mus+4.0, cmap=plt.get_cmap('PuOr'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xticks([0,L2//2,L2])
plt.yticks([0,L1//2,L1])

plt.colorbar()
plt.clim(-4.0, 4.0)
plt.title('$\\mu_{\\mathbf{x}}/t+4$')

# Save the image to a file.
save_my_fig('image_zero_modes_chemical_potentials_2d.pdf')

plt.figure()

# Plot the pairing in the x-direction.
plt.imshow(D1, cmap=plt.get_cmap('bwr'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xticks([0,L2//2,L2])
plt.yticks([0,L1//2,L1])

plt.colorbar()
plt.clim(-1.0, 1.0)
plt.title('$\\Delta_{\\mathbf{x},\\mathbf{x}+\\hat{\\mathbf{x}}}/t$')

# Save the image to a file.
save_my_fig('image_zero_modes_D1_2d.pdf')

plt.figure()

# Plot the pairing in the y-direction.
plt.imshow(D2, cmap=plt.get_cmap('bwr'))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.xticks([0,L2//2,L2])
plt.yticks([0,L1//2,L1])

plt.colorbar()
plt.clim(-1.0, 1.0)
plt.title('$\\Delta_{\\mathbf{x},\\mathbf{x}+\\hat{\\mathbf{y}}}/t$')

# Save the image to a file.
save_my_fig('image_zero_modes_D2_2d.pdf')

plt.show()

