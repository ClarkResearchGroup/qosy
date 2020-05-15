#!/usr/bin/env python
"""
This module has a few helper functions for 
manipulating and plotting MZM BdG Hamiltonians.
"""

import numpy as np
import numpy.linalg as nla

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

params = {
    'axes.grid': False,
    'font.size': 36, 
#    'font.family': 'serif',
    'lines.linewidth' : 3,
    'axes.linewidth' : 2,
    'image.interpolation' : 'none'
}
mpl.rcParams.update(params)

import qosy as qy

def save_my_fig(filename):
    """Save the current matplotlib figure to a
    file with the given name.
    """
    
    fig = plt.gcf()
    fig.set_size_inches(12,8)
    ax  = plt.gca()
    ax.tick_params(width=2)

    # If you want to remove the tick labels
    #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #plt.savefig(filename, bbox_inches=extent, pad_inches=0)
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

def read_zero_mode(zero_mode, L1, L2):
    """Read the \alpha_i, \beta_j parameters from the 2D zero mode Operator.
    """
    
    if isinstance(zero_mode, qy.Operator):
        alphas = np.zeros((L1,L2))
        betas  = np.zeros((L1,L2))

        for (coeff, os) in zero_mode:
            if len(os.orbital_operators) == 1:
                orb = os.orbital_labels[0]
                ind1 = orb % L1
                ind2 = orb // L1
            
                if os.orbital_operators[0] == 'A':
                    alphas[ind1,ind2] = coeff
                elif os.orbital_operators[0] == 'B':
                    betas[ind1,ind2]  = coeff
                else:
                    raise ValueError('Invalid zero mode operator string: {}'.format(os))
            else:
                raise ValueError('Invalid zero mode operator string: {}'.format(os))
    else:
        alphas = zero_mode[0]
        betas  = zero_mode[1]

    return (alphas, betas)

def plot_2d_zero_mode(zero_mode, L1, L2, origin=None):
    """Plot a 2D zero mode Operator as a heatmap. 
    The \alpha_i parameters are plotted blue and the 
    \beta_j parameters are plotted green.
    """
    
    (alphas, betas) = read_zero_mode(zero_mode, L1, L2)

    if origin != 'lower':
        (L1, L2) = (L2, L1)
        
    fig, ax  = plt.subplots(constrained_layout=True)
    divider = make_axes_locatable(ax)
    lax = divider.append_axes('right', size='5%', pad='5%')
    rax = divider.append_axes('right', size='5%', pad='8%')
    
    alphas_normalized = alphas/nla.norm(alphas)
    betas_normalized  = betas/nla.norm(betas)

    min_val = np.minimum(alphas_normalized.min(), betas_normalized.min())
    max_val = np.maximum(alphas_normalized.max(), betas_normalized.max())

    max_abs_val = np.maximum(np.abs(min_val), np.abs(max_val))
    print('min_val     = {}'.format(min_val))
    print('max_val     = {}'.format(max_val))
    print('max_abs_val = {}'.format(max_abs_val))
    
    cmap_a   = plt.get_cmap('Blues')
    norm_a   = mpl.colors.Normalize(0.0, max_abs_val, clip=True)
    colors_a = norm_a(alphas_normalized)
    colors_a = cmap_a(colors_a)
    colors_a[..., -1] = alphas/max_abs_val
    
    cmap_b   = plt.get_cmap('Greens')
    norm_b   = mpl.colors.Normalize(0.0, max_abs_val, clip=True)
    colors_b = norm_b(betas_normalized)
    colors_b = cmap_b(colors_b)
    colors_b[..., -1] = betas/max_abs_val

    pa = ax.imshow(colors_a, origin=origin, vmin=min_val, vmax=max_val)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #ax.xticks([])
        
    cba = mpl.colorbar.ColorbarBase(lax, orientation='vertical', cmap=cmap_a, norm=norm_a)
    cba.ax.set_title('$\\alpha_{\\mathbf{x}}$')
    lax.yaxis.set_ticks_position('right')
    
    pb = ax.imshow(colors_b, origin=origin, vmin=min_val, vmax=max_val)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    #ax.yticks([])
    #plt.yticks([0,L2//2,L2])

    cbb = mpl.colorbar.ColorbarBase(rax, orientation='vertical', cmap=cmap_b, norm=norm_b)
    cbb.ax.set_title('$\\beta_{\\mathbf{x}}$')
    rax.yaxis.set_ticks_position('right')
    #rax.ticklabel_format(axis='y', style='sci', useOffset=True, scilimits=(-2,2))
    
    plt.setp(lax.get_yticklabels(), visible=False)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    if origin == 'lower':
        ax.set_xlim([0,L1])
        ax.set_ylim([0,L2])
    else:
        ax.set_xlim([0,L1])
        ax.set_ylim([L2,0])

    ax.set_xticks([0,L1//2,L1])
    ax.set_yticks([0,L2//2,L2])
