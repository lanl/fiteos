""" This module contains functions for plotting confidence ellipses.
"""

import matplotlib.pyplot as plt
import numpy
from matplotlib import lines
from matplotlib import patches

def plot_ellipses(nfig, results, pcov, output_file=None):
    """ Plots fit.

        Parameters
        ----------
        nfig : int
        results : dict of float
        pcov : numpy.array
        output_file : str or None, optional
    """

    n_params = len(pcov[0])
    
    fig, axs = plt.subplots(n_params - 1, n_params - 1)
    axs = [[axs,],] if n_params == 2 else axs

    nstd = [3.0, 2.0, 1.0]
    
    err = [numpy.sqrt(pcov[0, 0]) for i in range(n_params)]
    
    err = numpy.outer(err, err)
    
    popt = results
    
    for i in range(n_params - 1):
    
        for j in range(i + 1, n_params):
            indices = numpy.array([i, j])
            projected_cov = (pcov)[indices[:, None], indices]
    
            scaled_pos = numpy.array([popt[i], popt[j]])
    
            cov = projected_cov
            pos = scaled_pos
    
            ellipse = plt_ellipse(cov, pos, nstd, ax=axs[j - 1][i])
    
            maxx = 1.5 * 2.2 * numpy.sqrt(projected_cov[0][0])
            maxy = 1.5 * 2.2 * numpy.sqrt(projected_cov[1][1])
            axs[j - 1][i].set_xlim(scaled_pos[0] - maxx, scaled_pos[0] + maxx)
            axs[j - 1][i].set_ylim(scaled_pos[1] - maxy, scaled_pos[1] + maxy)
            axs[j - 1][i].yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    
            if i == 0 and j == 1:
                axs[j - 1][i].errorbar(pos[0], pos[1], xerr=numpy.sqrt(err[0, 0]), yerr=numpy.sqrt(err[1,1]),
                                            linestyle="None", marker="None", color="black", capsize = 3, elinewidth=0.5)
                #print("V0 [A^3]: %s +/- %s" % (pos[0],numpy.sqrt(err[0, 0])))
                #print("K0 [GPa]: %s +/- %s" % (pos[1],numpy.sqrt(err[1, 1])))
            if i == 0 and j == 2:
                axs[j-1][i].errorbar(pos[0], pos[1], xerr=numpy.sqrt(err[0, 0]), yerr=numpy.sqrt(err[2, 2]),
                                          linestyle="None", marker="None", color="black", capsize = 3, elinewidth=0.5)
            if i == 1 and j == 2:
                axs[j-1][i].errorbar(pos[0], pos[1], xerr=numpy.sqrt(err[1,1]), yerr=numpy.sqrt(err[2, 2]),
                                          linestyle="None", marker="None", color="black", capsize = 3., elinewidth=0.5)
                #print("K0_prime: %s +/- %s" % (pos[1], numpy.sqrt(err[2, 2])))
    
    red_patch = patches.Patch(color="red", alpha=0.8, label="3$\sigma$")
    cyan_patch = patches.Patch(color="cyan", alpha=0.8, label="2$\sigma$")
    blue_patch = patches.Patch(color="blue", alpha=0.8, label="1$\sigma$")
    us_bm3 = lines.Line2D([], [], linestyle="none", marker="+", color="black")

    if n_params == 3:
        axs[0][1].set_axis_off()
        legend = axs[0][1].legend(handles=[red_patch, cyan_patch, blue_patch, us_bm3], loc="center")
    else:
        legend = axs[0][0].legend(handles=[red_patch, cyan_patch, blue_patch, us_bm3], loc="center")
    legend.get_frame().set_linewidth(0.0)
    param_names = ["$\mathbf{V_0}$", "$\mathbf{K_0}$", "$\mathbf{K'}$\'"]
    param_units = ["$\mathbf{\AA^3}$", "GPa", ""]
    if param_names != []:
        for i in range(n_params - 1):
            axs[n_params - 2][i].set_xlabel("{0:s} ({1:s})".format(param_names[i], param_units[i]), fontweight="bold")
        for j in range(1, n_params):
            if param_units[j] == param_units[2]:
                axs[j - 1][0].set_ylabel("{0:s}".format(param_names[j]), fontweight="bold")
            else:
                axs[j - 1][0].set_ylabel("{0:s} ({1:s})".format(param_names[j], param_units[j]), fontweight="bold")
            #axs[j - 1][0].yaxis.set_label_coords(-0.25, 0.5)

    if output_file:
        plt.savefig(output_file, dpi=1800, bbox_inches="tight")
        plt.close()

def plt_ellipse(cov, pos, nstdl, ax=None):
    def eigsorted(cov):
        vals, vecs = numpy.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = numpy.degrees(numpy.arctan2(*vecs[:,0][::-1]))
    palet = ['red', 'cyan', 'blue']

    for i in range(3):
        nstd = nstdl[i]
        hue = palet[i]
        width, height = 2 * nstd * numpy.sqrt(vals)
        ellip = patches.Ellipse(xy=pos, width=width, height=height, angle=theta, color=hue, alpha=0.8)
        ax.add_artist(ellip)
        ax.tick_params(direction="in", bottom=1, top=1, left=1, right=1)

    return ellip
