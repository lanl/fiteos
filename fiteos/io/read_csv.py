""" This module contains functions for reading CSV files containing pressure
and volume data.
"""

import numpy

def read_csv(input_file, delimiter=",", x="p", sigma_x="sigma_p",
                  y="v", sigma_y="sigma_v"):
    """
    Reads a CSV file with x and y data and standard deviation columns.

    Parameters
    ----------
    input_file : str
    delimiter : str
    y : str
    x : str
    sigma_y : str
    sigma_x : str

    Returns
    -------
    x, y, sigma_x, sigma_y : tuple of array_like
    """
    data = numpy.genfromtxt(input_file, delimiter=delimiter, names=True)
    return (data[x].flatten(), data[sigma_x].flatten(),
            data[y].flatten(), data[sigma_y].flatten())
