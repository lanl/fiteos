""" This module contains classes for evaluating the Vinet equation of state.
"""

class VinetEquation:
    """
    A callable object with equation information.

    Attributes
    ----------
    name : str
    parameter_names : list of str
    y_name : str
    x_name : str
    """

    name = "vinet"
    parameter_names = ["vo", "ko", "kp"]
    y_name = "p"
    x_name = "v"

    plot_names = ["plot_fit", "plot_ff", "plot_ellipses"]

    def __call__(self, params, x, y=None, sigma_x=None, sigma_y=None):
        """
        Computes pressure as a function of volume.

        Parameters
        ----------
        params : dict of float
        x : float
        y : array_like or None, optional
        sigma_x, sigma_y : array_like or None, optional

        Returns
        -------
        p : float
        """

        # interface to lmfit
        v = x
        vo = params['vo']
        ko = params['ko']
        kp = params['kp']

        # compute f
        f = (v / vo)**(1.0 / 3.0)

        # compute pressure
        p = 3.0 * ko * ((1.0 - f) / (f**2.0)) * numpy.exp((3.0 / 2.0) * (kp - 1.0) * (1.0 - f))

        # determine if going to compute residual of pressure and
        # reference pressure data
        if p_ref is None:
            return p
        elif sigma_p is None and sigma_v is None:
            return p_ref - p
        elif sigma_p is None or sigma_v is None:
            raise ValueError("Must specify both sigma_p and sigma_v!")

        # compute weight
        w = 1.0 / ((sigma_p**2.0) + ((sigma_v**2.0) * ((ko / vo)**2.0)))

        return w * (data - p)


