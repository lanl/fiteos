""" This module contains classes for evaluating the isothermal second-order
Burch-Murnaghan equation of state.
"""

class BM2Equation:
    """
    A callable object with equation information.

    Attributes
    ----------
    name : str
    parameter_names : list of str
    y_name : str
    x_name : str
    """

    name = "bm2"
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
        p_ref = y
        sigma_p = sigma_x
        sigma_v = sigma_y
        vo = params["vo"]
        ko = params["ko"]

        # compute pressure
        p = 3.0 * (ko / 2.0) * (((vo/v)**(7.0 / 3.0)) - ((vo/v)**(5.0 / 3.0)))

        # determine if going to compute residual of pressure and
        # reference pressure data 
        if p_ref is None:
            return p
        if sigma_x is None and sigma_y is None:
            return (data-p)

        # compute weight
        w = 1.0 / ((sigma_p**2.0) + ((sigma_v**2.0) * ((ko / vo)**2.0)))

        return w * (p_ref - p)
