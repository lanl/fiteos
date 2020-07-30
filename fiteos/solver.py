""" This module contains classes that manage the optimization.
"""

import lmfit
import numpy
from fiteos import plots
from scipy.stats import t

class Solver:
    """
    Object with optimization information.

    Attributes
    ----------
    x, y, sigma_x, sigma_y : array_like
    equation : fiteos.equation.Equation
    parameters : lmfit.Parameters
    out : lmfit.minimizer.MinimizerResult
    """

    def __init__(self, x, y, sigma_y, sigma_x, equation):
        """
        Initializes class.

        Parameters
        ----------
        x, y, sigma_x, sigma_y : array_like
        equation : fiteos.equation.Equation
        """
        self.y = y
        self.x = x
        self.sigma_y = sigma_y
        self.sigma_x = sigma_x
        self.equation = equation
        self.parameters = lmfit.Parameters()
        self.out = None

    def set_parameter(self, name, value, low=None, high=None, vary=True):
        """
        Sets parameter for optimization.

        Parameters
        ----------
        name : str
        value : float
        low : float, optional
        high : float, optional
        vary : bool, optional
        """
        self.parameters.add(name, value=value, min=low, max=high, vary=vary)

    def minimize(self, method="Nedler"):
        """
        Execute optimization.

        Parameters
        ----------
        method : str

        Returns
        -------
        out : lmfit.minimizer.MinimizerResult
        """
        params = self.out.params if self.out else None
        self.minimizer = lmfit.Minimizer(
            self.equation, self.parameters, fcn_args=(self.x,),
            fcn_kws={"y": self.y, "sigma_y": self.sigma_y, "sigma_x" : self.sigma_x})
        self.out = self.minimizer.minimize(method=method, params=params)
        return self.out

    def confidence_interval(self, out):
        """
        Sets parameter for optimization.

        Parameters
        ----------
        name : str
        value : float
        low : float, optional
        high : float, optional
        vary : bool, optional

        Returns
        -------
        out
        """
        return lmfit.conf_interval(self.minimizer, out)

    def fit(self, out, n=1000):
        """ Evaluates the equation for a set of values.

        Parameters
        ----------
        out : lmfit.minimizer.MinimizerResult
        n : int

        Returns
        -------
        x_fit, y_fit : array_like
        """
        x_fit = numpy.linspace(numpy.min(self.x) - 10,
                               numpy.max(self.x) + 10,
                               n)
        y_fit = self.equation(out.params, x_fit)
        return y_fit, x_fit

    def confidence_prediction_bands(self, x_fit, results, confidence_interval, pcov):
        """
        Computes confidence prediction bands.

        Parameters
        ----------
        x_fit : array_like
        results : int
        confidence_interaval : int
        pcov : numpy.array

        Returns
        -------
        cp_band_0, cp_band_1 : array_like
        """
        param_names = []
        param_values = []
        param_deltas = []
        for pname in self.minimizer.params.keys():
            if self.minimizer.params[pname].vary:
                param_names.append(pname)
                param_values.append(results[pname])
                param_deltas.append(1e-5 * results[pname])

        x_m_0s = numpy.empty_like(x_fit)
        f_m_0s = numpy.empty_like(x_fit)
        for i, xx in enumerate(x_fit):
            x_m_0s[i] = x_fit[i]
            f_m_0s[i] = self.equation(results, xx)
        
        diag_delta = numpy.diag(param_deltas)
        dxdbeta = numpy.empty([len(param_values), len(x_fit)])
        
        for i, value in enumerate(param_values):
        
            adj_param_values = param_values + diag_delta[i]

            for j, pname in enumerate(param_names):
                results[pname] = adj_param_values[j]
        
            for j, x_m_0 in enumerate(x_m_0s):
                dxdbeta[i][j] = (self.equation(results, x_m_0) - f_m_0s[j]) / diag_delta[i][i]
        
        variance = numpy.empty(len(x_fit))
        for i, gprime in enumerate(dxdbeta.T):
            variance[i] = gprime.T.dot(pcov).dot(gprime)
        
        critical_value = t.isf(0.5 * (confidence_interval + 1.0), len(param_names))
        
        confidence_half_widths = critical_value * numpy.sqrt(variance)
        
        cp_band_0 = f_m_0s - confidence_half_widths
        cp_band_1 = f_m_0s + confidence_half_widths

        return cp_band_0, cp_band_1

    def plot(self, **kwargs):
        """
        Executes equation-specific plots.

        Parameters
        ----------
        **kwargs : dict
        """
        for i, plot_name in enumerate(self.equation.plot_names):
            plot_fcn = getattr(self, plot_name)
            plot_fcn(i, **kwargs)

    def plot_fit(self, fig=0, **kwargs):
        """
        Plots data and fit of equation with confidence bands.

        Parameters
        ----------
        fig : int
        **kwargs : dict
        """
        confidence_interval = kwargs["confidence_interval"]
        output_file = kwargs["outpuy_fit_file"] if "outpuy_fit_file" in kwargs else None
        pcov = self.out.covar
        y_fit, x_fit = self.fit(self.out)
        results = self.out.params.valuesdict()
        cp_band_low, cp_band_high = self.confidence_prediction_bands(x_fit, results, confidence_interval, pcov)
        plots.plot_fit(fig, y_fit, x_fit, self.y, self.x, self.sigma_y, self.sigma_x,
                       cp_band_low, cp_band_high, output_file=output_file)

    def plot_ff(self, fig=1, **kwargs):
        """
        Plots strain plot.

        Parameters
        ----------
        fig : int
        **kwargs : dict
        """
        output_file = kwargs["output_ff_file"] if "output_ff_file" in kwargs else None
        pcov = self.out.covar
        y0 = self.out.params.valuesdict()["vo"]
        sigma_x0 = self.out.params["vo"].stderr
        plots.plot_ff(1, self.y, self.x, self.sigma_y, self.sigma_x, y0, sigma_x0, pcov, output_file=output_file)

    def plot_ellipses(self, fig=2, **kwargs):
        """
        Plots confidence ellipses.

        Parameters
        ----------
        fig : int
        **kwargs : dict
        """
        output_file = kwargs["output_ellipses_file"] if "output_ellipses_file" in kwargs else None
        pcov = self.out.covar
        results_list = [self.out.params.valuesdict()[pname] for pname in self.equation.parameter_names]
        plots.plot_ellipses(fig, results_list, pcov, output_file=output_file)

