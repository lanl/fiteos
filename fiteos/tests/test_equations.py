""" Tests for equations.
"""

import itertools
import numpy
import os
import unittest
from fiteos import equations
from fiteos import io
from fiteos import solver

class TestBM3(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBM3, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bm3(self):
        """
        Test data inside fiteos/tests/data/clinoenstatite.csv which is taken from:
            * Lazarz, John D., et al. "High-pressure phase transitions of clinoenstatite."
                  American Mineralogist 104.6 (2019): 897-904.
            * Angel, R. J., and D. A. Hugh‐Jones. "Equations of state and thermodynamic
                  properties of enstatite pyroxenes." Journal of Geophysical Research:
                  Solid Earth 99.B10 (1994): 19777-19783.
        """

        # set data file
        input_file = os.path.dirname(__file__) + "/data/clinoenstatite.csv"

        # set confidence interval
        confidence_interval = 0.95

        # set random seed
        numpy.random.seed(0)
        
        # equation to minimize
        eqn = equations.equations[equations.bm3.BM3Equation.name]()
        
        # read pressure and volume data
        x, sigma_x, y, sigma_y = io.read_csv(
                        input_file,
                        x=eqn.x_name, sigma_x="sigma_{}".format(eqn.x_name),
                        y=eqn.y_name, sigma_y="sigma_{}".format(eqn.y_name))
        
        # initialize solver
        s = solver.Solver(x=x, y=y, sigma_x=sigma_x, sigma_y=sigma_y, equation=eqn)

        # initial parameters, minimums, maximums, and varying flag
        parameters = {
            "vo" : (400.0, 0.0, None, True),
            "ko" : (134.0, 0.0, None, True),
            "kp" : (4.0, None, None, True),
        }
        for key, vals in parameters.items():
            s.set_parameter(key, value=vals[0], low=vals[1], high=vals[2], vary=vals[3])
        
        # optimize
        s.minimize(method="Nedler")
        s.minimize(method="leastsq")
        
        # plot
        s.plot(**{"confidence_interval" : confidence_interval})

        import matplotlib.pyplot as plt
        #plt.show()
        plt.close()

class TestBM2(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBM2, self).__init__(*args, **kwargs)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_bm2(self):
        """
        Test data inside fiteos/tests/data/clinoenstatite.csv which is taken from:
            * Lazarz, John D., et al. "High-pressure phase transitions of clinoenstatite."
                  American Mineralogist 104.6 (2019): 897-904.
            * Angel, R. J., and D. A. Hugh‐Jones. "Equations of state and thermodynamic
                  properties of enstatite pyroxenes." Journal of Geophysical Research:
                  Solid Earth 99.B10 (1994): 19777-19783.
        """

        # set data file
        input_file = os.path.dirname(__file__) + "/data/clinoenstatite.csv"

        # set confidence interval
        confidence_interval = 0.95

        # set random seed
        numpy.random.seed(0)

        # equation to minimize
        eqn = equations.equations[equations.bm2.BM2Equation.name]()

        # read pressure and volume data
        x, sigma_x, y, sigma_y = io.read_csv(
                        input_file,
                        x=eqn.x_name, sigma_x="sigma_{}".format(eqn.x_name),
                        y=eqn.y_name, sigma_y="sigma_{}".format(eqn.y_name))

        # initialize solver
        s = solver.Solver(x=x, y=y, sigma_x=sigma_x, sigma_y=sigma_y, equation=eqn)

        # initial parameters, minimums, maximums, and varying flag
        parameters = {
            "vo" : (400.0, 0.0, None, True),
            "ko" : (134.0, 0.0, None, True),
            "kp" : (4.0, None, None, False),
        }
        for key, vals in parameters.items():
            s.set_parameter(key, value=vals[0], low=vals[1], high=vals[2], vary=vals[3])

        # optimize
        s.minimize(method="Nedler")
        s.minimize(method="leastsq")

        # plot
        s.plot(**{"confidence_interval" : confidence_interval})

        import matplotlib.pyplot as plt
        #plt.show()
        plt.close()

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestBM3("test_bm2"))
    suite.addTest(TestBM3("test_bm3"))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
