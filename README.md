# FitEOS

A Python package for fitting equations. In particular, this package was written with equation of state information in mind. Where functions that describe pressure as a function of volume are fit with experimental data (with error bars) to determine the equation of state parameters.

## Installation

To install from ``pip`` do
```
python -m pip install git+https://github.com/lanl/fiteos
```

To install from source do
```
python setup.py install
```

## Usage

There are two ways to use FitEOS: a command line interface or through library calls.
Here, we go through examples of both usages.

There is an executable to perform fits from a command-line interface.
To fit a third-order Burch-Murnaghan equation of state can be fit with
```
fiteos_fit \
    --input-file fiteos/tests/data/clinoenstatite.csv \
    --eos bm3 \
    --parameters vo:400.0:0.0:None:True ko:134.0:0.0:None:True kp:4.0:None:None:True
```

Similarily, a second-order Burch-Murnaghan equation of state can be fit with
```
fiteos_fit \
    --input-file fiteos/tests/data/clinoenstatite.csv \
    --eos bm2 \
    --parameters vo:400.0:0.0:None:True ko:134.0:0.0:None:True kp:4.0:None:None:False
```
The option ``--input-file`` is the path to a CSV file.
The option ``--eos`` is the name of the equation to fit, in this case ``bm3``; see ``--help`` for more options.
The option ``--parameters`` is a list of parameters and each parameter is five values separated by ``:``; the five values corresponds to the parameter name, initial value, minimum, maximum, and whether the parameter is varied in the fit (``True`` or ``False``).
The user may give ``None`` if they wish to not use a minimium or maximum.
For example, the parameter ``vo:400.0:0.0:None:True`` corresponds to ``vo`` (reference volume) with a minimum of 0.0, no maximum, and it is varied in the fit.

Alternatively, the user can call the library directly in a script.
Here, we go through an example.
First import the modules we will use
```
import matplotlib.pyplot as plt
from fiteos import equations
from fiteos import solver
```

FitEOS has functions for reading CSV files, an example is included with this repository with columns for ``p`` (pressure), ``sigma_p`` (pressure standard deviation), ``v`` (volume), and ``sigma_v`` (volume variance).
The following can be used to load the example file
```
# set data file
input_file = "fiteos/tests/data/clinoenstatite.csv"

# read pressure and volume data
x, sigma_x, y, sigma_y = io.read_csv(
                input_file,
                x=eqn.x_name, sigma_x="sigma_{}".format(eqn.x_name),
                y=eqn.y_name, sigma_y="sigma_{}".format(eqn.y_name))
```

Next, we load the equaiton to minimize
```
# equation to minimize
eqn = equations.equations[equations.bm3.BM3Equation.name]()
```

Next, we can create the solver and minimize
```
# set random seed
numpy.random.seed(0)

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
```

Finally, there are several plots that can accompany equations, and the follow displays the results
```
# set confidence interval
confidence_interval = 0.95

# plot
s.plot(**{"confidence_interval" : confidence_interval})
plt.show()
```

## Tests

To run unit tests do
```
python -m unittest discover -v -b
```

## Copyright

© (or copyright) 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration.
All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration.
The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
