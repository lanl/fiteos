from . import bm2
from . import bm3
from . import vinet

# map for built-in equations
equations = {
    bm2.BM2Equation.name : bm2.BM2Equation,
    bm3.BM3Equation.name : bm3.BM3Equation,
    vinet.VinetEquation.name : vinet.VinetEquation,
}
