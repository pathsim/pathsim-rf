"""
PathSim-RF: RF Engineering toolbox for PathSim

A toolbox providing specialized blocks for RF and microwave engineering
simulations in the PathSim framework.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__all__ = ["__version__"]

from .transmission_line import *
from .amplifier import *
from .mixer import *
from .netlist_to_statespace import NetlistStateSpace

try: # test if scikit-rf is present in env (pip install scikit-rf)
    from .network import *
except ImportError:
    pass
try: # test if Elecsolver is present in env (pip install ElecSolver>=2.2.1)
    from .circuit import *
except ImportError:
    pass