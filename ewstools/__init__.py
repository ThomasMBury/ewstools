# __init__.py
name = "ewstools"

# Import relevant modules
from . import core
from . import helpers
from . import models
from . import spatial

# Import specific classes and functions
from .core import TimeSeries
from .core import MultiTimeSeries
from .spatial import SpatialEWS, lattice_weights
