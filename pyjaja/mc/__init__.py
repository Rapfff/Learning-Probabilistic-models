from .MC import *
from .BW_MC import *
try:
    from .BW_LTL import *
except ModuleNotFoundError as e:
    print(e,"\nModule spot required for learning with LTL")
from .alergia import *