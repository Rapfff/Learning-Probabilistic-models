import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from examples_models import modelMCGT_REBER
from tools import randomLTL

h = modelMCGT_REBER()

prop = randomLTL(1,3,['B','E','P','S','T','V','X'])

for i in prop:
	print(i)
print(h.checkListOrLTL(prop))