from examples_models import *
from Estimation_algorithms_MCGT_multiple import EM_ON_MCGT as EM_multiple
from Estimation_algorithms_MCGT import EM_ON_MCGT

alphabet = ['x','y','a','b','c','d']
h = modelMCGT_random(5,alphabet)
print("\nWith EM_multiple (same random initialization):")
res = EM_multiple(h,alphabet).problem3multiple([["yadd"],[1]])
print("------------------------")
print("With EM (same random initialization):")
EM_ON_MCGT(h,alphabet).problem3("yadd")
print()
print(res)