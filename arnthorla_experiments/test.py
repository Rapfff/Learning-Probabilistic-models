import os, sys, copy
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from EM_and_BW_algorithms.examples.examples_models import *
from EM_and_BW_algorithms.src.learning.Estimation_algorithm_HMM import *
from Experiment import *
#from ..src.models.HMM import *

#model = modelHMM1()
#model = modelHMM1_equiprobable()
#model = modelHMM2()
#model = modelHMM2_random()
#model = modelHMM2_equiprobable()
#model = modelHMM3()
#model = modelHMM4()
#model = modelHMM_random(10, "$abcdxy")
#model.pprint()
#observations = model.observations()
#algo = Estimation_algorithms_HMM(model,observations)
#
#seq = "$bd"
#p1 = algo.problem1(seq)
#print( p1 )
#p2 = algo.problem2(seq)
#print(p2)
#
#print()
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )
#print( model.run(10) )


#def getDimensions( li, dimensions=None ):
#    if dimensions is None:
#        dimensions = []
#    if type(li) != list:
#        return dimensions
#    else:
#        dimensions.append(len(li))
#        li = li[0]
#        return getDimensions( li, dimensions )
#
#def logList( li, dimensions, logFunc ):
#    if len(dimensions) == 1:
#        logFunc( li )
#    else:
#        length = dimensions.pop(0)
#        for i in range(length):
#            logList( li[i], copy.copy(dimensions), logFunc )
#
#
#
#res = ([[[[None for i in range(4)] 
#            for j in range(3)] 
#            for k in range(10)]
#            for m in range(5)])
##print()
##print( getDimensions( res ) )    
#
#for i in range(5):
#    for j in range(10):
#        print('[')
#        for k in range(3):
#                res[i][j][k][0] = i
#                res[i][j][k][1] = j
#                res[i][j][k][2] = k
#                print("[",i,",",j,",",k,"]","t[",k,"]: ", res[i][j][k])
#        print(']')
#
#print()
##print( getDimensions( res ) )
#dim = getDimensions( res )
#print( dim )    
#print()
#
#logList( res, dim, print )
#print()

#tmp = [ "blabla", "baba", "la" ]
#print( tmp )
#tmp2 = map( len, tmp )
#tmp3 = list(tmp2)
#print( tmp3 )
#print( max( 7,len(max( tmp, key=len ))))

exp1 = Ex_Generator_Model_Trained_On_Own_Output([13],2,[modelMCGT_random],Estimation_algorithm_MCGT,[2])
exp1.run()
