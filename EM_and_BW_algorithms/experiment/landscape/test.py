from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

from Estimation_algorithm_MCGT import Estimation_algorithm_MCGT, to_index, to_probas
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)
from examples.examples_models import modelMCGT5
from src.tools import generateSet
import numpy as np

original = modelMCGT5()
training_set = generateSet(original,1000,8)
test_set = generateSet(original,200,8)


accuracy_start = 1
accuracy_gen   = 3
"""
mov = {}
val = {}

for p1 in range(10**accuracy_start + 1):
	for p2 in range(10**accuracy_start + 1):
		for p3 in range(10**accuracy_start + 1):
			for p4 in range(10**accuracy_start + 1):
				if p1+p2+p3 <= 10**accuracy_start:
					print(p1/10**accuracy_start,p2/10**accuracy_start,p3/10**accuracy_start,p4/10**accuracy_start)
					algo = Estimation_algorithm_MCGT(modelMCGT5(p1/10**accuracy_start,p2/10**accuracy_start,p3/10**accuracy_start,p4/10**accuracy_start),original.observations())
					algo.learn(training_set,test_set,mov,val,accuracy_gen)

f = open("mov.txt",'w')
f.write(str(mov))
f.close()

f = open("val.txt",'w')
f.write(str(val))
f.close()
"""
f = open("mov.txt",'r')
l = f.readline()
f.close()

mov = {}
l = l[1:-2].split(', ')
l = [i.split(': ') for i in l]
for i in l:
	mov[i[0][1:-1]] = i[1][1:-1]

f = open("val.txt",'r')
l = f.readline()
f.close()

val = {}
l = l[1:-2].split(', ')
l = [i.split(': ') for i in l]
for i in l:
	val[i[0][1:-1]] = i[1][1:-1]


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

points_x = []
points_y = []
points_z = []
points_c = []
points_id  = []
for p1 in range(10**accuracy_start + 1):
	for p2 in range(10**accuracy_start + 1):
		for p3 in range(10**accuracy_start + 1):
			for p4 in range(10**accuracy_start + 1):
				if p1+p2+p3 <= 10**accuracy_start:
					points_x.append(p1/10**accuracy_start)
					points_y.append(p2/10**accuracy_start)
					points_z.append(p3/10**accuracy_start)
					points_c.append(p4/10**accuracy_start)
					
					p1_str = str(p1/10**accuracy_start)
					p2_str = str(p2/10**accuracy_start)
					p3_str = str(p3/10**accuracy_start)
					p4_str = str(p4/10**accuracy_start)

					points_id.append(p1_str+'|'+p2_str+'|'+p3_str+'|'+p4_str)

# Setting the axes properties
ax.set_xlim3d([0.0,1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0,1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0,1.0])
ax.set_zlabel('Z')

graph = ax.scatter(points_x, points_y, points_z,c=points_c, vmin=0.0,vmax=1.0)

def update(num,graph,points_id,mov):
	points_x = []
	points_y = []
	points_z = []
	points_c = []
	for i in range(len(points_id)):
		p = to_probas(mov[points_id[i]])
		points_id[i] = to_index(p)
		points_x.append(p[0])
		points_y.append(p[1])
		points_z.append(p[2])
		points_c.append(p[3])
	
	points_x = np.array(points_x)
	points_y = np.array(points_y)
	points_z = np.array(points_z)
	points_c = np.array(points_c)
	graph._offsets3d = (points_x, points_y, points_z)
	graph.set_array(points_c)

ani = matplotlib.animation.FuncAnimation(fig, update, fargs=(graph,points_id,mov))
ani.save('myAnimation.gif', dpi=1000, fps=30)
