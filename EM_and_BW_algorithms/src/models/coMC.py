import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom, normpdf
from math import  log
from ast import literal_eval
from numpy.random import normal
from models.Model import Model, Model_state

class coMC_state(Model_state):

	def __init__(self,next_matrix,obs_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...]]
		obs_matrix  = {next_state1: parameters1, next_state2: parameters2...}
		"""
		super().__init__(next_matrix)
		self.obs_matrix  = obs_matrix

	def next(self):
		c = resolveRandom(self.transition_matrix[0])
		next_state = self.transition_matrix[1][c]
		mu, sigma  = self.obs_matrix[next_state]
		next_obs   = normal(mu,sigma,1)[0]
		return [next_state,next_obs]

	def tau(self,state,obs):
		for i in range(len(self.transition_matrix[0])):
			if self.transition_matrix[1][i] == state:
				p1 = self.transition_matrix[0][i]
				if p1 > 0.0:
					return p1*normpdf(obs,self.obs_matrix[state])
		return 0.0

	#def observations(self) doesn't make sense

	def pprint(self,i):
		print("----STATE s",i,"----",sep='')
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.000001:
				print("s",i," - (mean: ",sep="",end="")
				print(round(self.obs_matrix[self.transition_matrix[1][j]][0],4),sep='',end='')
				print(', std: ',round(self.obs_matrix[self.transition_matrix[1][j]][1],4),sep='',end='')
				print(") -> s",self.transition_matrix[1][j]," : ",self.transition_matrix[0][j],sep='')

	def __str__(self):
		if len(self.transition_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.transition_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.transition_matrix[1]:
				res += str(state)+' '
			res += '\n'
			res += str(self.obs_matrix)
			res += '\n'
			return res


class coMC(Model):
	def __init__(self,states,initial_state,name="unknown coMC"):
		super().__init__(states,initial_state,name)


def loadcoMC(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(coMC_state([[],[],[]]))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = literal_eval(f.readline()[:-1])
			states.append(coMC_state([p,s],o))

		l = f.readline()

	return coMC(states,initial_state,name)
