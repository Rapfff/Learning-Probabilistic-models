import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom, normpdf
from math import  log
from ast import literal_eval
from numpy.random import normal
from models.MCGT import MCGT

class coMC_state:

	def __init__(self,next_matrix,obs_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...]]
		obs_matrix  = {next_state1: parameters1, next_state2: parameters2...}
		"""
		if round(sum(next_matrix[0]),2) < 1.0 and sum(next_matrix[0]) != 0:
			print("Sum of the probabilies of the next_matrix should be 1 or 0 here it's ",sum(next_matrix[0]))
			#return False
		self.next_matrix = next_matrix
		self.obs_matrix  = obs_matrix

	def next(self):
		c = resolveRandom(self.next_matrix[0])
		next_state = self.next_matrix[1][c]
		mu, sigma  = self.obs_matrix[next_state]
		next_obs   = normal(mu,sigma,1)[0]
		return [next_state,next_obs]

	def g(self,state,obs):
		for i in range(len(self.next_matrix[0])):
			if self.next_matrix[1][i] == state:
				p1 = self.next_matrix[0][i]
				p2 = normpdf(obs,self.obs_matrix[state])
				return p1*p2
		return 0.0

	def __str__(self):
		if len(self.next_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.next_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.next_matrix[1]:
				res += str(state)+' '
			res += '\n'
			res += str(self.obs_matrix)
			res += '\n'
			return res


class coMC(MCGT):
	def __init__(self,states,initial_state,name="unknown coMC"):
		super().__init__(states,initial_state,name)

	def pprint(self):
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.000001:
					print("s",i," - (mean: ",sep="",end="")
					print(self.states[i].obs_matrix[self.states[i].next_matrix[1][j]][0],sep='',end='')
					print(', std: ',self.states[i].obs_matrix[self.states[i].next_matrix[1][j]][1],sep='',end='')
					print(") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()

def loadcoMC(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = int(f.readline()[:-1])
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
