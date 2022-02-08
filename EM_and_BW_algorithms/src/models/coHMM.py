import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from numpy.random import normal
from tools import resolveRandom, normpdf
from math import  log
from ast import literal_eval
from models.Model import Model, Model_state

class coHMM_state(Model_state):

	def __init__(self,next_matrix,output_parameters):
		"""
		output_parameters = [mu,sigma]
		next_matrix = [[proba_state1,proba_state2,...],[state1,state2,...]]
		"""
		super().__init__(next_matrix)
		self.output_parameters = output_parameters

	def a(self,state):
		if state in self.next_matrix[1]:
			return self.next_matrix[0][self.next_matrix[1].index(state)]
		else:
			return 0.0

	def b(self,obs):
		return normpdf(obs,self.output_parameters)

	def next_obs(self):
		mu, sigma  = self.output_parameters
		return normal(mu,sigma,1)[0]

	def next_state(self):
		return self.next_matrix[1][resolveRandom(self.next_matrix[0])]

	def next(self):
		return [self.next_state(),self.next_obs()]
	
	def tau(self,state,obs):
		return self.a(state)*self.b(obs)

	#def observations(self) doesn't make sense
		
	def pprint(self,i):
		print("----STATE s",i,"----",sep='')
		for j in range(len(self.next_matrix[0])):
			if self.next_matrix[0][j] > 0.000001:
				print("s",i," -> s",self.next_matrix[1][j]," : ",self.next_matrix[0][j],sep='')
		print("************")
		print("mean:",round(self.output_parameters[0],4))
		print("std :",round(self.output_parameters[1],4))

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
			res += str(self.output_parameters)
			res += '\n'			
			return res

class coHMM(Model):
	def __init__(self,states,initial_state,name="unknown HMM"):
		super().__init__(states,initial_state,name)

	def a(self,s1,s2):
		return self.states[s1].a(s2)

	def b(self,s,o):
		return self.states[s].b(o)


def loadcoHMM(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MCGT_state([[],[],[]]))
		else:
			ps = [ float(i) for i in l[:-2].split(' ')]
			l  = f.readline()[:-2].split(' ')
			s  = [ int(i) for i in l ]
			o  = literal_eval(f.readline()[:-1])
			states.append(coHMM_state([ps,s],o))

		l = f.readline()

	return coHMM(states,initial_state,name)
