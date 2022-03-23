import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom
from models.Model import Model, Model_state
from ast import literal_eval
from math import  log

class MCGT_state(Model_state):

	def __init__(self,next_matrix):
		"""
		next_matrix = [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]]
		"""
		super().__init__(next_matrix)

	def next(self):
		c = resolveRandom(self.transition_matrix[0])
		return [self.transition_matrix[1][c],self.transition_matrix[2][c]]

	def tau(self,state,obs):
		for i in range(len(self.transition_matrix[0])):
			if self.transition_matrix[1][i] == state and self.transition_matrix[2][i] == obs:
				return self.transition_matrix[0][i]
		return 0.0

	def observations(self):
		return list(set(self.transition_matrix[2]))

	def pprint(self,i):
		print("----STATE s",i,"----",sep='')
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.000001:
				print("s",i," - (",self.transition_matrix[2][j],") -> s",self.transition_matrix[1][j]," : ",self.transition_matrix[0][j],sep='')

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
			for obs in self.transition_matrix[2]:
				res += str(obs)+' '
			res += '\n'
			return res


class MCGT(Model):
	def __init__(self,states,initial_state,name="unknown MCGT"):
		super().__init__(states,initial_state,name)


def HMMtoMCGT(h):
	states_g = []
	for sh in h.states:
		transitions = [[],[],[]]
		for sy in range(len(sh.output_matrix[0])):
			for ne in range(len(sh.next_matrix[0])):
				transitions[0].append(sh.output_matrix[0][sy]*sh.next_matrix[0][ne])
				transitions[1].append(sh.next_matrix[1][ne])
				transitions[2].append(sh.output_matrix[1][sy])
		states_g.append(MCGT_state(transitions))
	return MCGT(states_g,h.initial_state)


def loadMCGT(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MCGT_state([[],[],[]]))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(MCGT_state([p,s,o]))

		l = f.readline()

	return MCGT(states,initial_state,name)
