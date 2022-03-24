import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom
from models.Model import Model, Model_state
from ast import literal_eval

class HMM_state(Model_state):

	def __init__(self,output_matrix, next_matrix):
		"""
		output_matrix = [[proba_symbol1,proba_symbol2,...],[symbol1,symbol2,...]]
		next_matrix = [[proba_state1,proba_state2,...],[state1,state2,...]]
		"""
		super().__init__(next_matrix)
		if round(sum(output_matrix[0]),2) != 1.0 and sum(output_matrix[0]) != 0:
			print("Sum of the probabilies of the output_matrix should be 1 or 0 here it's ",sum(output_matrix[0]))
			#return False
		self.output_matrix = output_matrix

	def a(self,state):
		if state in self.next_matrix[1]:
			return self.next_matrix[0][self.next_matrix[1].index(state)]
		else:
			return 0.0

	def b(self,sigma):
		if sigma in self.output_matrix[1]:
			return self.output_matrix[0][self.output_matrix[1].index(sigma)]
		else:
			return 0.0

	def next_obs(self):
		return self.output_matrix[1][resolveRandom(self.output_matrix[0])]

	def next_state(self):
		return self.next_matrix[1][resolveRandom(self.next_matrix[0])]

	def next(self):
		return [self.next_state(),self.next_obs()]
	
	def tau(self,state,obs):
		return self.a(state)*self.b(obs)

	def observations(self):
		return list(set(self.output_matrix[1]))
		

	def pprint(self,i):
		print("----STATE s",i,"----",sep='')
		for j in range(len(self.next_matrix[0])):
			if self.next_matrix[0][j] > 0.000001:
				print("s",i," -> s",self.next_matrix[1][j]," : ",self.next_matrix[0][j],sep='')
		print("************")
		for j in range(len(self.output_matrix[0])):
			if self.output_matrix[0][j] > 0.000001:
				print("s",i," => ",self.output_matrix[1][j]," : ",self.output_matrix[0][j],sep='')


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

			for proba in self.output_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for obs in self.output_matrix[1]:
				res += str(obs)+' '
			res += '\n'
			return res

class HMM(Model):
	def __init__(self,states,initial_state,name="unknown HMM"):
		super().__init__(states,initial_state,name)

	def a(self,s1,s2):
		return self.states[s1].a(s2)

	def b(self,s,o):
		return self.states[s].b(o)

def loadHMM(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(HMM_state([[],[],[]]))
		else:
			ps = [ float(i) for i in l[:-2].split(' ')]
			l  = f.readline()[:-2].split(' ')
			s  = [ int(i) for i in l ]
			l  = f.readline()[:-2].split(' ')
			po = [ float(i) for i in l]
			o  = f.readline()[:-2].split(' ')
			states.append(HMM_state([po,o],[ps,s]))

		l = f.readline()

	return HMM(states,initial_state,name)
