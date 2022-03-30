#CONTINUOUS TIME MARKOV CHAIN
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from numpy.random import exponential


class CTMC_state:
	def __init__(self, next_matrix: list) -> None:
		self.next_matrix = next_matrix

	def next(self) -> list:
		exps = []
		for exp_lambda in self.next_matrix[0]:
			exps.append(exponential(1/exp_lambda))
		next_index = exps.index(min(exps))
		return [min(exps), self.next_matrix[1][next_index], self.next_matrix[2][next_index]]

	def __str__(self) -> str:
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

class CTMC:

	def __init__(self,states: list,initial_state,name: str="unknown CTMC") -> None:
		if type(initial_state) == int:
			self.initial_state = [0.0 for i in range(len(states))]
			self.initial_state[initial_state] = 1.0
		else:
			self.initial_state = initial_state
		self.states = states
		self.name = name


	def run(self,number_steps: int) -> list:
		output = []
		current = self.initial_state
		while len(output) < number_steps:
			[time_spent, next_state, symbol] = self.states[current].next()
			output += [time_spent,symbol]
			current = next_state
		return output

	def pprint(self) -> None:
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			print("lambda: ",str(self.states[i].exp_lambda))
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.000001:
					print("s",i," - (",self.states[i].next_matrix[2][j],") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()

def loadCTMC(file_path: str) -> CTMC:
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = int(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		exp_lambda = float(l[:-1])
		if l == '-\n':
			states.append(CTMC_state([[],[],[]],exp_lambda))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(CTMC_state([p,s,o],exp_lambda))

		l = f.readline()

	return CTMC(states,initial_state,name)
