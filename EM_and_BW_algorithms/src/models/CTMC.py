#CONTINUOUS TIME MARKOV CHAIN
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from numpy.random import exponential
from ast import literal_eval
from tools import resolveRandom


class CTMC_state:
	def __init__(self, lambda_matrix: list) -> None:
		#lambda should contain the parameters not the expectations !!!
		self.lambda_matrix = lambda_matrix

	def tau(self, s: int, obs: str) -> float:
		for i in range(len(self.lambda_matrix[0])):
			if self.lambda_matrix[1][i] == s and self.lambda_matrix[2][i] == obs:
				return self.lambda_matrix[0][i]/self.e()
		return 0.0

	def observations(self):
		return list(set(self.lambda_matrix[2]))

	def e(self) -> float:
		return sum(self.lambda_matrix[0])

	def expected_time(self) -> float:
		return 1/self.e()

	def next(self) -> list:
		exps = []
		for exp_lambda in self.lambda_matrix[0]:
			exps.append(exponential(1/exp_lambda))
		next_index = exps.index(min(exps))
		return [min(exps), self.lambda_matrix[1][next_index], self.lambda_matrix[2][next_index]]

	def pprint(self,i: int) -> None:
		print("\n----STATE s",i,"----",sep='')
		print("Exepected waiting time:",self.expected_time())
		
		den = sum(self.lambda_matrix[0])
		for j in range(len(self.lambda_matrix[0])):
			if self.lambda_matrix[0][j]/den > 0.0001:
				print("s",i," - (",self.lambda_matrix[2][j],") -> s",self.lambda_matrix[1][j]," : lambda = ",self.lambda_matrix[0][j],sep='')
	
	def pprint_untimed(self,i: int) -> None:
		print("\n----STATE s",i,"----",sep='')
		
		den = sum(self.lambda_matrix[0])
		for j in range(len(self.lambda_matrix[0])):
			p = self.lambda_matrix[0][j]/den
			if p > 0.0001:
				print("s",i," - (",self.lambda_matrix[2][j],") -> s",self.lambda_matrix[1][j]," : ",p,sep='')
		

	def __str__(self) -> str:
		if len(self.lambda_matrix[0]) == 0: #end state
			return "-\n"
		else:
			res = ""
			for proba in self.lambda_matrix[0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.lambda_matrix[1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.lambda_matrix[2]:
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

	def tau(self, s1: int, s2: int, obs: str) -> float:
		return self.states[s1].tau(s2,obs)

	def e(self,s: int) -> float:
		return self.states[s].e()

	def pi(self, s: int) -> float:
		return self.initial_state[s]

	def observations(self):
		"""
		Return the alphabet of the model

		:return: the alphabet of the model
		:rtype: list of string
		"""
		res = []
		for s in self.states:
			res += s.observations()
		return list(set(res))

	def run(self,number_steps: int, timed: bool = False) -> list:
		output = []
		current = resolveRandom(self.initial_state)
		c = 0
		while c < number_steps:
			[time_spent, next_state, symbol] = self.states[current].next()
			
			if timed:
				output.append(time_spent)

			output.append(symbol)
			current = next_state
			c += 1
		return output

	def pprint(self) -> None:
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			self.states[i].pprint(i)
		print()

	def pprint_untimed(self) -> None:
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			self.states[i].pprint_untimed(i)
		print()
	
	def save(self,file_path: str) -> None:
		"""
		Save the model into a txt file

		:param file_path: path of the output file
		:type states: str
		"""
		f = open(file_path,'w')
		f.write(self.name)
		f.write('\n')
		f.write(str(self.initial_state))
		f.write('\n')
		for s in self.states:
			f.write(str(s))
		f.close()


def loadCTMC(file_path: str) -> CTMC:
	"""
	Load a model saved into a text file

	:param file_path: location of the text file
	:type file_path: str

	:return: a CTMC
	:rtype: CTMC
	"""
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(CTMC_state([[],[],[]]))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(CTMC_state([p,s,o]))

		l = f.readline()

	return CTMC(states,initial_state,name)
