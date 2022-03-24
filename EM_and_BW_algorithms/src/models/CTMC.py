#CONTINUOUS TIME MARKOV CHAIN
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from tools import resolveRandom
from math import  log, exp
from numpy.random import exponential

from models.MC import MC_state, MC

class CTMC_state(MC_state):
	"""
	MCGT but stay in state for a duration given by an exp law of parameter exp_lambda
	"""
	def __init__(self, next_matrix, exp_lambda):
		super(CTMC_state, self).__init__(next_matrix)
		self.exp_lambda = exp_lambda

	def next(self):
		return [exponential(1/self.exp_lambda)]+super().next()

	def t(self,x):
		return self.exp_lambda * exp(-self.exp_lambda*x)

	def __str__(self):
		return str(self.exp_lambda)+"\n"+str(super())


class CTMC(MC):

	def __init__(self,states,initial_state,name="unknown CTMC"):
		super().__init__(states,initial_state,name)

	def v(self,s,x):
		return self.states[s].t(x)


	def run(self,number_steps):
		output = []
		current = self.initial_state
		while len(output) < number_steps:
			[time_spent, next_state, symbol] = self.states[current].next()
			output += [time_spent,symbol]
			current = next_state
		return output

	def pprint(self):
		print(self.name)
		print(self.initial_state)
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			print("lambda: ",str(self.states[i].exp_lambda))
			for j in range(len(self.states[i].next_matrix[0])):
				if self.states[i].next_matrix[0][j] > 0.000001:
					print("s",i," - (",self.states[i].next_matrix[2][j],") -> s",self.states[i].next_matrix[1][j]," : ",self.states[i].next_matrix[0][j],sep='')
		print()

	def logLikelihood_with_time(self,sequences):
		loglikelihood = 0.0

		for i in range(len(sequences[0])):
			seq = sequences[0][i]
			times=sequences[1][i]
			alpha_matrix = self._initAlphaMatrix(len(seq)//2)

			for k in range(0,len(seq),2):
				for s in range(len(self.states)):
					summ = 0.0
					for ss in range(len(self.states)):
						p = self.states[ss].g(s,seq[k+1])*self.states[ss].t(seq[k])
						summ += alpha_matrix[ss][k//2]*p
					alpha_matrix[s][(k//2)+1] = summ
			
			if sum([alpha_matrix[s][-1] for s in range(len(self.states))]) <= 0:
				return None
			else:
				loglikelihood += log(sum([alpha_matrix[s][-1] for s in range(len(self.states))])) * times

		return loglikelihood / sum(sequences[1])


def loadCTMC(file_path):
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
