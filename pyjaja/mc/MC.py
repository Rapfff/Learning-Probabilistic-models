from ..base.tools import resolveRandom, randomProbabilities
from ..base.Model import Model, Model_state
from ast import literal_eval

class MC_state(Model_state):
	"""
	Initialise an MC state
	Takes on input the transition matrix of this state

	:param next_matrix: [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_symbol,transition2_symbol,...]] . next_matrix[0][x] is the probability to move to state next_matrix[1][x] generating next_matrix[2][x]
	:type next_matrix: list of one list of float, one list of int and one list of str
	"""

	def __init__(self,next_matrix: list, idd: int) -> None:
		super().__init__(next_matrix,idd)

	def next(self) -> list:
		"""
		Return a state-observation pair according to the distributions described by next_matrix

		:return: a state-observation pair
		:rtype: list of one int and one str
		"""
		c = resolveRandom(self.transition_matrix[0])
		return [self.transition_matrix[1][c],self.transition_matrix[2][c]]

	def tau(self,state: int, obs: str) -> float:
		"""
		Return the probability of generating, from this state, observation <obs> and moving to state <s>

		:param s: a state ID
		:type s: int
		
		:param obs: an observation
		:type obs: str
		
		:return: the probability of generating, from this state, observation <obs> and moving to state <s>
		:rtype: float
		"""
		for i in range(len(self.transition_matrix[0])):
			if self.transition_matrix[1][i] == state and self.transition_matrix[2][i] == obs:
				return self.transition_matrix[0][i]
		return 0.0

	def observations(self) -> list:
		"""
		Return the list of all the observations that can be generated from this state

		:return: a list of observation
		:rtype: list of str
		"""
		return list(set(self.transition_matrix[2]))

	def __str__(self) -> str:
		"""
		Print the state on terminal on terminal.

		:param i: id of the current state
		:type i: int
		"""
		res = "----STATE s"+str(self.id)+"----\n"
		for j in range(len(self.transition_matrix[0])):
			if self.transition_matrix[0][j] > 0.000001:
				res += "s"+str(self.id)+" - ("+str(self.transition_matrix[2][j])+") -> s"+str(self.transition_matrix[1][j])+" : "+str(self.transition_matrix[0][j])+'\n'
		return res

	def save(self) -> str:
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


class MC(Model):
	"""
	Initialise a MCGT.

	:param states: a list of states
	:type states: list of  MCGT_state

	:param initial_state: determine which state is the initial one (then it's the id of the state), or what are the probability to start in each state (then it's a list of probabilities) 
	:type initial_state: int or list of float

	:param name: name of the model
	:type name: str
	"""
	def __init__(self,states: list, initial_state: int ,name: str ="unknown MC") -> None:
		super().__init__(states,initial_state,name)


def HMMtoMC(h) -> MC:
	"""
	Translate a given HMM <h> to a MC

	:param h: an HMM
	:type h: HMM

	:return: a MC equivalent to <h>
	:rtype: MC
	"""
	states_g = []
	for sh in h.states:
		transitions = [[],[],[]]
		for sy in range(len(sh.output_matrix[0])):
			for ne in range(len(sh.next_matrix[0])):
				transitions[0].append(sh.output_matrix[0][sy]*sh.next_matrix[0][ne])
				transitions[1].append(sh.next_matrix[1][ne])
				transitions[2].append(sh.output_matrix[1][sy])
		states_g.append(MC_state(transitions))
	return MC(states_g,h.initial_state)


def loadMC(file_path: str) -> MC:
	"""
	Load a model saved into a text file

	:param file_path: location of the text file
	:type file_path: str

	:return: an MC
	:rtype: MC
	"""
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = literal_eval(f.readline()[:-1])
	states = []
	c = 0
	l = f.readline()
	while l and l != '\n':
		if l == '-\n':
			states.append(MC_state([[],[],[]],c))
		else:
			p = [ float(i) for i in l[:-2].split(' ')]
			l = f.readline()[:-2].split(' ')
			s = [ int(i) for i in l ]
			o = f.readline()[:-2].split(' ')
			states.append(MC_state([p,s,o],c))
		c += 1
		l = f.readline()

	return MC(states,initial_state,name)

def MC_random(nb_states,alphabet,random_initial_state=False):
	s = []
	for i in range(nb_states):
		s += [i] * len(alphabet)
	obs = alphabet*nb_states
	
	states = []
	for i in range(nb_states):
		states.append(MC_state([randomProbabilities(len(obs)),s,obs],i))
	
	if random_initial_state:
		init = randomProbabilities(nb_states)
	else:
		init = 0
	return MC(states,init,"MCGT_random_"+str(nb_states)+"_states")