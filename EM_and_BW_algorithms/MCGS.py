from tools import resolveRandom

class MCGS_state:

	def __init__(self,output_symbol, next_matrix):
		"""
		next_matrix = [[proba_state1,proba_state2,...],[state1,state2,...]]
		"""
		if sum(next_matrix[0]) < 1.0:
			print("Sum of the probabilies of the next_matrix should be 1.0")
			return False
		self.output_symbol = output_symbol
		self.next_matrix = next_matrix

	def generate(self):
		return self.output_symbol

	def next(self):
		return self.next_matrix[1][resolveRandom(self.next_matrix[0])]


class MCGS:

	def __init__(self,states,initial_state):
		self.initial_state = initial_state
		self.states = states

	def run(self,number_steps):
		output = ""
		current = self.initial_state

		while len(output) < number_steps:
			output += self.states[current].generate()
			current = self.states[current].next()

		return output

def HMMtoMCGS(h):
	states_m = []
	new_states_m = []
	offset = len(h.states)
	
	for sh in h.states:
		states_m.append(MCGS_state('',[sh.output_matrix[0],[x+offset for x in range(len(sh.output_matrix[0]))]]))
		offset += len(sh.output_matrix[0])
		for sy in sh.output_matrix[1]:
			new_states_m.append(MCGS_state(sy,sh.next_matrix))
	return MCGS(states_m+new_states_m,h.initial_state)
