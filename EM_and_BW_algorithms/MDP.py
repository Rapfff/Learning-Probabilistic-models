from tools import resolveRandom
from math import log

class FiniteMemoryScheduler:
	def __init__(self,next_matrix,transition_matrix):
		"""
		next_matrix = {scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					   scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					   ...}
		transition_matrix = {obs1: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 obs2: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 ...}
		"""	
		self.s = 0
		self.next_matrix = next_matrix
		self.transition_matrix = transition_matrix

	def get_action(self):
		"""return an action to execute by the agent"""
		return self.next_matrix[self.s][1][resolveRandom(self.next_matrix[self.s][0])]

	def add_observation(self,obs):
		"""give to the scheduler the new observation seen by the agent"""
		if obs in self.transition_matrix:
			self.s = self.transition_matrix[obs][self.s]

	def get_actions(self):
		"""return the actions (and their probability) that the agent can execute now"""
		return self.next_matrix[self.s]


class MDP_state:

	def __init__(self,next_matrix):
		"""
		next_matrix = {action1 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]],
					   action2 : [[proba_transition1,proba_transition2,...],[transition1_state,transition2_state,...],[transition1_obs,transition2_obs,...]]
					   ...}
		"""
		for action in next_matrix:
			if round(sum(next_matrix[action][0]),2) < 1.0:
				print("Sum of the probabilies of the next_matrix should be 1.0 here it's ",sum(next_matrix[0]))
				return False
		self.next_matrix = next_matrix

	def next(self,action):
		if not action in self.next_matrix:
			print("ACTION",action,"is not available in state")
		c = resolveRandom(self.next_matrix[action][0])
		return [self.next_matrix[action][1][c],self.next_matrix[action][2][c]]

	def actions(self):
		return [i for i in self.next_matrix]

	def g(self,action,state,obs):
		if action not in self.actions():
			return 0.0
		for i in range(len(self.next_matrix[action][0])):
			if self.next_matrix[action][1][i] == state and self.next_matrix[action][2][i] == obs:
				return self.next_matrix[action][0][i]
		return 0.0


class MDP:

	def __init__(self,states,initial_state):
		self.initial_state = initial_state
		self.states = states

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0
			
	def run(self,number_steps,scheduler):
		#output = [self.states[self.initial_state].observation]
		res = []
		#actions = []
		current = self.initial_state

		while len(res)/2 < number_steps:
			action = scheduler.get_action()
			#actions.append(action)
			while action not in self.states[current].next_matrix:
				action = scheduler.get_action()

			res.append(action)
			next_state, observation = self.states[current].next(action)

			#output.append(observation)
			res.append(observation)
			scheduler.add_observation(observation)

			current = next_state

		#return [output,actions]
		return res

	def pprint(self):
		for i in range(len(self.states)):
			print("\n----STATE s",i,"----",sep='')
			for action in self.states[i].next_matrix:
				for j in range(len(self.states[i].next_matrix[action][0])):
					if self.states[i].next_matrix[action][0][j] > 0.0:
						print("s",i," - (",action,") -> s",self.states[i].next_matrix[action][1][j]," : ",self.states[i].next_matrix[action][2][j],' : ',self.states[i].next_matrix[action][0][j],sep='')
		print()
	
	def allStatesPathIterative(self, start, trace):
		"""return all the states path from start that can generate trace"""
		res = []
		action = trace[0]
		obs  = trace[1]

		if not action in self.states[start].actions():
			return []

		for i in range(len(self.states[start].next_matrix[action][1])):

			if self.states[start].next_matrix[action][0][i] > 0 and self.states[start].next_matrix[action][2][i] == obs:
				if len(trace) == 2:
					res.append([start,self.states[start].next_matrix[action][1][i]])
				else:
					t = self.allStatesPathIterative(self.states[start].next_matrix[action][1][i],trace[2:])
					for j in t:
						res.append([start]+j)
		return res
	
	def allStatesPathTrace(self,trace):
		"""return all the states path that can generate obs_seq"""
		res = []
		for j in self.allStatesPathIterative(self.initial_state,trace):
			res.append(j)
		return res

	def probabilityStateTrace(self,states_path, trace):
		"""return the probability to get this states_path generating this observations sequence"""
		if states_path[0] != self.initial_state:
			return 0.0
		else:
			res = 1.0
		for i in range(len(states_path)-1):
			if res == 0.0:
				return 0.0
			res *= self.states[states_path[i]].g(trace[i*2],states_path[i+1],trace[i*2+1])
		return res

	def probabilityTrace(self,trace):
		"""return the probability to get this trace"""
		res = 0
		for p in self.allStatesPathTrace(trace):
			res += self.probabilityStateTrace(p,trace)
		return res

	def logLikelihoodTraces(self,sequences):
		res = 0
		for i in range(len(sequences[0])):
			p = self.probabilityTrace(sequences[0][i])
			if p == 0:
				return -256
			res += log(p) * sequences[1][i]
		return res / sum(sequences[1])
