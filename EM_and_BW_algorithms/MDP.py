from tools import resolveRandom
from math import log
from functools import reduce
from operator import mul

class FiniteMemoryScheduler:
	def __init__(self,action_matrix,transition_matrix):
		"""
		action_matrix = {scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					     scheduler_state: [[proba1,proba2,...],[action1,action2,...]],
					     ...}
		transition_matrix = {obs1: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 obs2: [scheduler_state_dest_if_current_state_=_0,scheduler_state_dest_if_current_state_=_1,...]
							 ...}
		"""	
		self.s = 0
		self.action_matrix = action_matrix
		self.transition_matrix = transition_matrix

	def reset(self):
		self.s = 0

	def get_action(self):
		"""return an action to execute by the agent"""
		return self.action_matrix[self.s][1][resolveRandom(self.action_matrix[self.s][0])]

	def add_observation(self,obs):
		"""give to the scheduler the new observation seen by the agent"""
		if obs in self.transition_matrix:
			self.s = self.transition_matrix[obs][self.s]

	def get_actions(self,s=None):
		"""return the actions (and their probability) that the agent can execute now"""
		if s==None:
			return self.action_matrix[self.s]
		else:
			return self.action_matrix[s]

	def get_sequence_states(self,seq_obs):
		"""given a sequence of observations, return the sequence of states the scheduler goes through"""
		self.reset()
		res = [0]
		for o in seq_obs:
			self.add_observation(o)
			res.append(self.s)
		return res

	def get_probability(self,action,state=None):
		"""given a scheduler state and an action return the probability to execute this action in this state"""
		if state == None:
			state = self.s
		if not action in self.action_matrix[state][1]:
			return 0
		return self.action_matrix[state][0][self.action_matrix[state][1].index(action)]


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
			print("Not",action,"in",self.actions())
			return 0.0
		for i in range(len(self.next_matrix[action][0])):
			if self.next_matrix[action][1][i] == state and self.next_matrix[action][2][i] == obs:
				return self.next_matrix[action][0][i]
		return 0.0

	def __str__(self):
		res = ""
		for action in self.next_matrix:
			res += action
			res += '\n'
			for proba in self.next_matrix[action][0]:
				res += str(proba)+' '
			res += '\n'
			for state in self.next_matrix[action][1]:
				res += str(state)+' '
			res += '\n'
			for obs in self.next_matrix[action][2]:
				res += str(obs)+' '
			res += '\n'
		res += "*\n"
		return res

class MDP:

	def __init__(self,states,initial_state,name="unknown MDP"):
		self.initial_state = initial_state
		self.states = states
		self.name = name

	def __str__(self):
		return self.name

	def save(self,file_path):
		f = open(file_path,'w')
		f.write(self.name)
		f.write('\n')
		f.write(str(self.initial_state))
		f.write('\n')
		for s in self.states:
			f.write(str(s))
		f.close()

	def actions(self):
		res = []
		for s in self.states:
			res += s.actions()
		res = list(set(res))
		res.sort()
		return res

	def actions_state(self,s):
		return self.states[s].actions()

	def observations(self):
		res = []
		for s in self.states:
			for act in s.actions():
				res += s.next_matrix[act][2]
		res = list(set(res))
		res.sort()
		return res

	def pi(self,s):
		if s == self.initial_state:
			return 1.0
		else:
			return 0.0

	def g(self,s1,action,s2,obs):
		return self.states[s1].g(action,s2,obs)
			
	def run(self,number_steps,scheduler,with_action=True):
		#output = [self.states[self.initial_state].observation]
		res = []
		#actions = []
		current = self.initial_state
		scheduler.reset()

		current_len = 0

		while current_len < number_steps:
			action = scheduler.get_action()
			#actions.append(action)
			while action not in self.states[current].next_matrix:
				action = scheduler.get_action()

			if with_action:
				res.append(action)
			next_state, observation = self.states[current].next(action)

			#output.append(observation)
			res.append(observation)
			scheduler.add_observation(observation)

			current = next_state
			current_len += 1

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

	#-------------------------------------------
	def logLikelihoodTraces(self,sequences):
		probas = self.getProbaStateObservationsPaths(sequences[0])
		res = 0
		for p in range(len(probas)):
			if probas[p]== 0:
				return -256
			res += log(probas[p]) * sequences[1][p]
		return res / sum(sequences[1])

	def getProbaStateObservationsPaths(self,sequences):
		sequences = [list(i) if type(i) == type('x') else i for i in sequences]
		cur_state = self.initial_state
		states = [self.initial_state]
		final_probs = [[] for i in range(len(sequences))]

		finished = False

		prev_action = 0
		actions = [self.actions_state(cur_state)[0]]

		choices = [0]
		prev_choice = 0
		obs =   [self.states[cur_state].next_matrix[actions[-1]][2][0]]
		probs = [self.states[cur_state].next_matrix[actions[-1]][0][0]]
		states.append(self.states[cur_state].next_matrix[actions[-1]][1][0])
		cur_state =  states[-1]
		trace = []
		for i in range(len(obs)):
			trace.append(actions[i])
			trace.append(obs[i])

		while not finished:
			if (trace in sequences) or (trace not in [ s[:len(trace)] for s in sequences]) or (probs[-1] <= 0.0):
				if trace in sequences:
					#save it
					final_probs[sequences.index(trace)].append(reduce(mul, probs, 1))
				
				while True:
					#roll back choice
					obs = obs[:-1]
					probs = probs[:-1]
					prev_choice = choices[-1]
					choices = choices[:-1]
					states = states[:-1]
					cur_state = states[-1]
					
					#if last choice
					if prev_choice == len(self.states[cur_state].next_matrix[actions[-1]][0])-1:
						#roll back action
						prev_action = self.actions_state(cur_state).index(actions[-1])
						actions = actions[:-1]
					
						#if not last action
						if prev_action < len(self.actions_state(cur_state))-1:
							#take next action
							actions.append(self.actions_state(cur_state)[prev_action+1])
							#take first choice
							choices.append(0)
							obs.append(   self.states[cur_state].next_matrix[actions[-1]][2][choices[-1]])
							probs.append( self.states[cur_state].next_matrix[actions[-1]][0][choices[-1]])
							states.append(self.states[cur_state].next_matrix[actions[-1]][1][choices[-1]])
							cur_state =  states[-1]
							break					
						#if last action it will roll back choice again
						#if last action AND in initial state then it's done
						elif len(states) == 1:
							finished = True
							break

					#if not last choice
					else:
						#take next choice
						choices.append(prev_choice+1)
						obs.append(   self.states[cur_state].next_matrix[actions[-1]][2][choices[-1]])
						probs.append( self.states[cur_state].next_matrix[actions[-1]][0][choices[-1]])
						states.append(self.states[cur_state].next_matrix[actions[-1]][1][choices[-1]])
						cur_state =  states[-1]
						break
			
			else:
				#take first action
				prev_action = 0
				actions.append(self.actions_state(cur_state)[0])
				#take first choice
				choices.append(0)
				obs.append(   self.states[cur_state].next_matrix[actions[-1]][2][choices[-1]])
				probs.append( self.states[cur_state].next_matrix[actions[-1]][0][choices[-1]])
				states.append(self.states[cur_state].next_matrix[actions[-1]][1][choices[-1]])
				cur_state =  states[-1]

			trace = []
			for i in range(len(obs)):
				trace.append(actions[i])
				trace.append(obs[i])


		final_probs = [sum(i) for i in final_probs]
		return final_probs
	
	#-------------------------------------------
	def allStatesPathObservations(self,seq_obs):
		res = [[self.initial_state]]
		c = 0
		while c < len(seq_obs):
			new = []
			for p in res:
				for act in self.states[p[-1]].actions():
					for s in range(len(self.states)):
						if self.g(p[-1],act, s, seq_obs[c]) > 0.0:
							new.append(p+[act,seq_obs[c],s])
			c += 1
			res = new
		return res

	def probabilityStateActionObservationWithScheduler(self,path,scheduler):
		scheduler.reset()
		c = 0
		p = 1
		while p > 0.0 and c < len(path)-1:
			p *= scheduler.get_probability(path[c+1])
			p *= self.g(path[c],path[c+1],path[c+3],path[c+2])
			scheduler.add_observation(path[c+2])
			c += 3
		return p

	def probabilityObservationsScheduler(self,seq_obs,scheduler):
		"""return the probability to get this sequence of observations with this scheduler"""
		res = 0
		for p in self.allStatesPathObservations(seq_obs):
			res += self.probabilityStateActionObservationWithScheduler(p,scheduler)
		return res

	def logLikelihoodObservationsScheduler(self,data):
		res = 0
		nb_seq = 0
		for scheduler,sequences in data:
			nb_seq += sum(sequences[1])
			for i in range(len(sequences[0])):
				p = self.probabilityObservationsScheduler(sequences[0][i],scheduler)
				if p == 0:
					return -256
				res += log(p) * sequences[1][i]
		return res / nb_seq


def loadMDP(file_path):
	f = open(file_path,'r')
	name = f.readline()[:-1]
	initial_state = int(f.readline()[:-1])
	states = []
	
	l = f.readline()
	while l:
		d = {}
		while l != "*\n":
			a = l[:-1]
			p = [float(i) for i in  f.readline()[:-2].split(' ')]
			s = [int(i) for i in  f.readline()[:-2].split(' ')]
			t = f.readline()[:-2].split(' ')
			d[a] = [p,s,t]
			l = f.readline()
		states.append(MDP_state(d))
		l = f.readline()
	return MDP(states,initial_state,name)