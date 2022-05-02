import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from models.MDP import *
from math import sqrt, log


class IOFTPA_state:

	def __init__(self, obs, c, lbl):
		self.observation = obs
		self.counter = c
		self.label = lbl
		self.id = -1
		self.transitions = {}

	def counter_add(self,n):
		self.counter += n

	def set_id(self,i):
		self.id = i

	def successors(self):
		res = []
		for a in self.transitions:
			for s in self.transitions[a][1]:
				if s not in res:
					res.append(s)
		res.sort()
		return res

	def successors_action(self,action):
		if not self.action_allowed(action):
			return []
		return self.transitions[action][1]


	def transitions_add(self,action,prob,state):
		if not self.action_allowed(action):
			self.transitions[action] = [[prob],[state]]
		else:
			self.transitions[action][0].append(prob)
			self.transitions[action][1].append(state)

	def transition_change(self,action,old_state,new_state):
		self.transitions[action][1][self.transitions[action][1].index(old_state)] = new_state

	def transition_prob_add(self,action,state,x):
		self.transitions[action][0][self.transitions[action][1].index(state)] += x

	def get_transition_prob(self,action,s2):
		return self.transitions[action][0][self.transitions[action][1].index(s2)]

	def action_allowed(self,act):
		return act in self.transitions

	def actions_allowed(self):
		return [a for a in self.transitions]

	def pprint(self):
		print(self.label,self.transitions,sep="\t")



class IOFTPA:

	def __init__(self,states,o,a):
		self.states = states
		self.observations = o
		self.actions = a

	def pprint(self):
		for s in self.states:
			s.pprint()

	def successor(self,state_index,action,obs):
		if not self.states[state_index].action_allowed(action):
			return None

		for i in self.states[state_index].successors_action(action):
			if self.states[i].observation == obs:
				return i


	def compatible(self,s1,s2,alpha):
		if s1 == None or s2 == None:
			return True
		if self.states[s1].observation != self.states[s2].observation:
			return False
		
		for a in self.actions:
			for o in self.observations:
				if not self.hoeffding(s1,s2,a,o,alpha):
					return False
				if not self.compatible(self.successor(s1,a,o),self.successor(s2,a,o),alpha):
					return False

		return True
				
	def hoeffding(self,s1,s2,a,o,alpha):
		i1 = self.successor(s1,a,o)
		i2 = self.successor(s2,a,o)

		
		if i1 == None or i2 == None:
			return True
				
		f1 = self.states[s1].get_transition_prob(a,i1)
		n1 = sum(self.states[s1].transitions[a][0])
		f2 = self.states[s2].get_transition_prob(a,i2)
		n2 = sum(self.states[s2].transitions[a][0])

		if n1*n2 == 0:
			return True

		return abs((f1/n1)-(f2/n2)) < (sqrt(1/n1)+sqrt(1/n2))*sqrt(log(2/alpha)/2)

	def find_predec(self,s):
		res = []
		for i in range(len(self.states)):
			for a in self.states[i].actions_allowed():
				for j in self.states[i].successors_action(a):
					if j == s:
						res.append((i,a))
		return res

	def merge(self,s1,s2):
		#action = self.states[s2].label[-2]
		if s1 == None or s2 == None:
			return None

		predec = self.find_predec(s2)
		for sa in predec:
			self.states[sa[0]].transition_change(sa[1],s2,s1)

		self.states[s1].counter_add(self.states[s2].counter)
		self.states[s2].counter = 0 #useless but meaningfull

		for a in self.states[s2].actions_allowed():
			for o in self.observations:
				succ2 = self.successor(s2,a,o)
				if succ2 != None:
					succ1 = self.successor(s1,a,o)
					if succ1 != None:
						self.merge(succ1,succ2)
						#self.states[s1].transition_prob_add(a,succ1,self.states[s2].get_transition_prob(a,succ2))
						#self.states[succ1].counter_add(self.states[succ2].counter)
						#self.states[succ2].counter = 0 #useless but meaningfull
					else:
						self.states[s1].transitions_add(a,self.states[s2].get_transition_prob(a,succ2),succ2)

	def run_seq(self,seq):
		s_current = 0
		i_current = 0
		path = [s_current]
		while i_current < len(seq):
			f = True
			for s in self.states[s_current].successors_action(seq[i_current]):
				if self.states[s].observation == seq[i_current+1]:
					s_current = s
					path.append(s_current)
					i_current += 2
					f = False
					break
			if f:
				for s in path:
					self.states[s].pprint()
				print()
				print(seq)
				return False
		return True


	def cleanMDP(self,red):
		states = []
		for i in range(len(red)):
			self.states[red[i]].set_id(i)

		for i in red:
			dic = self.states[i].transitions
			for a in dic:
				dic[a].append([])
				tot = sum(self.states[i].transitions[a][0])
				for j in range(len(dic[a][0])):
					dic[a][0][j] /= tot
					dic[a][2].append(self.states[dic[a][1][j]].observation)
					dic[a][1][j] = self.states[dic[a][1][j]].id
			states.append(MDP_state(dic))
		return MDP(states,0)



class IOAlergia:

	def __init__(self):
		"""
		Given a set of seq of observations return the MCGT learned by ALERGIA
		sample = [[seq1,seq2,...],[val1,val2,...]]
		all seq have same length
		"""
		None
		
	def initialize(self,sample,alpha,actions=None,observations=None):
		pass
		self.alpha = alpha
		self.sample = sample


		if actions == None or observations == None:
			self.observations = []
			self.actions = []
			for seq in sample[0]:
				for i in range(1,len(seq),2):
					if observations == None:
						if not seq[i] in self.observations:
							self.observations.append(seq[i])
					if actions == None:
						if not seq[i-1] in self.actions:
							self.actions.append(seq[i-1])
		
		if actions != None:
			self.actions = actions
		if observations != None:
			self.observations = observations

		self.actions.sort()
		self.observations.sort()
		self.N = sum(sample[1])
		self.n = len(sample[0][0])

		self.t = self.buildIOFTPA()
		self.a = self.t
		
	def buildIOFTPA(self):
		states_lbl = [[]]

		states = [IOFTPA_state("",self.N,[])]
		
		#states_transitions = [
		#						state1: {action1: [[proba1,proba2,...],[state1,state2,...]], action2: [[proba1,proba2,...],[state1,state2,...]], ...},
		#						state2: {action1: [[proba1,proba2,...],[state1,state2,...]], action2: [[proba1,proba2,...],[state1,state2,...]], ...},
		#
		#						...
		#					  ]

		states_transitions = []

		#init states_lbl and states_counter
		for i in range(0,self.n,2):
			for seq in range(len(self.sample[0])):
				if not self.sample[0][seq][:i+2] in states_lbl:
					states_lbl.append(self.sample[0][seq][:i+2])
					states.append( IOFTPA_state(self.sample[0][seq][i+1], self.sample[1][seq], self.sample[0][seq][:i+2]))
				else:
					states[states_lbl.index(self.sample[0][seq][:i+2])].counter_add(self.sample[1][seq])

		#sorting states
		states_lbl.sort()
		for s in states:
			s.set_id(states_lbl.index(s.label))
	
		states_sorted = [None]*len(states)	
		for s in states:
			states_sorted[s.id] = s


		#init states_transitions
		for s1 in range(len(states_sorted)):
			
			len_s1 = len(states_sorted[s1].label)
			
			s2 = s1 + 1
			while s2 < len(states_sorted):
				if len(states_sorted[s2].label) < len_s1 + 2: # too short
					s2 += 1
				elif len(states_sorted[s2].label) > len_s1 + 2: # too long
					s2 += 1
				elif states_sorted[s2].label[:-2] != states_sorted[s1].label: # not same prefix
					s2 += 1
				else: # OK
					act = states_sorted[s2].label[-2]
					states_sorted[s1].transitions_add(act, states_sorted[s2].counter, s2)
					s2 += 1

		return IOFTPA(states_sorted,self.observations,self.actions)

	def learn(self,sample,alpha,actions=None,observations=None):

		self.initialize(sample,alpha,actions=None,observations=None)
		red = [0]
		blue = self.a.states[0].successors()

		while len(blue) != 0:
			state_b = blue[0]
			merged = False
			
			for state_r in red:
				if self.t.compatible(state_r,state_b,self.alpha):
					self.a.merge(state_r,state_b)
					merged = True
					break

			if not merged:
				red.append(state_b)
			blue.remove(state_b)

			for state_r in red:
				for state_b in self.a.states[state_r].successors():
					if state_b not in red:
						blue.append(state_b)

			blue = list(set(blue))
			blue.sort()

		return self.a.cleanMDP(red)