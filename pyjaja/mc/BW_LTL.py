import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from random import sample
from ..base.tools import randomProbabilities
from .MC import *
from .BW_MC import BW_MC
import spot

class BW_LTL:
	def __init__(self) -> None:
		pass
	
	def fit(self,formula: str,traces: list,alphabet: list,output_file: str=None,epsilon: float=0.01,pp: str='',nb_states: int=None) -> MC:
		"""
		Fits the model according to ``traces`` and ``formula``.

		Parameters
		----------
		formula : str
			LTL formula expressed as https://spot.lrde.epita.fr/concepts.html#ltl .
		traces : list
			training set.
		alphabet : list
			list containing all the possible observations.
		output_file : str, optional
			if set path file of the output model. Otherwise the output model
			will not be saved into a text file.
		epsilon : float, optional
			the learning process stops when the difference between the
			loglikelihood of the training set under the two last hypothesis is
			lower than ``epsilon``. The lower this value the better the output,
			but the longer the running time. By default 0.01.
		pp : str, optional
			Will be printed at each iteration. By default ''.
		nb_states : int, optional
			Desired number of states in the output model. By default None.

		Returns
		-------
		MC
			fitted MC.
		"""
		self.generateInitialModel(formula,nb_states,alphabet)
		bw = BW_MC()
		output_model = bw.fit(traces,self.initial_model,output_file=output_file,epsilon=epsilon,pp=pp)
		return output_model
	
	def generateInitialModel(self,formula: str,nb_states: int, alphabet: list) -> None:
		"""
		Generate a MC with ``nb_states`` states such that all traces generated
		by this MC will satisfy the LTL formula ``formula``.

		Parameters
		----------
		formula : str
			An LTL formula.
		nb_states : int
			Number of states in the output model.
		alphabet : list
			List of all possibles observations.
		"""
		hoa = spot.translate(formula, 'Buchi', 'deterministic', 'state-based').to_str("hoa")
		self.initial_model = HOAtoMC(hoa,alphabet)
		self.initial_model.pprint()
		
		if nb_states != None:
			missing_states = nb_states - len(self.initial_model.states)
			if missing_states > 0:
				self.addStates( missing_states )
		
		missing_observations = list(set(alphabet)-set(self.initial_model.observations()))
		if missing_observations != []:
			self.addObservations(missing_observations)
		self.initial_model.pprint()

	def addStates(self,to_add: int) -> None:
		"""
		Split the states with the highest number of incoming transition in 
		order to get as many
		states as desired by the user.

		Parameters
		----------
		to_add : int
			number of states to add
		"""
		incoming_edges = [[] for i in self.initial_model.states]
		for s in range(len(self.initial_model.states)):
			for ss in self.initial_model.states[s].next_matrix[1]:
				incoming_edges[ss].append(s)
		while to_add >= 0:
			nb_incoming_edges = [len(s) for s in incoming_edges]
			chosen = nb_incoming_edges.index(max(nb_incoming_edges))
			next_states = self.initial_model.states[chosen].next_matrix[1][:]
			next_obs    = self.initial_model.states[chosen].next_matrix[2][:]
			next_proba  = randomProbabilities(len(next_states))
			new_state   = MC_state([next_proba,next_states,next_obs])
			self.initial_model.states.append(new_state)
			self.initial_model.initial_state.append(self.initial_model.initial_state[chosen]/2)
			#split edges
			incoming_edges.append([])
			to_new = nb_incoming_edges[chosen]//2
			to_change = sample(incoming_edges[chosen],to_new)
			for s in to_change:
				self.initial_model.states[s].next_matrix[1][self.initial_model.states[s].next_matrix[1].index(chosen)] = len(self.initial_model.states)-1
				incoming_edges[-1].append(s)
				incoming_edges[chosen].remove(s)
			for s in next_states:
				incoming_edges[s].append(len(self.initial_model.states)-1)
			
			to_add -= 1
	
def HOAtoMC(hoa: str,alphabet: list) -> MC:
	"""
	Translate a MC expressed under HOA format to a pyjaja.MC

	Parameters
	----------
	hoa : str
		MC expressed under HOA format
	alphabet : list
		list of all possible observations

	Returns
	-------
	MC
		the equivalent MC.
	"""
	hoa = hoa.split("\n")
	states = []
	i = 0
	while hoa[i][:6] != "Start:":
		i += 1
	initial_state = int(hoa[i].split(" ")[1])
	i += 1
	alphabet_LTL = hoa[i][4:].replace('"','').split(' ')[1:]
	while hoa[i-1] != "--BODY--":
		i += 1
	while hoa[i] != "--END--":
		i += 1
		next_matrix = [[],[],[]]
		while hoa[i][0] == '[':
			hoa[i] = hoa[i].replace("{0}",'')
			hoa[i] = hoa[i].split("] ")
			dest_state = int(hoa[i][1])
			labels = _transitionLabels(hoa[i][0][1:],alphabet,alphabet_LTL)
			for l in labels:
				next_matrix[1].append(dest_state)
				next_matrix[2].append(l)
			i += 1
		next_matrix[0] = randomProbabilities(len(next_matrix[1]))
		states.append(MC_state(next_matrix))
	return MC(states,initial_state)

def _transitionLabels(l,alphabet,alphabet_LTL) -> list:
	l = l.replace(" ","")
	if l == "t":
		return alphabet
	g = l.split('|')
	res = []
	for l in g:
		l = l.split('&')
		pos_AP = [alphabet_LTL[int(i)]     for i in l if i[0] != '!'] # list of all positive AP for this transition
		neg_AP = [alphabet_LTL[int(i[1:])] for i in l if i[0] == '!'] # list of all negative AP for this transition
		nb_pos_AP = len(pos_AP)
		if nb_pos_AP > 1: # if more than one positive AP -> transition impossible
			pass
		elif nb_pos_AP == 1: # if exactly one positive AP -> keep this one only
			res.append(pos_AP[0])
		elif nb_pos_AP == 0: # if no positive AP -> keep all non negative AP
			for ap in list( set(alphabet) - set(neg_AP) ):
				res.append(ap)
	return res
