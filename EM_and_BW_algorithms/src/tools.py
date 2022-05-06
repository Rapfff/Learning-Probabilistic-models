from random import random
from numpy.random import geometric
from scipy.stats import norm
from ast import literal_eval


def normpdf(x: float, params: list, variation:float = 0.01) -> float:
	"""
	Return the probability of <x> under a normal distribution returns of 
	parameters <params>. Since this probability should be 0 it returns in
	fact the probability that the normal distribution gives us a value
	between x-variation and x+variation.

	:param x: the value of the normal distribution
	:type x: float

	:param params: the parameters of the distribution: [mean,sd]
	:type params: list of two float

	:param variation: the vicinity
	:type variation: float

	:return: the probability of <x> under a normal distribution returns of 
	parameters <params>.
	:rtype: float
	"""
	return norm.cdf(x+variation,params[0],params[1]) - norm.cdf(x-variation,params[0],params[1])


def loadSet(file_path:str , float_obs: bool = False) -> list:
	"""
	Load a training/test set saved into a text file.

	:param file_path: location of the file
	:type file_path: str

	:param float_obs: Should be True if the observations are float (False by default)
	:type float_obs: bool

	:return: a training/test set
	:rtype: float
	"""
	res_set = [[],[]]
	f = open(file_path,'r')
	l = f.readline()
	while l:
		res_set[0].append(literal_eval(l[:-1]))
		l = f.readline()
		res_set[1].append(int(l[:-1]))
		l = f.readline()
	f.close()
	return res_set

def saveSet(t_set: list, file_path: str) -> None:
	"""
	Save a training/test set into a text file.
	
	:param t_set: the training/test set to save
	:type t_set: list

	:param file_path: where to save
	:type file_path: str
	"""
	f = open(file_path,'w')
	for i in range(len(t_set[0])):
		f.write(str(t_set[0][i])+'\n')
		f.write(str(t_set[1][i])+'\n')
	f.close()

def resolveRandom(m: list) -> int:
	"""
	Given a list of probabilities it returns the index of the one choosen
	according to the probabilities.
	Example: if m=[0.7,0.3], it will returns 0 with probability 0.7, etc...
	
	:param m: list of probabilities
	:type m: list of float

	:return: an index
	:rtype: int
	"""
	while True:
		r = random()
		i = 0
		while r > sum(m[:i+1]) and i < len(m):
			i += 1
		if i < len(m):
			break
	return i

def correct_proba(ll: list,accuracy:int = 6,times:int =1) -> list:
	"""
	Given a list of float it adapts it such that the sum is equal to 1.0.

	:param ll: list of probabilities to normalize
	:type: list of float

	"""
	#return [i/sum(ll) for i in ll]
	
	diff = sum(ll)-1.0
	res =  [round(i-diff/len(ll),accuracy) for i in ll]
	f = False
	for i in range(len(res)):
		if res[i]>1.0:
			res[i] = 1.0
			f = True
		if res[i]<0.0:
			res[i] = 0.0
			f = True
	if f:
		if times >= 600:
			return res
		else:
			return correct_proba(res,times + 1)
	else:
		return res

def randomProbabilities(size):
	"""return of list l of length <size> of probailities s.t. sum(l) = 1.0"""
	rand = []
	for i in range(size-1):
		rand.append(random())
	rand.sort()
	rand.insert(0,0.0)
	rand.append(1.0)
	return [rand[i]-rand[i-1] for i in range(1,len(rand))]

def mergeSets(s1,s2):
	for i in range(len(s2[0])):
		if not s2[0][i] in s1[0]:
			s1[0].append(s2[0][i])
			s1[1].append(s2[1][i])
		else:
			s1[1][s1[0].index(s2[0][i])] += s2[1][i]
	return s1

def generateSet(model,set_size,param,scheduler=None,distribution=None,min_size=None,timed=False):
	"""
	If distribution=='geo' then the sequence length will be distributed by a geometric law 
	such that the expected length is min_size+(1/param).
	if distribution==None param can be an int, in this case all the seq will have the same len (param),
					   or param can be a list of int
	"""
	seq = []
	val = []
	for i in range(set_size):
		if distribution == 'geo':
			curr_size = min_size + int(geometric(param))
		else:
			if type(param) == list:
				curr_size = param[i]
			elif type(param) == int:
				curr_size = param

		if scheduler:
			trace = model.run(curr_size,scheduler)
		elif timed:
			trace = model.run(curr_size,timed)
		else:
			trace = model.run(curr_size)


		if not trace in seq:
			seq.append(trace)
			val.append(0)

		val[seq.index(trace)] += 1

	return [seq,val]

def generateSetUnique(model,set_size,sequence_size,scheduler=None):
	seq = []
	while len(seq) < set_size:
		if scheduler:
			trace = model.run(sequence_size,scheduler)
		else:
			trace = model.run(sequence_size)

		if not trace in seq:
			seq.append(trace)
	return seq


def getAlphabetFromSequences(sequences):
	sequences = sequences[0]
	if type(sequences) == str:
		return list(set(sequences))
	else:
		observations = []
		for seq in range(len(sequences)):
			sequence_obs = sequences[seq]
			for x in sequence_obs:
				if x not in observations:
					observations.append(x)
		return observations

def getActionsObservationsFromSequences(sequences):
	sequences = sequences[0]
	actions = []
	observations = []
	for seq in range(len(sequences)):
		sequence_actions = [sequences[seq][i] for i in range(0,len(sequences[seq]),2)]
		sequence_obs = [sequences[seq][i+1] for i in range(0,len(sequences[seq]),2)]
		for x in sequence_actions:
			if x not in actions:
				actions.append(x)
		for x in sequence_obs:
			if x not in observations:
				observations.append(x)

	return [actions,observations]

def setFromList(l):
	res = [[],[]]
	for s in l:
		s = list(s)
		if s not in res[0]:
			res[0].append(s)
			res[1].append(0)
		res[1][res[0].index(s)] += 1
	return res