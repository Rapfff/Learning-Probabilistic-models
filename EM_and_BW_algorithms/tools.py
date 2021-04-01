from random import random, randint
from fractions import gcd
from functools import reduce

def find_gcd(ll):
	x = reduce(gcd, ll)
	return x

def resolveRandom(m):
	"""
	m = [proba1,proba2,...]
	return index of the probability choosen
	"""
	r = random()
	i = 0
	while r > sum(m[:i+1]):
		i += 1
	return i

def correct_proba(ll,accuracy = 5,times=1):
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
		if times == 900:
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

def generateSet(model,set_size,sequence_size,scheduler=None,with_action=True):
	seq = []
	val = []
	for i in range(set_size):
		if scheduler and with_action:
			trace = model.run(sequence_size,scheduler,True)
		elif scheduler:
			trace = model.run(sequence_size,scheduler,False)
		else:
			trace = model.run(sequence_size)

		if not trace in seq:
			seq.append(trace)
			val.append(0)

		val[seq.index(trace)] += 1

	return [seq,val]

def randomLTL(depth, width, alphabet):
	"""
	Generate a random LTL formula given the depth, the width and the set of atomic prop (here set of observations).
	We need to adapt it to the uppaal format.
	"""
	h = randint(1,width)
	a = [ [alphabet[i]] for i in range(h)]
	if depth == 0:
		return a
	else:
		res = []
		b = randomLTL(depth-1,width,alphabet)
		for i in a:
			for j in b:
				res.append(i+j)
		return res

