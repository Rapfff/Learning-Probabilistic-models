from random import random

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

def correct_proba(ll,times=1):
	diff = sum(ll)-1.0
	res =  [round(i-diff/len(ll),5) for i in ll]
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
