import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import re
from tools import randomProbabilities
from models.MCGT import *

class TGBA_state:
	def __init__(self,f):
		self.f = f
		self.next = []

	def __str__(self):
		return "--"+str(self.f)+"--\n"+str(self.next)+'\n\n'

class TGBA:
	def __init__(self,f):
		self.states = []
		self.initial_state = [0]
		self.marked_forumlae = []
		self.expand([f],None)
		self.clean()
		#self.createTGBA()

	def __str__(self):
		res = ''
		for s in self.states:
			res += str(s)
		return res

	def addNewState(self,f):
		self.states.append(TGBA_state(f))
		return len(self.states)-1

	def removeState(self,i):
		self.states.remove(self.states[i])
		for s in self.states:
			for j in s.next:
				if j>i:
					j -= 1
				elif j == i:
					s.next.remove(j)

	def getNode(self,f):
		for n in range(len(self.states)):
			if self.states[n].f == f:
				return n
		return None

	def expand(self,f,previous):
		if self.getNode(f) != None:
			self.states[previous].next.append(self.getNode(f))
			return

		if previous != None:
			self.states[previous].next.append(len(self.states))
		index = self.addNewState(f)
		
		flag = False
		print("f:",f)
		for ff in f:
			print("ff:",ff)
			if not isElementary(ff) and not ff in self.marked_forumlae:
				flag = True
				self.marked_forumlae.append(ff)
				print("r:",r(ff))
				print()
				for son in r(ff):
					new_formula = [i for i in f if i != ff]+son
					self.expand(new_formula,index)
					
		if not flag:
			new_formula = []
			for ff in f:
				if ff[0] == 'X':
					if ff[1] == '(':
						new_formula.append(ff[2:-1])
					else:
						new_formula.append(ff[1:])
			self.expand(new_formula,index)

	def reachable(self,f,s):
		done = []
		todo = [s]
		while len(todo)>0:
			s = todo[0]
			todo.remove(s)
			done.append(s)
			for n in self.states[s].next:
				if not n in done:
					if f in self.states[n].f:
						return True
					if not n in todo:
						todo.append(n)
		return False


	def clean(self):
		s = 0
		while s < len(self.states): # remove impossible states
			state = self.states[s]
			for ff in [i for i in state.f if isElementary(i)]:
				if neg(ff) in state.f:
					self.removeState(s)
					s -= 1
					break
			s += 1
		
		s = 0 
		while s < len(self.states): #remove unreachable states
			flag = False
			for j in [i for i in range(len(self.states)) if i != s]:
				if s in self.states[j].next:
					flag = True
					break
			if not flag:
				self.removeState(s)
				s -= 1
			s += 1

		s = 0 
		while s < len(self.states): #remove unreachable states
			for ff in [i for i in self.states[s].f if i[0] == 'F' or ("U" in i and not i[0] != 'X')]:
				if "U" in ff:
					f = ff[ff.index('U')+1:]
				else:
					f = ff[1:]
				if not self.reachable(f,s):
					self.removeState(s)
					s -= 1
					break
			s += 1

	def createTGBA(self):
		new_states = [MCGT_state([[],[],[]])]
		done = [0]
		todo = self.states[0].next
		curr = 0
		while len(todo)>0:
			n = todo[0]
			todo.remove(n)
			done.append(n)
			transitions = []
			for i in self.states[n].f:
				if isElementary(i) and i[0] != 'X':
					transitions.append(i)
			if len(transitions) == 0:
				todo += [i for i in self.states[n].next if not i in todo and not i in done]
			else:
				for s in self.states[n].next:
					new_states.append(MCGT_state([[],[],[]]))
					for t in transitions:
						new_states[curr].next_matrix[0].append(0.0)
						new_states[curr].next_matrix[1].append()
						new_states[curr].next_matrix[2].append()
						#TO DO






def neg(f):
	if f[0] != '!':
		return '!'+f
	else:
		return f[1:]

def r(f):
	"""AP should NOT contain any capital letter"""
	if f[0] == 'X':
		return None
	if f == "!false":
		res =  [["true"]]
	elif f == "!true":
		res =  [["false"]]
	elif f[:2] == "!!":
		res =  [[f[2:]]]
	elif "|" in f:#										 OR
		print(1)
		res =  [[i] for i in f.split("|")]
	elif "&" in f:# 									 AND
		print(2)
		res =  [f.split("&")]
	elif re.search("!(.+&.+)",f): #						 NOT AND
		print(3)
		res =  [[neg(i)] for i in f[2:-1].split("&")]
	elif re.search("!(.+|.+)",f): #						 NOT OR
		print(4)
		res =  [[neg(i) for i in f[2:-1].split("|")]]
	elif f[:2] == "!X": # 								 NOT NEXT
		res =  [["X"+neg(f[2:])]]
	elif "U" in f: #									 UNTIL
		f2 = f[f.find("U")+len("U"):]
		f1 = f[:f.find("U")]
		res =  [[f2],[f1,"X("+f+")","P"+f2]]
	elif re.search("!(.+U.+)",f): # 					 NOT UNTIL
		f2 = f[f.find("U")+len("U"):]
		f1 = f[:f.find("U")]
		res =  [[neg(f1),neg(f2)],[neg(f2),"X("+f+")"]]
	elif f[0] == "F": # 								 FINALLY
		res =  [[f[1:],"XF"+f[1:]]]
	elif f[:2] == "!F": # 								 NOT FINALLY
		res =  [['!'+f[2:],"!XF"+f[2:]]]
	else:
		return None

	print(res)
	for i in range(len(res)):
		for j in range(len(res[i])):
			if res[i][j][0] == '(' and res[i][j][-1] == ')':
				res[i][j] = res[i][j][1:-1]
	return res


def isElementary(f):
	return r(f) == None or r(f) == ['true'] or r(f) == ['false']



formula = "(Xa)&(bU!a)"

print(TGBA(formula))
