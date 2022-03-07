import re


class TGBA_state:
	def __init__(self,f):
		self.f = f
		self.next = []

class TGBA:
	def __init__(self):
		self.states = []
		self.initial_state = []

	def addState(s):
		self.states.append(s)

def neg(f):
	if f[0] != '!':
		return '!'+f
	else:
		return f[1:]

def r(f):
	if f == "!false":
		return ["true"]
	elif f == "!true":
		return ["false"]
	elif f[:2] == "!!":
		return [f[2:]]
	elif "|" in f:
		return f.split("|")
	elif "&" in f:
		return [f.split("&")]
	elif re.search("!(.+&.+)",f):
		return [neg(i) for i in f[2:-1].split("&")]
	elif re.search("!(.+|.+)",f):
		return [[neg(i) for i in f[2:-1].split("|")]]
	elif f[:2] == "!X":
		return "X"+neg(f[2:])
	elif "U" in f:
		f2 = f[f.find("U")+len("U"):]
		f1 = f[:f.find("U")]
		return [f2,[f1,"X("+f+")","P"+f2]]
	elif re.search("!(.+U.+)",f):
		f2 = f[f.find("U")+len("U"):]
		f1 = f[:f.find("U")]
		return [[neg(f1),neg(f2)],[neg(f2),"X("+f+")"]]
		
	else:
		return f

def LTLtoTGBA(f):
	

formula = "f U g"

