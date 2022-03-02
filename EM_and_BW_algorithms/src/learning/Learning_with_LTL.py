class Node:
	def __init__(self,incoming=[],now=[],nnext=[]):
		self.incoming = incoming
		self.now = now
		self.next = nnext
		self.labels = []
		self.next_nodes = []

def isLiteral(f):
	if not " U " in f and not " R " in f and not " or " in f and not " and " in f and not "X " in f:
		return True
	return False

def neg(f):
	if f[0] == '!':
		return f[1:]
	return'!'+f

def expand(curr, old, nnext, incoming):
	"""	LTLSet:   curr
		LTLSet:   old
		LTLSet:   nnext
		list int: incoming
	"""
	if len(curr) == 0:
		flag = True
		for n in nodes:
			if n.next == nnext and n.now == old:
				n.incoming += incoming
				flag = False
				#break
		if flag:
			nodes.append(Node(incoming,old,nnext))
			expand(nnext,[],[],[len(nodes)-1])
	
	else:
		while len(curr)>0:
			f = curr[0]
			curr = curr[1:]
			old.append(f)
			
			if isLiteral(f):
				if not f in literals:
					literals.append(f)
				if f !="false" and not neg(f) in old:
					expand(curr, old, nnext, incoming)
			
			elif " and " in f:
				to_add = [i for i in f.split(" and ") if not i in old]
				expand(curr+to_add, old, nnext, incoming)
			
			elif "X " in f:
				expand(curr, old, nnext+[f[2:]], incoming)
			
			elif " or " in f or " U " in f or " R " in f:
				if " U " in f:
					curr1 = [f.split(" U ")[0]] if f.split(" U ")[0] not in old else []
					curr2 = [f.split(" U ")[1]] if f.split(" U ")[1] not in old else []
					next1 = [f]
				elif " R " in f:
					curr1 = [f.split(" R ")[1]] if f.split(" R ")[1] not in old else []
					curr2 = [i for i in f.split(" R ") if not i in old]
					next1 = [f]
				else: # "or" case
					curr1 = [f.split(" or ")[1]] if f.split(" or ")[1] not in old else []
					curr2 = [f.split(" or ")[0]] if f.split(" or ")[0] not in old else []
					next1 = []

				expand(curr+curr1, old, nnext+next1, incoming)
				expand(curr+curr2, old, nnext, incoming)

def LGBA_construct():
	for n_index in range(len(nodes)):
		n = nodes[n_index]
		for a in literals:
			if (neg(a) not in n.now) and a in n.now:
				n.labels.append(a)
		for nprime in n.incoming:
			nodes[nprime].next_nodes.append(n_index)
	initial_states = nodes[0].next_nodes
	return initial_states

nodes = [Node()]
literals = ["true","false"]
formula = "f U g"
expand([formula],[],[],[0])
initial_states = LGBA_construct()

print(initial_states)
for i in nodes:
	print("\n",i.labels)
	print(i.next_nodes)


