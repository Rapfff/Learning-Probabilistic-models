import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import re
from tools import randomProbabilities
from models.MCGT import *
import spot

formula = '(a U b) & GFc & GFd'

#STEP 1 : Translate to Buchi
a = spot.translate('(a U b) & GFc & GFd', 'Buchi', 'state-based').to_str("hoa")
print(a)
#STEP 2 : Transalte to MCGT
def HOAtoMCGT(hoa):
	hoa = hoa.split("\n")
	states = []
	i = 0
	while hoa[i][:6] != "Start:":
		i += 1
	initial_state = int(hoa[i].split(" ")[1])
	i += 1
	alphabet = hoa[i][4:].replace('"','').split(" ")[1:]
	print(alphabet)
	while hoa[i-1] != "--BODY--":
		i += 1
	while hoa[i] != "--END--":
		i += 1
		next_matrix = [[],[],[]]
		while hoa[i][0] == '[':
			hoa[i] = hoa[i].split(" ")
			dest_state = int(hoa[i][1])
			labels = _transitionLabels(hoa[i][0],alphabet)
			for l in labels:
				next_matrix[1].append(dest_state)
				next_matrix[2].append(l)
			i += 1
		next_matrix[0] = randomProbabilities(len(next_matrix[1]))
		states.append(MCGT_state(next_matrix))
	return MCGT(states,initial_state)

def _transitionLabels(l,alphabet):
	if l == "[t]":
		return alphabet
	l = l[1:-1]
	l = l.split('&')
	pos_AP = [alphabet[int(i)]     for i in l if i[0] != '!'] # list of all positive AP for this transition
	neg_AP = [alphabet[int(i[1:])] for i in l if i[0] == '!'] # list of all negative AP for this transition
	nb_pos_AP = len(pos_AP)
	if nb_pos_AP > 1: # if more than one positive AP -> transition impossible
		return []
	if nb_pos_AP == 1: # if exactly one positive AP -> keep this one only
		return pos_AP
	if nb_pos_AP == 0: # if no positive AP -> keep all non negative AP
		return list( set(alphabet) - set(neg_AP) )

initial_model = HOAtoMCGT(a)
initial_model.pprint()
#STEP 3 Learning
