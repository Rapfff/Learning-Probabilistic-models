import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from MDP import maxReachabilityScheduler, loadMDP
from Active_Learning_MDP import ActiveLearningScheduler
from examples.tictactoeMDP import TicTacToe
from statistics import mean

nb_games = 50
res1 = {"win": [], "loose" : [], "draw": []}
res2 = {"win": [], "loose" : [], "draw": []}

original = TicTacToe()

win_state_1 = 15
win_state_2 = 15

m1 = loadMDP("modelEM_1.txt")
m2 = loadMDP("modelEM_2.txt")
s1 = maxReachabilityScheduler(m1,win_state_1)
s1 = ActiveLearningScheduler(s1,m1)
s2 = maxReachabilityScheduler(m2,win_state_2)
s2 = ActiveLearningScheduler(s2,m2)

for g in range(nb_games):
	print(g)
	c = 0
	while c < 25:
		act = s1.get_action()
		obs = original.roundMove(act)
		s1.add_observation(obs)
		c += 1
		if obs in ["draw","loose","win"]:
			break
	if c == 25:
		res1["draw"].append(25)
	else:	
		res1[obs].append(len(s1.seq_act))
	original.reset()
	s1.reset()

for g in range(nb_games):
	print(g)
	c = 0
	while c < 25:
		act = s2.get_action()
		obs = original.roundMove(act)
		s2.add_observation(obs)
		c += 1
		if obs in ["draw","loose","win"] :
			break
	if c == 25:
		res2["draw"].append(25)
	else:	
		res2[obs].append(len(s2.seq_act))
	original.reset()
	s2.reset()

print("With model 1:")
print("---- Win  :",2*len(res1["win"]),"%,",mean(res1["win"]),"moves in average")
print("---- Loose:",2*len(res1["loose"]),"%,",mean(res1["loose"]),"moves in average")
print("---- Draw :",2*len(res1["draw"]),"%,",mean(res1["draw"]),"moves in average")
print()
print("With model 2:")
print("---- Win  :",2*len(res2["win"]),"%,",mean(res2["win"]),"moves in average")
print("---- Loose:",2*len(res2["loose"]),"%,",mean(res2["loose"]),"moves in average")
print("---- Draw :",2*len(res2["draw"]),"%,",mean(res2["draw"]),"moves in average")