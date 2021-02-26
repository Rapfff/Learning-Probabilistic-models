from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from examples.examples_models import modelMCGT_random
from statistics import stdev, mean

def str_to_date(s):
	s = s[1:s.rfind(',')-1]
	months_cumul_length_norm = {"Jan":0,
						   "Feb":31,
						   "Mar":59,
						   "Apr":90,
						   "May":120,
						   "Jun":151,
						   "Jul":181,
						   "Aug":212,
						   "Sep":243,
						   "Oct":273,
						   "Nov":304,
						   "Dec":334}
	months_cumul_length_bisext = {"Jan":0,
						   "Feb":31,
						   "Mar":60,
						   "Apr":91,
						   "May":121,
						   "Jun":152,
						   "Jul":183,
						   "Aug":213,
						   "Sep":244,
						   "Oct":274,
						   "Nov":305,
						   "Dec":335}
	months_cumul_length = [months_cumul_length_norm,months_cumul_length_bisext]

	return months_cumul_length[1 if s[-4:] == "2016" else 0][s[:3]]+int(s[4:6])

def toalphabet(line):
	s = float(line[line.rfind(',')+1:-2])
	return 0.25 + 0.5*(s//0.5)

def csv_to_list(file_path):
	res_seq = []
	res_val = []
	f = open(file_path,'r')

	line = f.readline()


	line = f.readline()
	week = [toalphabet(line)]
	previous_date = str_to_date(line)

	line = f.readline()
	while line:
		curr_date = str_to_date(line)
		if previous_date - curr_date == 1:
			week.append(toalphabet(line))
		
		elif len(week) == 5:
			if week in res_seq:
				res_val[res_seq.index(week)] += 1
			else:
				res_seq.append(week)
				res_val.append(1)
			week = [toalphabet(line)]
		
		else:
			week = [toalphabet(line)]

		previous_date = curr_date

		line = f.readline()

	f.close()
	return [res_seq,res_val]


training_set = csv_to_list("datasets/Oreal/oreal_2015_2018.csv")

alphabet = [-5.25+0.5*i for i in range(23)]
length_seq = 5
size_alphabet = len(alphabet)

#####################################################
matrix = [[0 for j in range(size_alphabet)] for i in range(size_alphabet+1) ]


for seq in range(len(training_set[0])):
	matrix[0][alphabet.index(training_set[0][seq][0])] += training_set[1][seq]
	for k in range(1,length_seq):
		matrix[alphabet.index(training_set[0][seq][k-1])+1][alphabet.index(training_set[0][seq][k])] += training_set[1][seq]

states = []

#for s in matrix:
#	states.append(MCGT_state([[p/sum(s) for p in s], list(range(1,size_alphabet+1)) , alphabet ]))
#
#m1 = MCGT(states,0,"initial_model")
#####################################################

money_agg = 0
have_it_agg = False
prix_achat_agg = 0

money_ultra_agg = 0
have_it_ultra_agg = False
prix_achat_ultra_agg = 0


f = open("datasets/Oreal/oreal_2018_2019_with_value.csv",'r')
l = f.readline()
l = f.readline()

while l:
	l = l.split(',')
	price = float(l[1])
	last_change = 100*float(l[2])
	last_change = min(last_change,5.75)
	last_change = max(last_change,-5.25)
	probas = matrix[1+alphabet.index(0.25 + 0.5*(last_change//0.5))]
	
	for i in range(len(alphabet)):
		probas[i] *= alphabet[i]
	
	if mean(probas) > 0 and not have_it_ultra_agg:
		money_ultra_agg -= price
		prix_achat_ultra_agg = price
		have_it_ultra_agg = True
	if mean(probas) < 0 and  have_it_ultra_agg:
		money_ultra_agg += price
		have_it_ultra_agg = False
		print("Ultra agg",price - prix_achat_ultra_agg)

	if mean(probas) > 0 and last_change < 0 and not have_it_agg:
		money_agg -= price
		prix_achat_agg = price
		have_it_agg = True
	if mean(probas) < 0 and last_change > 0 and have_it_agg and price > prix_achat_agg:
		money_agg += price
		have_it_agg = False
		print("Agg",price - prix_achat_agg)

	
	l = f.readline()

if have_it_agg:
	money_agg += price
	print("Agg",price - prix_achat_agg)
if have_it_ultra_agg:
	money_ultra_agg += price
	print("Ultra agg",price - prix_achat_ultra_agg)

print("Offensif      :",money_agg)
print("Ultra Offensif:",money_ultra_agg)
