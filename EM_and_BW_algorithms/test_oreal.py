from Estimation_algorithms_MCGT_multiple import Estimation_algorithm_MCGT
from examples.examples_models import modelMCGT_random

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
test_set = csv_to_list("datasets/Oreal/oreal_2018_2019.csv")

alphabet = [-5.25+0.5*i for i in range(23)]


model = modelMCGT_random(10,alphabet)

algo = Estimation_algorithm_MCGT(model,alphabet)

final_loglikelihood, running_time = algo.problem3(training_set)
output_model = algo.h
output_model.save('orealModel.txt')
