import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append('../EM_and_BW_algorithms')

# from src.models.MCGT import *
from numpy import mean, std
from test import countSwaps # TODO

class ExperimentResult:
    def __init__(self, output_folder = 'results', result_file= 'result'):
        self.output_folder = output_folder
        self.result_file=result_file
        self.results=dict()
        self.get_result()

    def get_result(self):
        f= open(self.output_folder+"/"+self.result_file+".txt", 'r')
        line= f.readline()

        while line:
            model_name = line.split(' ')[3]
            line = f.readline()
            size_training_set, len_training_set= int(line.split(' ')[1]), int(line.split(' ')[4])
            line = f.readline()
            size_test_set, len_test_set= int(line.split(' ')[1]), int(line.split(' ')[4])
            line = f.readline()
            log_like_org = float(line.split(' ')[4])
            f.readline() # alphabet
            f.readline() # algorithm
            f.readline() # emptyline

            line= f.readline()
            data= dict()
            while line[0:5]!= 'Model' and line:
                line= line.split(', ')
                if line[0] in data:
                    data[line[0]].append(tuple(map(float, line[1:4])))
                else:
                    data[line[0]]= [tuple(map(float, line[1:4])), ]
                line= f.readline()

            self.results[model_name.strip()]= {
                'size_training_set': size_training_set, 
                'len_training_set': len_training_set,
                'size_test_set': size_test_set, 
                'len_test_set': len_test_set,
                'log_like_org': log_like_org,
                'data': data,
            }
        
        f.close()

    def get_best_model(self, model_name, nb_states):
        return loadMCGT(self.output_folder+"/"+self.result_file+"_best_learrned_model_"+str(model_name)+"_"+str(nb_states)+".txt")
    
    def get_results(self, model_name):
        return self.results[model_name]
    
    def get_latex_table_1(self):
        print("""\\begin{table\}[]
            \\begin{tabular\}{|c|c|c|c|c|}
            \\hline
            model name & $\\#$states& Mean $L(h)$ (sd) & Mean $L(BW(h))$ (sd) & $|L(BW(h))-L(org)|$ (sd)\\\\
            \\hline""")
        for (key, value) in self.results.items():
            for (key_, value_) in value['data'].items():
                l1= [x[1] for x in value_]
                l2= [x[2] for x in value_]
                l3= [abs(x[2]-value['log_like_org']) for x in value_]

                print(key+ ' & '+key_+ ' & '+str(round(mean(l1),3))+' ('+str(round(std(l1),3))+') & '+str(round(mean(l2),3))+' ('+str(round(std(l2),3))+') & '+str(round(mean(l3),3))+' ('+str(round(std(l3),3))+')\\\\')
                print('\\hline')
        print("""\\end{tabular\}
            \\caption{Experiment 0\}
            \\label{tab:my_label\}
            \\end{table\}""")

    def get_latex_table_2(self):
        for (key, value) in self.results.items():
            for (key_, value_) in value['data'].items():
                l=value_
                l.sort(key= lambda x:x[0])
                l = [(i, l[i][0], l[i][2]) for i in range(len(l))]
                l.sort(key= lambda x:x[2])
                nf= countSwaps([x[0] for x in l], len(l))
                all =100*99//2
                print(key+ ' & '+key_+ ' & '+str(all-nf)+'&'+str((all-nf)/all)+"\\\\")
                print('\\hline')

    