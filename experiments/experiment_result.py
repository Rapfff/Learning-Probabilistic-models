import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
sys.path.append('../EM_and_BW_algorithms')

from numpy import mean, std

class ExperimentResult:
    '''
    A class used for the results of the experiments

    Attributes:
        output_folder:  Folder where the results of the experement is stored
            Includes both result_file and the best hypothesis model for each nb of states and model
        result_file:    Name of the text file which the results are stored
            FORMAT for each model:
                Model to learn: <Model name>
                Training_set: <int> sequences of <int> observations
                Testing_set: <int> sequences of <int> observations
                logLikelihood of original model: <float>
                Observation alphabet: <a list of strings>
                Learning algorithm <name of algorithm>, epsilon: <float>
                Hypothesis generator: <name of generator>
                For each test
                    <number of states>, <L(h, train)>, <L(h, test)>, <L(learn(h), test)>
        results:        A dictionary with the resut of the experiments
            FORMAT:
                { 
                    <model name 1>: 
                    {
                        'size_training_set': <int>, 
                        'len_training_set': <int>, 
                        'size_test_set': <int>, 
                        'len_test_set': <int>, 
                        'log_like_org': <float>, 
                        'data': 
                        {
                            <nb of states>: [(<L(h, trainset)>, <L(h, testset)>, <L(learn(h), testset)>) ... for all tests]
                            ... for all number of states
                        }
                    },
                    ... for all models
                }
    
    Methods:

        get_results()
            Fetches results from file

        get_best_model(model_name: string)
            Returns the best result model we got in this experiment for the original model <model name>
        
        get_data(model_name: string)
            Returns the data we gott from model <model name> in this experiment 
            FORMAT:
                {
                    <nb of states>: [(<L(h, trainset)>, <L(h, testset)>, <L(learn(h), testset)>) ... for all tests]
                    ... for all number of states
                }
        
        get_mean_results_model(model_name: string)
            Returns the mean and standard deviation of the differnce between 
            logLikelihood(model_name, testset) and logLikelihood(learn(h), testset) for model <model name>

        get_mean_logLikelihood_hypo_train(model_name: string)
            Returns the mean and standard deviation of logLikelihood(h, training_set)

        get_mean_logLikelihood_hypo_test(model_name: string)
            Returns the mean and standard deviation of logLikelihood(h, test_set)

        get_mean_results_nb(nb: int)
            Returns the mean and standard deviation of the differnce between 
            logLikelihood(model_name, testset) and logLikelihood(learn(h), testset) for each <nb> states
    
        get_mean_results(model_name: string, nb_states: int):
            Returns the mean and standard deviation of the differnce between 
            logLikelihood(model_name, testset) and logLikelihood(learn(h), testset) for model <model name> and <nb> states

        vs(other: ExperimentResult)
            Returns a list of cases when other gives worse results then self
        
        get_mean_results_all()
            Returns a list with mean result for all cases
    '''
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
            size_training_set, len_training_set= line.split(' ')[1], int(line.split(' ')[4])
            line = f.readline()
            size_test_set, len_test_set= int(line.split(' ')[1]), int(line.split(' ')[4])
            line = f.readline()
            log_like_org = float(line.split(' ')[4])
            f.readline() # alphabet
            f.readline() # algorithm
            f.readline() # emptyline

            line= f.readline()
            if model_name.strip() in self.results:
                data= self.results[model_name.strip()]['data']
            else:
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
    
    def get_data(self, model_name):
        return self.results[model_name]
    
    def get_mean_results_model(self, model_name):
        l3= list()
        for (k, l) in self.results[model_name]['data'].items():
            l3+= [abs(x[2]-self.results[model_name]['log_like_org']) for x in l]
        return mean(l3), std(l3)
    
    def get_mean_logLikelihood_hypo_train(self, model_name):
        l3= list()
        for (k, l) in self.results[model_name]['data'].items():
            l3+= [x[1] for x in l]
        return mean(l3), std(l3)
    
    def get_mean_logLikelihood_hypo_test(self, model_name):
        l3= list()
        for (k, l) in self.results[model_name]['data'].items():
            l3+= [x[2] for x in l]
        return mean(l3), std(l3)
    
    def get_mean_results_nb(self, nb):
        l3= list()
        for (k, l) in self.results.items():
            l3+= [abs(x[2]-self.results[k]['log_like_org']) for x in l['data'][str(nb)]]
        return mean(l3)
    
    def get_mean_results(self, model_name, nb_states):
        l3= [abs(x[2]-self.results[model_name]['log_like_org']) for x in self.results[model_name]['data'][str(nb_states)]]
        return mean(l3), std(l3)

    def vs(self, other):
        ret=list()
        for (key, value) in self.results.items():
            for (key_, value_) in value['data'].items():
                ls= [abs(x[2]-value['log_like_org']) for x in value_]
                lo= [abs(x[2]-other.results[key]['log_like_org']) for x in other.results[key]['data'][key_]]
                if mean(ls)<=mean(lo):
                    ret.append((key, key_))
        return ret
    
    def get_mean_results_all(self):
        l=list()
        for (_, value) in self.results.items():
            for (_, value_) in value['data'].items():
                ls= [abs(x[2]-value['log_like_org']) for x in value_]
                l+=[mean(ls),]

                
        return l
