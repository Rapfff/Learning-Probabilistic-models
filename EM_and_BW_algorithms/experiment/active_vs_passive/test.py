import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir1 = os.path.dirname(currentdir)
parentdir2 = os.path.dirname(parentdir1)
sys.path.append(parentdir2)

from examples.examples_runner import *
from examples.exmaples_models import modelMDP1_fullyobservable, scheduler_uniform
from src.tools import generateSet

tr_size = 50
tr_len  = 10
nb_sta = 7
nb_seq = 10
nb_it = 20
output_folder = "results"
kind_model = MDP
algorithm = "Active MDP-BW"


original = modelMDP1_fullyobservable()
training_set = generateSet(original,tr_size,tr_len,scheduler_uniform(original.actions()))

run_experiment(training_set,output_folder,kind_model,algorithm,model=original,nb_states=nb_sta,epsilon=0.01,df=0.9,lr=0,nb_sequences=nb_seq,nb_iteration=nb_it)