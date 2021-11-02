import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import random
from EM_and_BW_algorithms.src.tools import loadSet, saveSet

def split_set(file, p):
    '''Randomly splits dataset from file to two datasets'''
    ds= loadSet(file)
    nl=list()
    for i in range(len(ds[0])):
        nl+=[tuple(ds[0][i]) for _ in range(ds[1][i])]
    random.shuffle(nl)
    l= sum(ds[1])
    lp= int(l*p)
    l1=nl[:lp]
    l2=nl[lp:]
    l10=list(set(l1))
    l11=[l1.count(l10[i]) for i in range(len(l10))]
    l20=list(set(l2))
    l21=[l2.count(l20[i]) for i in range(len(l20))] 
    return [l10, l11], [l20, l21]

# datasets= {
#     'MCGT_games': {'trainingset': 'datasets/MCGT_gamestrainingset.txt', 'testset': 'datasets/MCGT_gamestestset.txt'}, 
#     'MCGT_REBER': {'trainingset': 'datasets/MCGT_REBERtrainingset.txt', 'testset': 'datasets/MCGT_REBERtestset.txt'}, 
#     'MCGT5': {'trainingset': 'datasets/MCGT5trainingset.txt', 'testset': 'datasets/MCGT5testset.txt'},
#     'MCGT6': {'trainingset': 'datasets/MCGT6trainingset.txt', 'testset': 'datasets/MCGT6testset.txt'},
#     'MCGT7': {'trainingset': 'datasets/MCGT7trainingset.txt', 'testset': 'datasets/MCGT7testset.txt'},
#     'MCGT8': {'trainingset': 'datasets/MCGT8trainingset.txt', 'testset': 'datasets/MCGT8testset.txt'},
#     'MCGT9': {'trainingset': 'datasets/MCGT9trainingset.txt', 'testset': 'datasets/MCGT9testset.txt'},
#     'MCGT10': {'trainingset': 'datasets/MCGT10trainingset.txt', 'testset': 'datasets/MCGT10testset.txt'},
#     'MCGT11': {'trainingset': 'datasets/MCGT11trainingset.txt', 'testset': 'datasets/MCGT11testset.txt'},
#     'MCGT12': {'trainingset': 'datasets/MCGT12trainingset.txt', 'testset': 'datasets/MCGT12testset.txt'},
#     }
# datasets_split_50= dict()
# for key in datasets.keys():
#     l1, l2 = split_set(datasets[key]['trainingset'], 0.2)
#     p= 'datasets/split_20/'
#     saveSet(l1, p+key+'trainingset1.txt')
#     saveSet(l2, p+key+'trainingset2.txt')
#     datasets_split_50[key]={'trainingset1': p+key+'trainingset1.txt', 'trainingset2': p+key+'trainingset2.txt'}
# print(datasets_split_50)
