import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def result_to_list(file_name):
    ret= dict()

    f = open(file_name,'r')
    i = 0
    for j in range(4*5):
        if j in {0, 5, 10, 15}:
            for i in range(6):
                print(f.readline())
        l=list()
        for i in range(100):
            line = f.readline()
            l_s= line.split(',')
            l.append((i, float(l_s[1]), float(l_s[2]), float(l_s[3])))
            i=i+1
        name='MCGT'+str(j//4+1)+'-'+str(j%5+2)
        ret[name]= l

    return ret

# result_to_list('experiments/test.txt')