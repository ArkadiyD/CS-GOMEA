import pickle
import time
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from objectives import *
import numpy as np
import csv

D = -1
function = ''
filename = ''
number_of_evaluations = 0

def objective(x):
    global number_of_evaluations

    csvfile = open(filename, 'a')
    writer = csv.writer(csvfile, delimiter=',')

    array = []
    for i in range(D):
        array.append(x['x%d' % i])
    array = np.array(array)
    value = function(array)    
    writer.writerow(array.tolist() +  [value])
    
    return {'loss': -value, 'status': STATUS_OK }

def create_output():
    data = np.loadtxt(filename, delimiter=',').astype(np.float32)
    new_data = []
    line = 1
    best_value = -10**9
    solutions = set([])
    print data

    for d in data:
        fitness = d[-1]
        solution = tuple(d[:-1].tolist())

        if fitness > best_value:
            best_value = fitness
            new_data.append((line, best_value))

        #print solutions, solution
        if solution not in solutions:
            line += 1
            solutions.add(tuple(solution))
        else:
            print 'repeated solution'

    new_data = np.array(new_data, dtype = np.float32)
    print new_data
    np.savetxt(filename, new_data, fmt = '%d %.5f', header = 'Evaluations Objective', comments='')

def optimize():
    space = {}
    for i in range(D):
        space['x%i' % i] = hp.choice('x%i' % i, [0,1])

    trials = Trials()
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=number_of_evaluations,
        trials=trials)


def run_hyperopt(D_, function_, number_of_evaluations_, folder_name):
    global D, function, filename, number_of_evaluations
    number_of_evaluations = number_of_evaluations_
    D = D_
    filename = '%s/hyperopt_elitist_solutions.dat' % (folder_name)
    print filename

    if function_ == 0:
        function = onemax
    elif function_ == 1:
        function = ktrap_tight_evaluation4
    elif function_ == 2:
        function = ktrap_loose_evaluation4
    elif function_ == 3:
        function = adf
    elif function_ == 4:
        function = hiff

    else:
        print 'function not implemented'
        exit(-1)

    csvfile = open(filename, 'w')
    csvfile.close()

    optimize()
    create_output() 

if __name__ == '__main__':
    run_hyperopt()