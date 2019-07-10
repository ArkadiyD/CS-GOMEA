import sys
sys.path.insert(0, 'utils/')

import test_hyperopt
import subprocess
import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy
import time
import signal


parser = argparse.ArgumentParser(description='parse parameters')
parser.add_argument('PROBLEM_NUMBER', metavar='PROBLEM_NUMBER',
                    type=int, nargs=1, help='problem number')
parser.add_argument('DIMENSIONALITY', metavar='DIMENSIONALITY',
                    type=int, nargs=1, help='problem dimensionality')
parser.add_argument('FIRST_RUN', metavar='FIRST_RUN', type=int,
                    nargs=1, help='first run to execute')
parser.add_argument('N_RUNS', metavar='N_RUNS', type=int,
                    nargs=1, help='number of optimization runs')
parser.add_argument('N_EVALUATIONS', metavar='N_EVALUATIONS',
                    type=int, nargs=1, help='number of evaluations')

args = parser.parse_args()

PROBLEM_NUMBER = args.PROBLEM_NUMBER[0]
DIMENSIONALITY = args.DIMENSIONALITY[0]
FIRST_RUN = args.FIRST_RUN[0]
N_RUNS = args.N_RUNS[0]
FOLDER_NAME = str(PROBLEM_NUMBER) + '_' + str(DIMENSIONALITY)
N_EVALUATIONS = args.N_EVALUATIONS[0]

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)


def run_algorithm(call_, elitists_filename, prefix_results, run_count = 0):
   
    evaluations_by_run = []
    elitist_value_by_run = []
    call = deepcopy(call_)

    max_non_zero = -1

    for run in range(FIRST_RUN, N_RUNS):
        
        FOLDER_NAME_RUN = FOLDER_NAME + '/%d' % run
        if not os.path.exists(FOLDER_NAME_RUN):
            os.makedirs(FOLDER_NAME_RUN)

        call(DIMENSIONALITY, PROBLEM_NUMBER, N_EVALUATIONS, FOLDER_NAME_RUN)

        cur_run = pd.read_csv('%s/%s' % (FOLDER_NAME_RUN, elitists_filename), delimiter=' ')
        
        print cur_run
        evaluations_by_run.append(cur_run['Evaluations'].values)
        elitist_value_by_run.append(cur_run['Objective'].values)

        print evaluations_by_run
        print elitist_value_by_run

        array_evaluations = np.array(evaluations_by_run)
        array_elitist_value_by_run = np.array(elitist_value_by_run)      

        elitist_till_current_iteration = np.zeros(
            (N_EVALUATIONS * 2, array_evaluations.shape[0]))

        # iterate through different runs
        for i in range(array_evaluations.shape[0]):
            for j in range(array_evaluations[i].shape[0]):
                iter = array_evaluations[i][j]
                value = array_elitist_value_by_run[i][j]
                print iter, value
                elitist_till_current_iteration[iter][i] = value

            cur_value = 0
            for j in range(elitist_till_current_iteration.shape[0]):
                if elitist_till_current_iteration[j][i] != 0:
                    cur_value = elitist_till_current_iteration[j][i]
                    max_non_zero = max(max_non_zero, j)
                else:
                    elitist_till_current_iteration[j][i] = cur_value

        elitist_till_current_iteration = elitist_till_current_iteration[1:N_EVALUATIONS * 2]

        print elitist_till_current_iteration, '\n'
        np.savetxt('%s/%s_results_%d_%d.csv' % (FOLDER_NAME, prefix_results,
                                                PROBLEM_NUMBER, DIMENSIONALITY), elitist_till_current_iteration, delimiter=',')


run_algorithm(test_hyperopt.run_hyperopt, 'hyperopt_elitist_solutions.dat', 'hyperopt')
