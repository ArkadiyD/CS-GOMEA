import subprocess
import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy
import time


parser = argparse.ArgumentParser(description='parse parameters')
parser.add_argument('PROBLEM_NUMBER', metavar='PROBLEM_NUMBER',
                    type=int, nargs=1, help='problem number')
parser.add_argument('DIMENSIONALITY', metavar='DIMENSIONALITY',
                    type=int, nargs=1, help='problem dimensionality')
parser.add_argument('FOS', metavar='FOS_TYPE', type=int,
                    nargs=1, help='GOMEA FOS type')
parser.add_argument('MAX_EVALUATIONS', metavar='MAX_EVALUATIONS',
                    type=int, nargs=1, help='maximum number of evaluations')
parser.add_argument('DELTA', metavar='DELTA',
                    type=float, nargs=1, help='DELTA')
parser.add_argument('WARMUP_PERIOD', metavar='WARMUP_PERIOD',
                    type=int, nargs=1, help='warm-up period')
parser.add_argument('FIRST_RUN', metavar='FIRST_RUN', type=int,
                    nargs=1, help='first run to execute')
parser.add_argument('N_RUNS', metavar='N_RUNS', type=int,
                    nargs=1, help='number of optimization runs')
parser.add_argument('DEVICE_ID', metavar='DEVICE_ID',
                    type=int, nargs=1, help='gpu device id')
parser.add_argument('TIME', metavar='TIME',
                    type=int, nargs=1, help='max time in minutes')

args = parser.parse_args()

PROBLEM_NUMBER = args.PROBLEM_NUMBER[0]
DIMENSIONALITY = args.DIMENSIONALITY[0]
FOS = args.FOS[0]
MAX_EVALUATIONS = args.MAX_EVALUATIONS[0]
DELTA = args.DELTA[0]
WARMUP_PERIOD = args.WARMUP_PERIOD[0]
FIRST_RUN = args.FIRST_RUN[0]
N_RUNS = args.N_RUNS[0]
FOLDER_NAME = str(PROBLEM_NUMBER) + '_' + str(DIMENSIONALITY)
DEVICE_ID = args.DEVICE_ID[0]
TIME = args.TIME[0]

if not os.path.exists(FOLDER_NAME):
    os.makedirs(FOLDER_NAME)


def write_vtr(FOLDER_NAME_RUN):
    f = open('%s/vtr.txt' % FOLDER_NAME_RUN, 'w')

    if PROBLEM_NUMBER == 0 or PROBLEM_NUMBER == 11 or PROBLEM_NUMBER == 7:
        vtr = DIMENSIONALITY
    elif PROBLEM_NUMBER == 1 or PROBLEM_NUMBER == 2:
        vtr = DIMENSIONALITY // 4
    elif PROBLEM_NUMBER == 3 or PROBLEM_NUMBER == 4:
        vtr = DIMENSIONALITY // 5
    elif PROBLEM_NUMBER == 9:
        vtr = DIMENSIONALITY // 3
    elif PROBLEM_NUMBER == 6:
        if DIMENSIONALITY == 12:
            vtr = 127
        elif DIMENSIONALITY == 25:
            vtr = 530
        elif DIMENSIONALITY == 50:
            vtr = 2050
    elif PROBLEM_NUMBER == 8:
        if DIMENSIONALITY == 12:
            vtr = 52
        if DIMENSIONALITY == 25:
            vtr = 143
        if DIMENSIONALITY == 50:
            vtr = 336
        elif DIMENSIONALITY == 75:
            vtr = 611
        elif DIMENSIONALITY == 100:
            vtr = 772
        elif DIMENSIONALITY == 200:
            vtr = 1774
        elif DIMENSIONALITY == 8:
            vtr = 32
        elif DIMENSIONALITY == 16:
            vtr = 80
        elif DIMENSIONALITY == 32:
            vtr = 192
        elif DIMENSIONALITY == 64:
            vtr = 448
        elif DIMENSIONALITY == 128:
            vtr = 1024
        elif DIMENSIONALITY == 256:
            vtr = 2304

    elif PROBLEM_NUMBER == 12:
        vtr = int(DIMENSIONALITY) // 2.0 - 1e-6
    elif PROBLEM_NUMBER == 10:
        vtr = (DIMENSIONALITY // 4) * 15
    elif PROBLEM_NUMBER == 5:
        if DIMENSIONALITY == 25:
            vtr = 15.779790
        if DIMENSIONALITY == 50:
            vtr = 35.591645
    else:
        vtr = 10000

    vtr -= 1e-5

    f.write(str(vtr))
    f.close()


def run_gomea(call_, elitists_filename, prefix_results, run_count=0):

    evaluations_by_run = []
    elitist_value_by_run = []
    call = deepcopy(call_)

    max_non_zero = -1

    for run in range(FIRST_RUN, N_RUNS):

        FOLDER_NAME_RUN = FOLDER_NAME + '/%d' % run
        if not os.path.exists(FOLDER_NAME_RUN):
            os.makedirs(FOLDER_NAME_RUN)

        write_vtr(FOLDER_NAME_RUN)

        start_time = time.time()

        if isinstance(call, list):
            if len(call) > 10:  # surrogate gomea call
                call[-2] = call_[-2] + '/%d' % run
            else:  # vanilla gomea call
                call[-1] = call_[-1] + '/%d' % run
            print call
            subprocess.call(call)
        else:
            call(DIMENSIONALITY, PROBLEM_NUMBER, min(
                200, MAX_EVALUATIONS_INCREASED), FOLDER_NAME_RUN)

        time_for_run = int(time.time() - start_time)
        time_arr = np.array([time_for_run], dtype=np.int32)
        np.savetxt('%s/%s_time.dat' %
                   (FOLDER_NAME_RUN, prefix_results), time_arr)

        cur_run = pd.read_csv(
            '%s/%s' % (FOLDER_NAME_RUN, elitists_filename), delimiter=' ')
        
        print cur_run
        evaluations_by_run.append(cur_run['Evaluations'].values)
        elitist_value_by_run.append(cur_run['Objective'].values)

        print evaluations_by_run
        print elitist_value_by_run

        array_evaluations = np.array(evaluations_by_run)
        array_elitist_value_by_run = np.array(elitist_value_by_run)

        elitist_till_current_iteration = np.zeros(
            (MAX_EVALUATIONS, array_evaluations.shape[0]))

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

        elitist_till_current_iteration = elitist_till_current_iteration

        print elitist_till_current_iteration, '\n'
        np.savetxt('%s/%s_results_%d_%d.csv' % (FOLDER_NAME, prefix_results,
                                                PROBLEM_NUMBER, DIMENSIONALITY), elitist_till_current_iteration, delimiter=',')


time_limit = TIME * 60 * 1000

cs_gomea_call = ['./CS_GOMEA', '-v', str(PROBLEM_NUMBER), str(DIMENSIONALITY), str(
    FOS), str(MAX_EVALUATIONS), str(time_limit), str(DELTA), str(WARMUP_PERIOD), '0', FOLDER_NAME, str(DEVICE_ID)]
run_gomea(cs_gomea_call, 'elitist_solutions.dat', 'cs')