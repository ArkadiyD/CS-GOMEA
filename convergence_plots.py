import math
import subprocess
import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import matplotlib.ticker as mticker

from matplotlib import rcParams
rcParams['figure.dpi'] = 300

folder_name = 'plots'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

parser = argparse.ArgumentParser(description='parse parameters')
parser.add_argument('PROBLEM_NUMBER', metavar='PROBLEM_NUMBER',
                    type=int, nargs=1, help='problem number')
parser.add_argument('FIRST_RUN', metavar='FIRST_RUN', type=int,
                    nargs=1, help='first run to execute')
parser.add_argument('N_RUNS', metavar='N_RUNS', type=int,
                    nargs=1, help='number of optimization runs')
parser.add_argument('SMAC_HYPEROPT', metavar='SMAC_HYPEROPT',
                    type=int, nargs=1, help='whether to run SMAC and Hyperopt')

args = parser.parse_args()

PROBLEM_NUMBER = args.PROBLEM_NUMBER[0]
FIRST_RUN = args.FIRST_RUN[0]
N_RUNS = args.N_RUNS[0]
SMAC_HYPEROPT = args.SMAC_HYPEROPT[0]

def plot_mean_and_CI(mean, lb, ub, color_mean=None, linestyle=None, axis=None, DIM=1, dashes = None, set_y = False):
    x = np.arange(mean.shape[0])+1
    y = mean
    if dashes == None:
        line, = axis.plot(np.log10(x), y, color_mean, linestyle=linestyle)
    else:
        line, = axis.plot(np.log10(x), y, color_mean, linestyle=linestyle, dashes = dashes)
        
    max = math.ceil(np.max(np.log10(x)))+1
    axis.set_xticks(np.arange(max))
    axis.set_xticklabels(10**np.arange(max))
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    f.set_powerlimits((-100, 100))
    g = lambda x,pos : "${}$".format(f._formatSciNotation('%.10e' % (10**x))) 
    axis.xaxis.set_major_formatter(mticker.FuncFormatter(g))

    yticks = np.linspace(np.min(mean), 1, 10)
    yticks = np.round(yticks, 1)
    axis.set_yticks(yticks)

    if set_y:    
        axis.set_yticklabels(yticks)
    else:
        axis.set_yticklabels([])

    axis.set_title(r'$l=%d$' % DIM)
    axis.grid(True)
    return line


def get_data(elitists_filename, prefix_results, run_count=0, DIMENSIONALITY=1, evals=1000):

    evaluations_by_run = []
    elitist_value_by_run = []

    FOLDER_NAME = str(PROBLEM_NUMBER) + '_' + str(DIMENSIONALITY)

    vtr = np.loadtxt('%s/0/vtr.txt' % (FOLDER_NAME)).astype(np.float32)

    for run in range(FIRST_RUN, N_RUNS):

        FOLDER_NAME_RUN = FOLDER_NAME + '/%d' % run

        cur_run = pd.read_csv(
            '%s/%s' % (FOLDER_NAME_RUN, elitists_filename), delimiter=' ')
        
        evaluations_by_run.append(cur_run['Evaluations'].values)
        elitist_value_by_run.append(cur_run['Objective'].values)

        array_evaluations = np.array(evaluations_by_run)
        array_elitist_value_by_run = np.array(elitist_value_by_run)

        elitist_till_current_iteration = np.zeros(
            (min(300000,evals*50), array_evaluations.shape[0]))

    hits, no_hits = 0,0
        
    for i in range(array_evaluations.shape[0]):
        hit = 0
        for j in range(array_evaluations[i].shape[0]):
            iter = array_evaluations[i][j]
            value = array_elitist_value_by_run[i][j] / vtr
           
            if value >= 1:
                hit = 1

            elitist_till_current_iteration[iter][i] = value

        if hit:
            hits += 1
        else:
            no_hits += 1

        cur_value = 0
        for j in range(elitist_till_current_iteration.shape[0]):
            if elitist_till_current_iteration[j][i] != 0:
                cur_value = elitist_till_current_iteration[j][i]
            else:
                elitist_till_current_iteration[j][i] = cur_value

    elitist_till_current_iteration = elitist_till_current_iteration[1:min(300000,evals*50)]

    np.savetxt('%s/%s_results_%d_%d.csv' % (FOLDER_NAME, prefix_results,
                                            PROBLEM_NUMBER, DIMENSIONALITY), np.round(elitist_till_current_iteration * vtr, 3), delimiter=',')

    mean = np.mean(elitist_till_current_iteration[:evals], axis=1)
    ub = np.max(elitist_till_current_iteration[:evals], axis=1)
    lb = np.min(elitist_till_current_iteration[:evals], axis=1)

    print elitists_filename, 'D = %d' % int(DIMENSIONALITY), 'Optimum achieved in %.2f percent runs' % (100*float(hits) / (hits + no_hits))

    return mean, lb, ub


if SMAC_HYPEROPT:
    colors = ['blue', 'red', 'cyan', 'green']
else:
    colors = ['blue', 'red']


if PROBLEM_NUMBER in [1,2,4]:
    DIMS = [8,16,32,64,128, 256]
    SMAC_DIMS = [8, 16]
elif PROBLEM_NUMBER in [0]:
    DIMS = [25, 50, 100,200, 400]
    SMAC_DIMS = [25]
elif PROBLEM_NUMBER in [3]:
    DIMS = [25, 50]
    SMAC_DIMS = [25]

fig, ax = plt.subplots(1, len(DIMS), figsize=(15, 2.75))

PROBLEM_NAMES = ['Onemax',
                 'Trap4-Tight',
                 'Trap4-Loose',
                 'NK-S1',
                 'HIFF']


for i in range(len(DIMS)):

    if PROBLEM_NUMBER == 3:
        if i == 0:
            evals = 50000
        elif i == 1:
            evals = 100000
    else:
        if PROBLEM_NUMBER == 1 or PROBLEM_NUMBER == 2:
            if i == 0 or i == 1 or i == 2 or i == 3:
                evals = 1000
            elif i == 4:
                evals = 5000
            elif i == 5:
                evals = 10000
        elif PROBLEM_NUMBER == 4:
            if i == 0 or i == 1 or i == 2 :
                evals = 1000
            elif i == 3:
                evals = 4000
            elif i == 4:
                evals = 6000
            elif i == 5:
                evals = 10000
        else:
            evals = 1000

    cnt = 0

    mean, ub, lb = get_data('elitist_solutions.dat',
                             'surrogate', DIMENSIONALITY=DIMS[i], evals=evals)
    line1 = plot_mean_and_CI(
        mean, lb, ub, color_mean=colors[cnt], linestyle='-', axis=ax[i], DIM=DIMS[i], set_y = (i == 0))
    cnt += 1

    mean, ub, lb = get_data(
        'vanilla_elitist_solutions.dat', 'vanilla', DIMENSIONALITY=DIMS[i], evals=evals)
    line2 = plot_mean_and_CI(
        mean, lb, ub, color_mean=colors[cnt], linestyle='-.', axis=ax[i], DIM=DIMS[i], set_y = (i == 0))
    cnt += 1

    if SMAC_HYPEROPT and DIMS[i] in SMAC_DIMS:
        mean, ub, lb = get_data(
            'smac_elitist_solutions.dat', 'smac', DIMENSIONALITY=DIMS[i], evals=evals)
        line3 = plot_mean_and_CI(
            mean, lb, ub, color_mean=colors[cnt], linestyle=':', axis=ax[i], DIM=DIMS[i], set_y = (i == 0))
        cnt += 1

        mean, ub, lb = get_data(
            'hyperopt_elitist_solutions.dat', 'hyperopt', DIMENSIONALITY=DIMS[i], evals=evals)
        line4 = plot_mean_and_CI(
            mean, lb, ub, color_mean=colors[cnt], linestyle='-', axis=ax[i], DIM=DIMS[i], dashes=[6, 2], set_y = (i == 0))
        cnt += 1

if PROBLEM_NUMBER == 5:
    plt.subplots_adjust(left=0.1, bottom=0.25,
                        right=0.95, top=0.85, wspace=0.2)
    
    if SMAC_HYPEROPT:
        fig.legend([line1, line2, line3, line4], ['CS-GOMEA', 'GOMEA', 'SMAC', 'Hyperopt'],
                   loc='lower center', bbox_to_anchor=(0.5, 0.01), fancybox=False, borderaxespad=0., ncol=5, frameon=False)

    else:
        fig.legend([line1, line2], ['CS-GOMEA', 'GOMEA'],
                   loc='lower center', bbox_to_anchor=(0.5, 0.01), fancybox=False, borderaxespad=0., ncol=3, frameon=False)

    fig.text(
        0.5, 0.94, PROBLEM_NAMES[PROBLEM_NUMBER], ha='center', fontsize=12)
    fig.text(0.5, 0.1, 'Number of evaluations', ha='center', fontsize=12)
else:
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.85, wspace=0.2)
    fig.text(
        0.5, 0.94, PROBLEM_NAMES[PROBLEM_NUMBER], ha='center', fontsize=12)
    if PROBLEM_NUMBER == 2:
        fig.text(0.04, 0.5, 'Ratio of optimal value achieved',
                 va='center', rotation='vertical', fontsize=12)

fig.savefig('plots/convergence_plot_%d.png' % (PROBLEM_NUMBER))
