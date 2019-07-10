import subprocess
import pandas as pd
import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import colorConverter as cc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rcParams
rcParams['figure.figsize'] = 20, 4
rcParams['figure.dpi'] = 300

folder_name = 'plots'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def plot_mean_and_CI(values, lb, ub, dims, color_values=None, linestyle=None, axis = None, marker = None, dashes = None, set_ticks = False):

    if len(values) == 0:
        return

    y, lb, ub = np.log10(values), np.log10(lb), np.log10(ub)
    x = dims
    axis.set_title(PROBLEM_NAME)
    if dashes == None:
        line = axis.errorbar(np.log10(x), y, yerr = [y-lb, ub-y], color=color_values, capsize = 10, capthick=1, linestyle = linestyle, marker = marker)
    else:
        line = axis.errorbar(np.log10(x), y, yerr = [y-lb, ub-y], color=color_values, capsize = 10, capthick=1, linestyle = linestyle, marker = marker, dashes = dashes)
    
    if set_ticks:

        axis.set_xticks(np.log10(x))
        axis.set_xticklabels(dims)
        
        yticks = np.arange(int(y[-1])+1)+1
        axis.set_yticks(yticks)
        axis.set_yticklabels(10**yticks.astype(np.int32))    
        f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        g = lambda x,pos : "${}$".format(f._formatSciNotation('%.10e' % (10**x)))
        
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(g))
        
    axis.grid(True)

    return line

def load_data(prefix_results, PROBLEM_NAME, PROBLEM_NUMBER):
    FOLDER_SUFFIX = ''
    if PROBLEM_NUMBER in [1,2,4]:
        DIM = 8
        NUM = 6
    elif PROBLEM_NUMBER in [0]:
        DIM = 25
        NUM = 5
    elif PROBLEM_NUMBER in [3]:
        DIM = 25
        NUM = 2
    
    all_vtr_hits = []
    dims = []
    for i in range(NUM):

        FOLDER_NAME = str(PROBLEM_NUMBER) + '_' + str(DIM)
        
        try:
            values = pd.read_csv('%s/%s_results_%d_%d.csv' % (FOLDER_NAME, prefix_results, PROBLEM_NUMBER, DIM)).values
            vtr = np.loadtxt('%s/0/vtr.txt' % (FOLDER_NAME)).astype(np.float32)
            vtr_hits = []
            hits, no_hits  = 0, 0 
            for i in range(values.shape[1]):
                hit = False
                for j in range(values.shape[0]):
                    if values[j,i] >= vtr:
                        vtr_hits.append(j+1)
                        hit = True
                        break
                if hit:
                    hits+=1
                else:
                    no_hits += 1

            succesful_runs = hits / float(no_hits + hits)
            print '%s PROBLEM NAME = %s | D = %d' % (prefix_results, PROBLEM_NAME, DIM), 'optimum hits = %d out of %d runs' % (hits, hits + no_hits)

            if succesful_runs == 1:
                vtr_hits = np.array(np.sort(vtr_hits))
                all_vtr_hits.append(np.sort(vtr_hits))
                dims.append(DIM)
        
        except Exception as e:
            pass
    
        DIM *= 2


    all_vtr_hits = np.array(all_vtr_hits)
    ub, lb, mean_hits, median_hits = [], [], [], []
    for i in range(all_vtr_hits.shape[0]):
        if len(all_vtr_hits[i]) >= 10:
            ub.append(all_vtr_hits[i][-2])
            lb.append(all_vtr_hits[i][1])
            mean_hits = [np.mean(all_vtr_hits[i]) for i in range(all_vtr_hits.shape[0])]
            median_hits = [np.median(all_vtr_hits[i]) for i in range(all_vtr_hits.shape[0])]
        
        else:
            ub.append(all_vtr_hits[i][-1])
            lb.append(all_vtr_hits[i][0])            
            mean_hits = [np.mean(all_vtr_hits[i]) for i in range(all_vtr_hits.shape[0])]
            median_hits = [np.median(all_vtr_hits[i]) for i in range(all_vtr_hits.shape[0])]
    
    lb, ub, mean_hits, median_hits = np.array(lb), np.array(ub), np.array(mean_hits), np.array(median_hits)
    if len(mean_hits):
        print 'evaluations for achieving optimum:'
        print '     mean:', mean_hits
        print '     median:', median_hits
        print '     max:', ub
        print '     min:', lb
    return mean_hits, median_hits, lb, ub, dims, all_vtr_hits


colors = ['blue', 'red', 'cyan', 'green']


fig, ax = plt.subplots(1, 5, figsize = (15, 3))

PROBLEM_NAMES = ['Onemax',
                 'Trap4-Tight',
                 'Trap4-Loose',
                 'NK-S1',
                 'HIFF']

from scipy.stats import t, wilcoxon, mannwhitneyu, linregress

line_i_surr, line_surr, line_vanilla, line_smac, line_hyperopt = None,None,None,None,None
cnt = 0
for i in [0,1,2,3,4]:
    PROBLEM_NAME = PROBLEM_NAMES[i]

    mean, median, lb, ub, dims, surr = load_data('surrogate', PROBLEM_NAME, i)
    line1 = plot_mean_and_CI(median, lb, ub, dims, color_values=colors[0], linestyle='--', axis=ax[cnt], marker = 'o', set_ticks = False)
    if line1 != None:
        line_surr = line1
    
    mean, median, lb, ub, dims, vanilla = load_data('vanilla', PROBLEM_NAME, i)
    line2 = plot_mean_and_CI(median, lb, ub, dims, color_values=colors[1], linestyle = '-.', axis=ax[cnt], marker = 's', set_ticks = True)
    if line2 != None:
        line_vanilla = line2
    
    mean, median, lb, ub, dims, smac = load_data('smac', PROBLEM_NAME, i)
    line3 = plot_mean_and_CI(median, lb, ub, dims, color_values=colors[2], linestyle = ':', axis=ax[cnt], marker = 'v', set_ticks = False)
    if line3 != None:
        line_smac = line3

    mean, median, lb, ub, dims, hyperopt = load_data('hyperopt', PROBLEM_NAME, i)
    line4 = plot_mean_and_CI(median, lb, ub, dims, color_values=colors[3], linestyle = '-', axis=ax[cnt], marker = '^', dashes=[6, 2], set_ticks = False)
    if line4 != None:
        line_hyperopt = line4

    for i in range(surr.shape[0]):
        p = mannwhitneyu(surr[i], vanilla[i], alternative = 'less')[1]
        print 'STATISTICAL TEST for %s | p_value=%lf | statistical siginificance with Bonferroni correction=%d' % (PROBLEM_NAME, p, p < 0.05 / surr.shape[0])

    cnt += 1

plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.915, wspace=0.3, hspace=0.2)

fig.legend([line_surr, line_vanilla, line_smac, line_hyperopt], ['CS-GOMEA', 'GOMEA', 'SMAC', 'Hyperopt'],
    loc='lower center', bbox_to_anchor=(0.5, 0.01), fancybox=False, borderaxespad=0., ncol = 5, frameon=False)


fig.text(0.5, 0.1, 'Problem dimensionality', ha='center', fontsize=12)
fig.text(0.04, 0.5, 'Number of evaluations', va='center', rotation='vertical', fontsize=12)

fig.savefig('plots/scalability_plots.png' )