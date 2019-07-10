
import json
from nets import *
import csv
import numpy as np
import pickle
import os
from scipy.stats import spearmanr
from numpy.random import RandomState
from copy import deepcopy
import subprocess
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils import data
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda as cutorch

N_MODELS = 1
BATCH_SIZE = 512
MAX_TIME = -1
MAX_GOMEAS = 20
FOLDER_NAME = ''

current_model_bias = 0
current_max = 0
model_quality = []
all_data_x, all_data_y, all_data_max = {}, {}, {}

device = ''

def set_gpu_device(id, max_time):
    global device, MAX_TIME

    print 'setting cuda device', id
    if id >= 0:
        device = 'cuda:%d' % id
    else:
        device = 'cpu'

    MAX_TIME = max_time

def function_init(FOLDER_NAME_):
    global FOLDER_NAME
    FOLDER_NAME = str(FOLDER_NAME_)
    if not os.path.exists(FOLDER_NAME):
        os.makedirs(FOLDER_NAME)
    
    filelist = [f for f in os.listdir(FOLDER_NAME) if  f in ['params.txt', 'params_value.txt']] + [f for f in os.listdir(FOLDER_NAME) if  '.pkl' in f]
    for f in filelist:
        os.remove(os.path.join(FOLDER_NAME, f))

def function_reset_file():
    for i in range(-1, MAX_GOMEAS):
        csvfile = open('%s/solutions_%d.csv' % (FOLDER_NAME, i), 'w')
        csvfile.close()

def function_save(array, objective_value, gomea_index, check):
    array = np.array(array).astype(np.float32)
    array -= 0.5

    if check:
        csvfile = open('%s/solutions_%d.csv' % (FOLDER_NAME, gomea_index), 'r')
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            if objective_value == float(row[-1]): 
                if np.array_equal(np.array(row[:-1]).astype(np.float32), array):
                    return
        csvfile.close()

    try:
        csvfile = open('%s/solutions_%d.csv' % (FOLDER_NAME, gomea_index), 'a')
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(list(array)+[objective_value])
        csvfile.close()

    except Exception as e:
        print e

def load_data(gomea_index):

    dataframe = []
    
    if gomea_index > 0:
        csvfile = open('%s/solutions_-1.csv' % (FOLDER_NAME), 'r')
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            dataframe.append(row)
        csvfile.close()

    csvfile = open('%s/solutions_%d.csv' % (FOLDER_NAME, gomea_index), 'r')
    reader = csv.reader(csvfile, delimiter=' ')
    for row in reader:
        dataframe.append(row)
    csvfile.close()


    dataframe = np.array(dataframe)
    dataframe = np.unique(dataframe, axis = 0)
    
    data_x = dataframe[:, :-1].astype(np.float32)
    data_y = dataframe[:, -1].astype(np.float32)

    current_max = np.max(np.abs(data_y))

    return data_x, data_y, current_max


def train_routine(model, train_x, train_y, val_x, val_y, without_improving = 5):
    global device

    if train_x.shape[0] >= 10000:
        without_improving = 3
    if train_x.shape[0] >= 50000:
        without_improving = 2
    if train_x.shape[0] >= 100000:
       without_improving = 1

    max_epochs = 10000

    train_dataset = dataset(train_x, train_y)
    val_dataset = dataset(val_x, val_y)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0.001)

    model.to(device)
    model.train()

    val_losses = []
    best_model = None
    for epoch in range(max_epochs):

        model.train()       

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        train_epoch_loss, epoch_count = 0.0, 0
        for i, data in enumerate(train_loader):
            x, y = data[0].float().to(
                device), data[1].float().reshape(-1, 1).to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            # print outputs,y
            train_epoch_loss += loss.item()
            epoch_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_epoch_loss /= epoch_count

        if epoch % 2 == 0:
            model.eval()
            epoch_loss, epoch_count = 0.0, 0
            for i, data in enumerate(val_loader):
                x, y = data[0].float().to(
                    device), data[1].float().reshape(-1, 1).to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                epoch_loss += loss.item()
                epoch_count += 1
            epoch_loss /= epoch_count

            val_losses.append(epoch_loss)
            if epoch_loss == np.min(val_losses):
                best_model = deepcopy(model).to(device)
            

            if len(val_losses) >= without_improving:
                val_improved = 0
                for i in range(len(val_losses)-without_improving, len(val_losses)):
                    if np.min(val_losses[:i+1]) == val_losses[i]:
                        val_improved = 1

                if not val_improved and len(val_losses) >= 5:
                    break


    model = deepcopy(best_model).to(device)

    preds = best_model(torch.from_numpy(val_x).to(
        device).float()).detach().cpu().numpy().flatten()

    try:
        plt.scatter(val_y, preds)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.savefig('%s/true_prediction_training.png' % FOLDER_NAME)
        plt.close()
    except Exception as e:
        print e

    torch.cuda.empty_cache()
    r2 = 1 - np.mean((preds - val_y)**2) / np.var(val_y)
    return r2


def find_params(input_width, train_x, train_y, val_x, val_y, train_x_next, train_y_next, val_x_next, val_y_next):
    
    start_time = time.time()

    try:
        best_params = json.load(open('%s/params.txt' % FOLDER_NAME, 'r'))
        return best_params
    except Exception as e:
        pass

    array_size = train_x.shape[-1]
    evaluations = {}

    def objective(params):
        for key, value in sorted(evaluations.iteritems(), key=lambda (k, v): (v, k)):
            print "%s: %s" % (key, value)

        pair = tuple(params.values())
        if pair not in evaluations:
            
            filter_size, stride_size, dilation_size, filter_size2, nb_fc_layers = params['filter_size'], params['stride'], params['dilation'], params['filter_size2'], params['nb_fc_layers']
            filter_size, stride_size, dilation_size, filter_size2, nb_fc_layers = int(filter_size), int(stride_size),  int(dilation_size),  int(filter_size2),  int(nb_fc_layers)
            if stride_size > filter_size or filter_size2 > 2*filter_size:
                val_error = -1000
                evaluations[pair] = -val_error
                return {'loss':-val_error}


            try:
                model = ConvNet(array_size, input_width, params)

                val_error = train_routine(
                    model, train_x, train_y, val_x, val_y, without_improving = 5)

            except Exception as e:
                val_error = -1000
            
            cur_best = -10000
            for key in evaluations:
                cur_best = max(cur_best, -evaluations[key])

            if val_error > cur_best:
                cur_best = val_error
                json.dump(params, open('%s/params.txt' % FOLDER_NAME, 'w'))
                np.savetxt('%s/params_value.txt' %
                           FOLDER_NAME, np.array([val_error]))

            evaluations[pair] = -val_error

            if val_error != -1000:
                return {'loss':-val_error}
            else:
                return {'loss':-val_error}

        else:
            return {'loss':evaluations[pair]}
            

    best_params = {}
    best_obj = 10**6

    params_found = False
    for filter_size in range(1, min(10, array_size // 2 + 1)):
        
        if best_obj <= -0.95:
            break

        for dilation in range(1, array_size // 4 + 1):
            
            if best_obj <= -0.95:
                break
        
            for stride in range(1, filter_size + 1):
                space = {'filter_size': filter_size,
                'dilation': dilation,
                'filter_size2': 2,
                'stride': stride,
                'nb_fc_layers': 0}
                cur_value = objective(space)['loss']        
                if cur_value < best_obj:
                    best_params = space
                    best_obj = cur_value        

                print space, best_params, best_obj

                if best_obj <= -0.97:
                    return best_params

                if best_obj <= -0.95:
                    break

                print 'TIME PASSED %lf seconds' % (time.time() - start_time)
                if time.time() - start_time >= MAX_TIME // 2:
                    return best_params

    for filter_size2 in range(1, 2 * best_params['filter_size'] + 1):
        for nb_fc_layers in range(0, 2):
            space = deepcopy(best_params)
            space['filter_size2'] = filter_size2
            space['nb_fc_layers'] = nb_fc_layers
            print space, best_params
            cur_value = objective(space)['loss']        
            if cur_value < best_obj:
                best_params = space
                best_obj = cur_value 

            if best_obj <= -0.97:
                return best_params

            if time.time() - start_time >= MAX_TIME // 2:
                return best_params
                
    print best_params
    return best_params


def train_model_pairwise_regression(gomea_index):
    global all_models, current_model_bias, true_prediction
    global all_data_x, all_data_y, all_train_ind, all_data_max

    data_x, data_y, current_max = load_data(gomea_index)
    data_y /= current_max
    all_data_x[gomea_index] = data_x
    all_data_y[gomea_index] = data_y
    all_data_max[gomea_index] = current_max

    
    all_val_y = []
    all_val_preds = []

    array_size = data_x.shape[1]
    try:
        model = ConvNet(array_size, 2, {})
    except Exception as e:
        print e
    
    best_train_ind = None
    max_val_r2 = -10**9
    cur_best_model = None
    all_train_ind = []
    all_val_y, all_val_preds = [], []

    ind_sort = np.argsort(data_y)
        
    for q in range(N_MODELS):

        val_ind = ind_sort[q::5]
        train_ind = ind_sort[np.mod(np.arange(ind_sort.size), 5) != q]

        train_ind, val_ind = np.array(train_ind), np.array(val_ind)
        train_ind, val_ind = np.random.permutation(
            train_ind), np.random.permutation(val_ind)

        train_x_reg, val_x_reg = data_x[train_ind], data_x[val_ind]
        train_y_reg, val_y_reg = data_y[train_ind], data_y[val_ind]
        train_x, val_x, train_y, val_y = [], [], [], []

        for i in range(train_x_reg.shape[0]):
            for j in range(train_x_reg.shape[0]):
                train_x.append(np.vstack([train_x_reg[i], train_x_reg[j]]))
                y = train_y_reg[i] - train_y_reg[j]
                train_y.append(y)
            
            if len(train_y) > 10**6//2:
                break

        for i in range(train_x_reg.shape[0]):
            for j in range(val_x_reg.shape[0]):
                val_x.append(np.vstack([train_x_reg[i], val_x_reg[j]]))
                val_y.append(train_y_reg[i] - val_y_reg[j])
                val_x.append(np.vstack([val_x_reg[j], train_x_reg[i]]))
                val_y.append(val_y_reg[j] - train_y_reg[i])

            if len(val_y) > 10**4:
                break

        train_x, train_y, val_x, val_y = np.array(train_x), np.array(
            train_y), np.array(val_x), np.array(val_y)
        train_x, train_y = train_x[:10**6//2], train_y[:10**6//2]
        val_x, val_y = val_x[:10**4], val_y[:10**4]
        
        train_x = train_x.reshape(
            train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
        val_x = val_x.reshape(
            val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])


        ###############################################################################

        best_params = {}
        try:
            best_params = find_params(2, train_x, train_y, val_x, val_y, None, None, None, None)
        except Exception as e:
            print e
        
        print 'best params', best_params
        model = ConvNet(array_size, 2, best_params)
        
        try:
            val_r2 = train_routine(
                model, train_x, train_y, val_x, val_y)
            print 'saving with r^2 %f' % val_r2
            torch.save(model, '%s/model_%d_%d.pkl' % (FOLDER_NAME, gomea_index, q))
            if val_r2 > max_val_r2:
                max_val_r2 = val_r2

        except Exception as e:
            print e

        all_train_ind.append(train_ind)

        indices = np.arange(train_x_reg.shape[0])[:100]

        for i in range(val_x_reg.shape[0]):

            array = val_x_reg[i]
            
            torch_array = np.zeros(
                (indices.shape[0]*2, 1, 2, array.shape[0]))
            torch_array[:indices.shape[0], 0, 0] = array
            torch_array[:indices.shape[0], 0, 1] = train_x_reg[indices]
            torch_array[indices.shape[0]:, 0, 1] = array
            torch_array[indices.shape[0]:, 0, 0] = train_x_reg[indices]
            torch_array = torch.from_numpy(torch_array).float()

            model.eval()
            
            torch_array = torch_array.to(device)
            preds = model(torch_array).detach().cpu().numpy().flatten()

            f1 = train_y_reg[indices] * current_max + \
                preds[:preds.shape[0]//2] * current_max
            f2 = train_y_reg[indices] * current_max - \
                preds[preds.shape[0]//2:] * current_max
            fitness = np.hstack([f1, f2])
            fitness = np.mean(fitness)

            all_val_y.append(val_y_reg[i]*current_max)
            all_val_preds.append(fitness)

    all_val_y, all_val_preds = np.array(all_val_y), np.array(all_val_preds)
    r2 = 1 - np.mean((all_val_preds-all_val_y)**2) / np.var(all_val_y)
    print "VALIDATION R^2 ",  r2
    rho, pvalue = spearmanr(all_val_y, all_val_preds)
    if np.isnan(rho):
        rho = -1
    print 'VALIDATION SPEARMAN ', rho
    
    current_model_bias = np.mean(all_val_y - all_val_preds)  # error

    try:
        plt.scatter(all_val_y, all_val_preds)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.savefig('%s/true_prediction_training_pairwise.png' % FOLDER_NAME)
        plt.close()
    except Exception as e:
        print e

    model_quality.append((data_x.shape[0], r2, rho))
    
    try:
        p1, = plt.plot(np.array(model_quality)[
                       :, 0], np.array(model_quality)[:, 1])
        p2, = plt.plot(np.array(model_quality)[
                       :, 0], np.array(model_quality)[:, 2])
        plt.xlabel('number of evaluations')
        plt.ylabel('score')
        plt.legend((p1, p2), ('R^2', 'Spearman Rho'))
        plt.savefig('%s/model_quality.png' % FOLDER_NAME)
        plt.close()
    except Exception as e:
        print e

    try:
        all_models = load_models(gomea_index)
    except Exception as e:
        print e
    print 'training finished'

    torch.cuda.empty_cache()

    return rho


def load_models(gomea_index_):
    models_ = {}
    for gomea_index in range(MAX_GOMEAS):
        models_[gomea_index] = []
        try:
            for i in range(N_MODELS):
                models_[gomea_index].append(torch.load('%s/model_%d_%d.pkl' %
                                  (FOLDER_NAME, gomea_index, i)).to(device).eval())
        except Exception as e:
            pass

    return models_


def get_model_prediction_fitness_pairwise(array, gomea_index, bias = False):
    global current_model_bias, all_data_x, all_data_y, all_data_max, all_models
    
    data_x, data_y, current_max = all_data_x[gomea_index], all_data_y[gomea_index], all_data_max[gomea_index]
    models = all_models[gomea_index]
    all_fitnesses = []
    for m in range(N_MODELS):
        
        ind_sort = np.argsort(data_y)
        val_ind = ind_sort[m::5]
        train_ind = ind_sort[np.mod(np.arange(ind_sort.size), 5) != m]

        train_x_reg = data_x[train_ind[:100]]
        train_y_reg = data_y[train_ind[:100]]

        torch_array = np.zeros(
            (train_x_reg.shape[0]*2, 1, 2, data_x.shape[1]))
        torch_array[:train_x_reg.shape[0], 0, 0] = array
        torch_array[:train_x_reg.shape[0], 0, 1] = train_x_reg
        torch_array[train_x_reg.shape[0]:, 0, 1] = array
        torch_array[train_x_reg.shape[0]:, 0, 0] = train_x_reg
        torch_array = torch.from_numpy(torch_array).float()

        torch_array = torch_array.to(device)
        preds = models[m](torch_array).detach().to('cpu').numpy().flatten()

        f1 = train_y_reg * current_max + \
            preds[:preds.shape[0]//2] * current_max
        f2 = train_y_reg * current_max - \
            preds[preds.shape[0]//2:] * current_max
        fitness = np.hstack([f1, f2])

        fitness_mean = np.mean(fitness)
        fitness_var = np.var(fitness)
        all_fitnesses.append(fitness_mean)

    fitness_models_mean = np.mean(all_fitnesses)
    fitness_models_var = np.var(all_fitnesses)

    if bias:
        fitness += current_model_bias

    return fitness_models_mean, fitness_models_var


def function_get_fitness_pairwise(array, gomea_index):
    array = np.array(array).reshape(1, -1)
    array -= 0.5
    pred, var = get_model_prediction_fitness_pairwise(array, gomea_index, bias=False)
    return pred, var