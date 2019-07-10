import numpy as np

def onemax(array):
  array_flattened = array.flatten()
  return np.sum(array_flattened)

def ktrap_tight_evaluation4(array):
    k = 4
    array_flattened = array.flatten()
    m      = array_flattened.shape[0] // k
    result = 0.0
    for i in range(m):
        u = 0
        for j in range(k):
            u += array_flattened[i*k+j]

        if u == k:
          result += 1.0
        else:
          result += float(k-1.0-u)/float(k)
    
    return result

def ktrap_loose_evaluation4(array):
    k = 4
    array_flattened = array.flatten()
    m      = array_flattened.shape[0] // k
    result = 0.0;
    for i in range(m):
        u = 0
        for j in range(k):
          u += array[i+m*j];

        if u == k:
            result += 1.0
        else:
          result += float(k-1.0-u)/float(k)
  
    return result

def hiff(array):
  array_flattened = array.flatten()
  result     = 0.0
  block_size = 1
  number_of_parameters = array.shape[0]
  
  while block_size <= number_of_parameters:

    for i in range(0, number_of_parameters, block_size):
        
        same = 1
        for j in range(block_size):
            if i + j >= number_of_parameters:
              continue
            if array[i+j] != array[i]:
                same = 0
                break

        if same:
            result += block_size
    
    block_size *= 2

  return result


def adf(array):
  d = array.shape[0]

  if d == 25:
      filename = 'problem_data/problem_data/nk-s1/N25K5S1.txt'
  elif d == 50:
      filename = 'problem_data/problem_data/nk-s1/N50K5S1.txt'
  
  result = 0
  f = open(filename, 'r')
  lines = f.readlines()
  for i in range(len(lines)):
    line = lines[i]
    line =  line.replace('\n','').split(' ')
    if len(line) == 5:
      indices = line
      indices = [int(ind) for ind in indices]
      for j in range(32):
        line_next = lines[i + 1 + j].split(' ')
        combination, value = line_next[0], line_next[1]
        cnt = 0
        for q in range(1,6):
          if int(combination[q]) == int(array[indices[q-1]]):
            cnt += 1
        if cnt == 5:
          result += float(value)
  f.close()
  return result