# CS-GOMEA

Paper: Arkadiy Dushatskiy, Adriënne M. Mendrik, Tanja Alderliesten, and Peter A. N. Bosman. 2019. Convolutional neural network surrogate-assisted GOMEA. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19), Manuel López-Ibáñez (Ed.). ACM, New York, NY, USA, 753-761. DOI: https://doi.org/10.1145/3321707.3321760 

Link to the paper: https://dl.acm.org/citation.cfm?id=3321760

---
### Compilation

To compile CS-GOMEA: `./m_cs_gomea`

To compile vanilla GOMEA: `./m_vanilla_gomea`

---
### Benchmark problems
Problems:

0. Onemax
1. Tight Trap4
2. Loose Trap4
3. NK Landscapes
4. HIFF

### Run algorithms

Several runs can be made, folders with results are created automatically:

1. Convolutional Neural Net Surrogate-Assisted GOMEA (CS-GOMEA):

      `python run_cs_gomea.py PROBLEM_NUMBER DIMENSIONALITY FOS_TYPE MAX_EVALUATIONS DELTA WARMUP_PERIOD FIRST_RUN N_RUNS DEVICE_ID TIME_LIMIT`
  
2. Vanilla GOMEA: 

      `python run_vanilla_gomea.py PROBLEM_NUMBER DIMENSIONALITY FOS_TYPE MAX_EVALUATIONS FIRST_RUN N_RUNS`

3. SMAC (https://github.com/automl/pysmac): 

      `python run_vanilla_gomea.py  PROBLEM_NUMBER DIMENSIONALITY FIRST_RUN N_RUNS N_EVALUATIONS`

4. Hyperopt (implementation of Tree Parzen Estimator, https://github.com/hyperopt/hyperopt): 

      `python run_vanilla_gomea.py  PROBLEM_NUMBER DIMENSIONALITY FIRST_RUN N_RUNS N_EVALUATIONS`

Parameters description:
1. `PROBLEM_NUMBER` - problem number chosen from above-mentioned problems
2. `DIMENSIONALITY` - number of variables
3. `FOS_TYPE` - FOS algorithm of GOMEA, 1 (the Linkage Tree) is recommended
4. `MAX_EVALUATIONS` - maximum number of function evaluations (real ones) allowed
5. `DELTA` - a parameter of CS-GOMEA determining how aggressive real evaluations are. The recommended value is 1.02
6. `WARMUP_PERIOD` - the number of solutions in warm-up period of CS-GOMEA. This parameter is problem dependent, but it is suggested to generate at least 100 solutions to train the surrogate model.
7. `FIRST_RUN` - the id of the first run of in a series of runs. While running experiments, folders with names `P_S/R` are created, where P, S, R are `PROBLEM_NUMBER`, `DIMENSIONALITY` and id of run respectively.
8. `N_RUNS` - number of algorithm runs.
9. `DEVICE_ID` - If there any CUDA devices, the device id. -1 means CPU usage. It is recommended to use a GPU for acceleration. 
10. `TIME_LIMIT` - algorithm time limit (in minutes)

### Make plots
1. Convergence plots: 

      `python convergence_plots.py PROBLEM_NUMBER FIRST_RUN N_RUNS SMAC_HYPEROPT`
      
      Creating convergence plots.
      `FIRST_RUN, N_RUNS` indicate the folders with experiments to look in;
      `SMAC_HYPEROPT` indicates whether to include SMAC and Hyperopt runs on plots

2. Scalability plots: `python scalability_plots.py`. 

      Simply creating all scalability plots for all available problems instances
