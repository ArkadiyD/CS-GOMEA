# CS-GOMEA

Paper: Arkadiy Dushatskiy, Adriënne M. Mendrik, Tanja Alderliesten, and Peter A. N. Bosman. 2019. Convolutional neural network surrogate-assisted GOMEA. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '19), Manuel López-Ibáñez (Ed.). ACM, New York, NY, USA, 753-761. DOI: https://doi.org/10.1145/3321707.3321760 

Link to the paper: https://dl.acm.org/citation.cfm?id=3321760

To compile CS-GOMEA: `./m_cs_gomea`

To compile vanilla GOMEA: `./m_vanilla_gomea`

Problems:

0. Onemax
1. Tight Trap4
2. Loose Trap4
3. NK Landscapes
4. HIFF

Run algorithms (make several runs, folders with results are created automatically):

1. CS-GOMEA: `python un_cs_gomea.py PROBLEM_NUMBER DIMENSIONALITY FOS_TYPE MAX_EVALUATIONS DELTA WARMUP_PERIOD FIRST_RUN N_RUNS DEVICE_ID TIME`

2. vanilla GOMEA: `python run_vanilla_gomea.py PROBLEM_NUMBER DIMENSIONALITY FOS_TYPE MAX_EVALUATIONS FIRST_RUN N_RUNS`

3. SMAC: `python run_vanilla_gomea.py  PROBLEM_NUMBER DIMENSIONALITY FIRST_RUN N_RUNS N_EVALUATIONS`

4. Hyperopt: `python run_vanilla_gomea.py  PROBLEM_NUMBER DIMENSIONALITY FIRST_RUN N_RUNS N_EVALUATIONS`

