/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Header -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * GOMEA.c
 *
 * Copyright (c) Peter A.N. Bosman
 *
 * The software in this file is the proprietary information of
 * Peter A.N. Bosman.
 *
 * IN NO EVENT WILL THE AUTHOR OF THIS SOFTWARE BE LIABLE TO YOU FOR ANY
 * DAMAGES, INCLUDING BUT NOT LIMITED TO LOST PROFITS, LOST SAVINGS, OR OTHER
 * INCIDENTIAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR THE INABILITY
 * TO USE SUCH PROGRAM, EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGES, OR FOR ANY CLAIM BY ANY OTHER PARTY. THE AUTHOR MAKES NO
 * REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE SOFTWARE, EITHER
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE
 * AUTHOR SHALL NOT BE LIABLE FOR ANY DAMAGES SUFFERED BY ANYONE AS A RESULT OF
 * USING, MODIFYING OR DISTRIBUTING THIS SOFTWARE OR ITS DERIVATIVES.
 *
 * Genepool Optimal Mixing Evolutionary Algorithm
 *
 * In this implementation, maximization is assumed.
 *
 * The software in this file is the result of (ongoing) scientific research.
 * The following people have been actively involved in this research over
 * the years:
 * - Peter A.N. Bosman
 * - Dirk Thierens
 * - Hoang N. Luong
 * - Silvio Rodrigues
 * - Roy de Bokx
 * - Willem den Besten
 * - Arkadiy Dushatskiy

 */
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Includes -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
#include <stdio.h>
#include <stdlib.h>
#ifdef OS_WIN
#include <stdint.h>
#endif
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

#include <Python.h>
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=- Section Global Variables -=-=-=-=-=-=-=-=-=-=-=-=-*/
typedef struct
{
  int    *indices;
  int     number_of_indices;
  double *cumulative_probabilities;
  int     number_of_cumulative_probabilities;
} binmarg;


/* Global variables */
char       *elitist_solution,                                       /* The very best solution ever evaluated. */
          **sostr,                                                  /* Set of solutions to reach. */
         ***populations,                                            /* The populations containing the solutions. */
         **evaluated_solutions,
         **evaluated_random_archive,
         **surrogate_evaluated_solutions,
         ***offsprings,                                             /* Offspring solutions (one set per population). */
           *terminated,
           **elitist_archive;                                             /* Whether a specific GOMEA with the restart scheme has terminated. */
short       write_generational_statistics,                          /* Whether to compute and write statistics every generation (0 = no). */
            write_generational_solutions,                           /* Whether to write the population every generation (0 = no). */
            print_verbose_overview,                                 /* Whether to print a overview of settings (0 = no). */
            print_FOSs_contents,                                    /* Whether to print the contents of the FOS structure each generation (0 = no). */
            use_ilse,                                               /* Whether incremental linkage subset evaluation is to be used. */
            vosostr_hit_status,                                     /* Whether the vosostr hit has happened yet: a solution has been evaluated and a value >= VTR or a STR has been found (0 = no, 1 = yes, 2 = yes, but this is no longer the first time. */
            vtr_exists,                                             /* Whether a vtr exists. */
            sostr_exists;                                           /* Whether a sostr exists. */
int         problem_index,                                          /* The index of the optimization problem. */
            FOSs_structure_index,                                   /* The index of the FOS structure. */
            number_of_parameters,                                   /* The number of parameters to be optimized. */
            number_of_solutions_in_sostr,                           /* The number of solutions in the set of solutions to reach. */
            number_of_generations,                                  /* The current generation count. */
           *population_sizes,                                       /* The number of solutions in each population. */
            base_population_size,                                   /* The minimum population size used in the smallest GOMEA instance. */
            number_of_subgenerations_per_GOMEA_factor,              /* The factor by which the number of subgenerations increases with every new population. */
           *no_improvement_stretchs,                                /* The number of subsequent generations without an improvement for every GOMEA. */
            number_of_GOMEAs,                                       /* The number of GOMEAs currently running in multipop configuration. */
            maximum_number_of_GOMEAs,                               /* The maximum number of GOMEAs running in multipop configuration. */
         ***FOSs,                                                   /* The family of subsets linkage struture. */
          **FOSs_number_of_indices,                                 /* The number of variables in each linkage subset. */
           *FOSs_length,                                            /* The number of linkage subsets. */
            minimum_GOMEA_index;                                    /* The minimum GOMEA index that corresponds to the GOMEA that is still allowed to run (lower ones should be stopped because of average fitness being lower than that of a higher one). */
long        maximum_number_of_evaluations,                          /* The maximum number of evaluations. */
            maximum_number_of_milliseconds,                         /* The maximum number of milliseconds. */
            timestamp_start,                                        /* The time stamp in milliseconds for when the program was started. */
            timestamp_start_after_init,                             /* The time stamp in milliseconds for when the algorithm was started (after problem initialization). */
            number_of_evaluations = 0,                              /* The current number of times a function evaluation was performed. */
            number_of_surrogate_evaluations = 0,                    /* The current number of times a surrogate function evaluation was performed. */
            previous_training_evaluation,                           /* The number of evaluations (real) when previous model training was performed*/
            elitist_solution_number_of_evaluations,                 /* The number of evaluations until the elitist solution. */
            elitist_solution_hitting_time,                          /* The hitting time of the elitist solution. */
            vosostr_number_of_evaluations,                          /* The number of evaluations until a solution that was to be reached (either vtr or in sostr). */
            vosostr_hitting_time;                                   /* The hitting time of a solution that was to be reached (either vtr or in sostr). */
long long   number_of_bit_flip_evaluations,                         /* The number of bit-flip evaluations. */
            elitist_solution_number_of_bit_flip_evaluations,        /* The number of bit-flip evaluations until the elitist solution. */
            vosostr_number_of_bit_flip_evaluations;                 /* The number of bit-flip evaluations until a solution that was to be reached (either vtr or in sostr). */
double      elitist_solution_objective_value, 
            elitist_solution_constraint_value,                      /* The constraint value of the elitist solution. */
            vtr,                                                    /* The value to reach (fitness of best solution that is feasible). */
            *evaluated_archive, 
            *evaluated_random_archive_values,
            *surrogate_evaluated_archive,  
            *surrogate_elitist_solution_objective_value,
            *gomeawise_elitist_solution_objective_value,                               
            *sorted_evaluated_archive,                                       
          **objective_values, 
          **not_surrogate_objective_values,  
          **real_objective_values,                                   /* Objective values for population members. */
          **constraint_values,                                      /* Sum of all constraint violations for population members. */
          **objective_values_offsprings,                            /* Objective values of selected solutions. */
          **constraint_values_offsprings,                          /* Sum of all constraint violations of selected solutions. */
           *objective_values_best_of_generation,
           *constraint_values_best_of_generation,
           *average_objective_values,
           *average_constraint_values,
         ***MI_matrices;                                            /* Mutual information between any two variables. */
int64_t     random_seed,                                            /* The seed used for the random-number generator. */
            random_seed_changing;                                   /* Internally used variable for randomly setting a random seed. */


PyObject   *module,
           *function_evaluate,
           *function_save,
           *function_reset_file,
           *function_train_model;
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

double max_delta_parameter;
int WARMUP_THRESHOLD;
int max_evaluated_solutions = 100000;
int *gomeaUpdatesCounter;
int *surrogate_evaluations_when_updated_elitist;
int *updated_solutions_count;
int *updated_solutions_count_when_updated_elitist;
int verbose = 0;
int gpu_device;
int evaluated_random_archive_size = 0;
int real_solutions_part;

double model_quality;
bool mixed_populations_mode = false;

char  filename_elitist_solution_hitting_time[200],
      filename_elitist_solutions[200],
      filename_elitist_solution[200],
      filename_solutions[200];
char  folder_name[100];

double random_solutions_min = 1e+308;

struct archiveRecord
{
  bool found;
  double value;
};

/*-=-=-=-=-=-=-=-=-=-=-=-= Section Header Functions -=-=-=-=-=-=-=-=-=-=-=-=*/
void *Malloc( long size );
int *mergeSortIntegersDecreasing( int *array, int array_size );
void mergeSortIntegersDecreasingWithinBounds( int *array, int *sorted, int *tosort, int p, int q );
void mergeSortIntegersDecreasingMerge( int *array, int *sorted, int *tosort, int p, int r, int q );
void interpretCommandLine( int argc, char **argv );
void parseCommandLine( int argc, char **argv );
void parseOptions( int argc, char **argv, int *index );
void printAllInstalledProblems();
void printAllInstalledFOSs();
void optionError( char **argv, int index );
void parseParameters( int argc, char **argv, int *index );
void printUsage();
void checkOptions();
void printVerboseOverview();
double randomRealUniform01();
int randomInt( int maximum );
int *randomPermutation( int n );
char *installedProblemName( int index );
int numberOfInstalledProblems();
void installedProblemEvaluation( int index, char *parameters, double *objective_value, double *constraint_value, int number_of_touched_parameters, int *touched_parameters_indices, char *parameters_before, double objective_value_before, double constraint_value_before );
void onemaxFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void deceptiveTrap4TightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void deceptiveTrap4LooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void deceptiveTrap5TightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void deceptiveTrap5LooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value, int k );
void deceptiveTrapKLooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value, int k );
void expensiveProblemEvaluation(int function_number, int gomea_index, char *parameters, double *objective_value, double *constraint_value, int number_of_touched_parameters, int *touched_parameters_indices, char *parameters_before, double objective_value_before, double constraint_value_before, int use_surrogate, bool *is_surrogate_used );
void adfFunctionProblemInitialization();
short adfReadInstanceFromFile();
void adfFunctionProblemNoitazilaitini();
void adfFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void hiffProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void htrap3ProblemEvaluation( char *parameters, double *objective_value, double *constraint_value );
void initializeNewGOMEA();
void initializeNewGOMEAMemory();
void initializeNewGOMEAPopulationAndFitnessValues(int gomea_index);
void initializeValueAndSetOfSolutionsToReach();
short initializeValueToReach();
short initializeSetOfSolutionsToReach();
void initializeRandomNumberGenerator();
void initializeProblem( int index );
void selectFinalSurvivorsSpecificGOMEA( int gomea_index );
char betterFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y );
char equalFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y );
void writeGenerationalStatistics();
void writeGenerationalSolutions( char is_final_generation );
void writeRunningTime( char *filename );
void writeElitistSolution();
char checkTermination();
char checkNumberOfEvaluationsTerminationCondition();
char checkVOSOSTRTerminationCondition();
char checkNumberOfMilliSecondsTerminationCondition();
void generationalStepAllGOMEAs();
void makeOffspringSpecificGOMEA( int gomea_index );
char *installedFOSStructureName( int index );
int numberOfInstalledFOSStructures();
void learnFOSSpecificGOMEA( int gomea_index );
void learnUnivariateFOSSpecificGOMEA( int gomea_index );
int **learnLTFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices, short compute_parent_child_relations, int *number_of_parent_child_relations );
int determineNearestNeighbour( int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length );
int **learnMLNFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices, short compute_parent_child_relations, int *number_of_parent_child_relations );
void learnLTNFOSSpecificGOMEA( int gomea_index );
void learnLTNFOSWithOrWithoutFilteringSpecificGOMEA( int gomea_index, short use_filtering );
void learnFilteredLTFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices );
void learnFilteredMLNFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices );
void learnFilteredLTNFOSSpecificGOMEA( int gomea_index );
void filterParentChildRelationsAndCreateNewFOSSpecificGOMEA( int gomea_index, int **parent_child_relations, int number_of_parent_child_relations );
double computeLinkageStrengthSpecificGOMEA( int gomea_index, int *variables, int number_of_variables );
void computeMIMatrixSpecificGOMEA( int gomea_index );
double *estimateParametersForSingleBinaryMarginalSpecificGOMEA( int gomea_index, int *indices, int number_of_indices, int *factor_size );
void uniquifyFOSSpecificGOMEA( int gomea_index );
short linkageSubsetsOfSameLengthAreDuplicates( int *linkageSubset0, int *linkageSubset1, int length );
void learnMPMFOSSpecificGOMEA( int gomea_index );
binmarg *learnMPMSpecificGOMEA( int gomea_index, int *mpm_length );
void estimateParametersForSingleBinaryMarginalInBinMargTypeSpecificGOMEA( int gomea_index, binmarg *binary_marginal );
double computeMetricReductionAfterMerge( int gomea_index, binmarg binary_marginal_0, binmarg binary_marginal_1 );
void printFOSContentsSpecificGOMEA( int gomea_index );
double log2( double x );
void generateAndEvaluateNewSolutionsToFillOffspringSpecificGOMEA( int gomea_index );
char *generateAndEvaluateNewSolutionSpecificGOMEA( int gomea_index, int parent_index, int offspring_index, double *obj, double *con );
void shuffleFOSSpecificGOMEA( int gomea_index );
void shuffleFOSSubsetsSpecificGOMEA( int gomea_index );
void ezilaitiniAllGOMEAs();
void ezilaitiniSpecificGOMEA( int gomea_index );
void ezilaitiniSpecificGOMEAMemoryForPopulationAndOffspring( int gomea_index );
void ezilaitiniValueAndSetOfSolutionsToReach();
void ezilaitiniProblem( int index );
long getMilliSecondsRunning();
long getMilliSecondsRunningAfterInit();
long getMilliSecondsRunningSinceTimeStamp( long timestamp );
long getCurrentTimeStampInMilliSeconds();
void run();
void multiPopGOMEA();

void mergeSortObjectivesDecreasingWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q );
void mergeSortObjectivesDecreasingMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q );

bool realEvaluation(int problem_index, int gomea_index, char *parameters, double *objective_value, double *constraint_value, int number_of_touched_parameters, int *touched_parameters_indices, char *parameters_before, double objective_value_before, double constraint_value_before);
void makeRealEvaluationsSpecificGomea(int gomea_index); //real evaluations of specific population
void makeRealEvaluations(); //real evaluations of all populations

void writeElitistEvaluationsInit( char *filename ); //routine to initilize file with optimiation results
void writeElitistEvaluations( char *filename, double elitist_objective, double elitist_constraint ); //write achieved elitist values

double call_function_train_model(int gomea_index); //calling Python function to train model
void save_new_evaluation(char *parameters, double objective_value, int gomea_index, bool check); //saving new evalaution to file with Python function call
void getFitnessPredictionsFromModel(char *parameters, double *mean_pred, int gomea_index); //call Python function to make prediction by trained model

void updateSurrogateValuesAndElitist(int gomea_model_index);  //update surrogate values with current model and local surrogate elitist for speciifc population
void updateRealElitist(int gomea_index, char *parameters, double objective_value, double constraint_value); //update real global elitist

void addToEvaluated(char *parameters, double objective_value); //add to archive of evalauted solutions
void addToRandomEvaluated(char *parameters, double objective_value); //add to archive of evaluated random solutions
archiveRecord checkAlreadyEvaluated(char *parameters); //check whether the solution has already been evaluated
void generateRandomSolutions(int number_of_solutions); //generate several random solutions
void generateSolutionsToGetQuality(double model_score); //generate random solutions and retrain model until the sufficient model quality is achieved or the termination criterion is activated

int main( int argc, char **argv );
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/


/*-=-=-=-=-=-=-=-=-=-=-= Section Elementary Operations -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Allocates memory and exits the program in case of a memory allocation failure.
 */
void *Malloc( long size )
{
  void *result;

  result = (void *) malloc( size );
  if( !result )
  {
    printf( "\n" );
    printf( "Error while allocating memory in Malloc( %ld ), aborting program.", size );
    printf( "\n" );

    exit( 0 );
  }

  return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Merge Sort -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Sorts an array of integers and returns the sort-order (large to small).
 */
int *mergeSortIntegersDecreasing( int *array, int array_size )
{
  int i, *sorted, *tosort;

  sorted = (int *) Malloc( array_size * sizeof( int ) );
  tosort = (int *) Malloc( array_size * sizeof( int ) );
  for( i = 0; i < array_size; i++ )
    tosort[i] = i;

  if( array_size == 1 )
    sorted[0] = 0;
  else
    mergeSortIntegersDecreasingWithinBounds( array, sorted, tosort, 0, array_size-1 );

  free( tosort );

  return( sorted );
}

/**
 * Subroutine of merge sort, sorts the part of the array between p and q.
 */
void mergeSortIntegersDecreasingWithinBounds( int *array, int *sorted, int *tosort, int p, int q )
{
  int r;

  if( p < q )
  {
    r = (p + q) / 2;
    mergeSortIntegersDecreasingWithinBounds( array, sorted, tosort, p, r );
    mergeSortIntegersDecreasingWithinBounds( array, sorted, tosort, r+1, q );
    mergeSortIntegersDecreasingMerge( array, sorted, tosort, p, r+1, q );
  }
}

/**
 * Subroutine of merge sort, merges the results of two sorted parts.
 */
void mergeSortIntegersDecreasingMerge( int *array, int *sorted, int *tosort, int p, int r, int q )
{
  int i, j, k, first;

  i = p;
  j = r;
  for( k = p; k <= q; k++ )
  {
    first = 0;
    if( j <= q )
    {
      if( i < r )
      {
        if( array[tosort[i]] > array[tosort[j]] )
          first = 1;
      }
    }
    else
      first = 1;

    if( first )
    {
      sorted[k] = tosort[i];
      i++;
    }
    else
    {
      sorted[k] = tosort[j];
      j++;
    }
  }

  for( k = p; k <= q; k++ )
    tosort[k] = sorted[k];
}

//merge sort by decreasing objectives
void mergeSortObjectivesDecreasingWithinBounds( double *objectives, double *constraints, int *sorted, int *tosort, int p, int q )
{
  int r;

  if( p < q )
  {
    r = (p + q) / 2;
    mergeSortObjectivesDecreasingWithinBounds( objectives, constraints, sorted, tosort, p, r );
    mergeSortObjectivesDecreasingWithinBounds( objectives, constraints, sorted, tosort, r+1, q );
    mergeSortObjectivesDecreasingMerge( objectives, constraints, sorted, tosort, p, r+1, q );
  }
}

//merge sort routine
void mergeSortObjectivesDecreasingMerge( double *objectives, double *constraints, int *sorted, int *tosort, int p, int r, int q )
{
  int i, j, k, first;

  i = p;
  j = r;
  for( k = p; k <= q; k++ )
  {
    first = 0;
    if( j <= q )
    {
      if( i < r )
      {
        if( betterFitness(objectives[tosort[i]], constraints[tosort[i]], objectives[tosort[j]], constraints[tosort[j]] ))
          first = 1;
      }
    }
    else
      first = 1;

    if( first )
    {
      sorted[k] = tosort[i];
      i++;
    }
    else
    {
      sorted[k] = tosort[j];
      j++;
    }
  }

  for( k = p; k <= q; k++ )
    tosort[k] = sorted[k];
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/




/*-=-=-=-=-=-=-=-=-=-=- Section Interpret Command Line -=-=-=-=-=-=-=-=-=-=-*/
/**
 * Parses and checks the command line.
 */
void interpretCommandLine( int argc, char **argv )
{
  parseCommandLine( argc, argv );
  
  checkOptions();
}

/**
 * Parses the command line.
 * For options, see printUsage.
 */
void parseCommandLine( int argc, char **argv )
{
  int index;

  index = 1;

  parseOptions( argc, argv, &index );
  
  parseParameters( argc, argv, &index );
}

/**
 * Parses only the options from the command line.
 */
void parseOptions( int argc, char **argv, int *index )
{
  double dummy;

  write_generational_statistics = 0;
  write_generational_solutions  = 0;
  print_verbose_overview        = 0;
  print_FOSs_contents           = 0;
  use_ilse                      = 0;

  for( ; (*index) < argc; (*index)++ )
  {
    if( argv[*index][0] == '-' )
    {
      /* If it is a negative number, the option part is over */
      if( sscanf( argv[*index], "%lf", &dummy ) && argv[*index][1] != '\0' )
        break;

      if( argv[*index][1] == '\0' )
        optionError( argv, *index );
      else if( argv[*index][2] != '\0' )
        optionError( argv, *index );
      else
      {
        switch( argv[*index][1] )
        {
          case '?': printUsage(); break;
          case 'P': printAllInstalledProblems(); break;
          case 'F': printAllInstalledFOSs(); break;
          case 's': write_generational_statistics = 1; break;
          case 'w': write_generational_solutions  = 1; break;
          case 'v': print_verbose_overview        = 1; break;
          case 'f': print_FOSs_contents           = 1; break;
          case 'i': use_ilse                      = 1; break;
          default : optionError( argv, *index );
        }
      }
    }
    else /* Argument is not an option, so option part is over */
     break;
  }
}

/**
 * Writes the names of all installed problems to the standard output.
 */
void printAllInstalledProblems()
{
  int i, n;

  n = numberOfInstalledProblems();
  printf( "Installed optimization problems:\n" );
  for( i = 0; i < n; i++ )
    printf( "%3d: %s\n", i, installedProblemName( i ) );

  exit( 0 );
}

/**
 * Writes the names of all installed FOS structures to the standard output.
 */
void printAllInstalledFOSs()
{
  int i, n;

  n = numberOfInstalledFOSStructures();
  printf( "Installed FOS structures:\n" );
  for( i = 0; i < n; i++ )
    printf( "%3d: %s\n", i, installedFOSStructureName( i ) );

  exit( 0 );
}

/**
 * Informs the user of an illegal option and exits the program.
 */
void optionError( char **argv, int index )
{
  printf( "Illegal option: %s\n\n", argv[index] );

  printUsage();
}

/**
 * Parses only the EA parameters from the command line.
 */
void parseParameters( int argc, char **argv, int *index )
{
  int noError;

  if( (argc - *index) != 10 )
  {
    printf( "Number of parameters is incorrect, require 10 parameters (you provided %d).\n\n", (argc - *index) );

    printUsage();
  }

  noError = 1;
  noError = noError && sscanf( argv[*index+0], "%d", &problem_index );
  noError = noError && sscanf( argv[*index+1], "%d", &number_of_parameters );
  noError = noError && sscanf( argv[*index+2], "%d", &FOSs_structure_index );
  noError = noError && sscanf( argv[*index+3], "%ld", &maximum_number_of_evaluations );
  noError = noError && sscanf( argv[*index+4], "%ld", &maximum_number_of_milliseconds );
  noError = noError && sscanf( argv[*index+5], "%lf", &max_delta_parameter );
  noError = noError && sscanf( argv[*index+6], "%d", &WARMUP_THRESHOLD );
  noError = noError && sscanf( argv[*index+7], "%d", &verbose );
  noError = noError && sscanf( argv[*index+8], "%s", &folder_name );
  noError = noError && sscanf( argv[*index+9], "%d", &gpu_device );

  if( !noError )
  {
    printf( "Error parsing parameters.\n\n" );

    printUsage();
  }
}

/**
 * Prints usage information and exits the program.
 */
void printUsage()
{
  printf("Usage: GOMEA [-?] [-P] [-F] [-s] [-w] [-v] [-f] [-i] pro dim fos eva mil\n");
  printf("   -?: Prints out this usage information.\n");
  printf("   -P: Prints out a list of all installed optimization problems.\n");
  printf("   -F: Prints out a list of all installed FOS structures.\n");
  printf("   -s: Enables computing and writing of statistics every generation.\n");
  printf("   -w: Enables writing of solutions and their fitnesses every generation.\n");
  printf("   -v: Enables verbose mode. Prints the settings before starting the run.\n");
  printf("   -f: Enables printing the contents of the FOS every generation.\n");
  printf("   -i: Enables Incremental Linkage Subset Evaluation (ILSE).\n");
  printf("\n");
  printf("  pro: Index of optimization problem to be solved (minimization).\n");
  printf("  dim: Number of parameters.\n");
  printf("  fos: Index of FOS structure to use.\n");
  printf("  eva: Maximum number of evaluations allowed (-1 for no limit).\n");
  printf("  mil: Maximum number of milliseconds allowed (-1 for no limit).\n");
  printf("  delta parameter: [1; inf], 1.02 is recommended.\n");
  printf("  warmup period\n");
  printf("  0 - silent, 1 - verbose\n");
  printf("  folder name, in this folder all generated files are stored\n");
  printf("  gpu device number, -1 for using CPU\n");

  exit( 0 );
}

/**
 * Checks whether the selected options are feasible.
 */
void checkOptions()
{
  if( number_of_parameters < 1 )
  {
    printf("\n");
    printf("Error: number of parameters < 1 (read: %d). Require number of parameters >= 1.", number_of_parameters);
    printf("\n\n");

    exit( 0 );
  }

  if( installedProblemName( problem_index ) == NULL )
  {
    printf("\n");
    printf("Error: unknown index for problem (read index %d).", problem_index );
    printf("\n\n");

    exit( 0 );
  }

  if( installedFOSStructureName( FOSs_structure_index ) == NULL )
  {
    printf("\n");
    printf("Error: unknown index for FOS structure (read index %d).", FOSs_structure_index );
    printf("\n\n");

    exit( 0 );
  }
}

/**
 * Prints the settings as read from the command line.
 */
void printVerboseOverview()
{
  printf("### Settings ######################################\n");
  printf("#\n");
  printf("# Statistics writing every generation: %s\n", write_generational_statistics ? "enabled" : "disabled");
  printf("# Population file writing            : %s\n", write_generational_solutions ? "enabled" : "disabled");
  printf("# FOS contents printing              : %s\n", print_FOSs_contents ? "enabled" : "disabled");
  printf("# ILSE                               : %s\n", use_ilse ? "enabled" : "disabled" );
  printf("#\n");
  printf("###################################################\n");
  printf("#\n");
  printf("# Problem                            = %s\n", installedProblemName( problem_index ));
  printf("# Number of parameters               = %d\n", number_of_parameters);
  printf("# FOS structure                      = %s\n", installedFOSStructureName( FOSs_structure_index ));
  printf("# Maximum number of evaluations      = %ld\n", maximum_number_of_evaluations);
  printf("# Maximum number of milliseconds     = %ld\n", maximum_number_of_milliseconds);
  printf("# Random seed                        = %ld\n", (long) random_seed);
  printf("#\n");
  printf("###################################################\n");
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Random Numbers -=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns a random double, distributed uniformly between 0 and 1.
 */
double randomRealUniform01()
{
  int64_t n26, n27;
  double  result;

  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n26                  = (int64_t)(random_seed_changing >> (48 - 26));
  random_seed_changing = (random_seed_changing * 0x5DEECE66DLLU + 0xBLLU) & ((1LLU << 48) - 1);
  n27                  = (int64_t)(random_seed_changing >> (48 - 27));
  result               = (((int64_t)n26 << 27) + n27) / ((double) (1LLU << 53));

  return( result );
}
        
/**
 * Returns a random integer, distributed uniformly between 0 and maximum.
 */
int randomInt( int maximum )
{
  int result;
  
  result = (int) (((double) maximum)*randomRealUniform01());
  
  return( result );
}

/**
 * Returns a random compact (using integers 0,1,...,n-1) permutation
 * of length n using the Fisher-Yates shuffle.
 */
int *randomPermutation( int n )
{
  int i, j, dummy, *result;

  result = (int *) Malloc( n*sizeof( int ) );
  for( i = 0; i < n; i++ )
    result[i] = i;

  for( i = n-1; i > 0; i-- )
  {
    j         = randomInt( i+1 );
    dummy     = result[j];
    result[j] = result[i];
    result[i] = dummy;
  }

  return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-= Section Problems -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns the name of an installed problem.
 */
char *installedProblemName( int index )
{
  switch( index )
  {
    case  0: return( (char *) "Onemax" );
    case  1: return( (char *) "Deceptive Trap 4 - Tight Encoding" );
    case  2: return( (char *) "Deceptive Trap 4 - Loose Encoding" );
    case  3: return( (char *) "NK-landscapes" );
    case  4: return( (char *) "HIFF - Tight Encoding" );
  }

  return( NULL );
}

/**
 * Returns the number of problems installed.
 */
int numberOfInstalledProblems()
{
  static int result = -1;
  
  if( result == -1 )
  {
    result = 0;
    while( installedProblemName( result ) != NULL )
      result++;
  }
  
  return( result );
}


void onemaxFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  int    i;
  double result;

  result = 0.0;
  for( i = 0; i < number_of_parameters; i++ )
    result += parameters[i];

  *objective_value  = result;
  *constraint_value = 0;
}

void deceptiveTrap4TightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  deceptiveTrapKTightEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 4 );
}

void deceptiveTrap4LooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  deceptiveTrapKLooseEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 4 );
}

void deceptiveTrap5TightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  deceptiveTrapKTightEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 5 );
}

void deceptiveTrap5LooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  deceptiveTrapKLooseEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 5 );
}

void deceptiveTrapKTightEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value, int k )
{
  int    i, j, m, u;
  double result;

  if( (number_of_parameters % k) != 0 )
  {
    printf("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.\n");
    exit( 0 );
  }

  m      = number_of_parameters / k;
  result = 0.0;
  for( i = 0; i < m; i++ )
  {
    u = 0;
    for( j = 0; j < k; j++ )
      u += parameters[i*k+j];

    if( u == k )
      result += 1.0;
    else
      result += ((double) (k-1-u))/((double) k);
  }

  *objective_value  = result;
  *constraint_value = 0;
}


void deceptiveTrapKLooseEncodingFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value, int k )
{
  int    i, j, m, u;
  double result;

  if( (number_of_parameters % k) != 0 )
  {
    printf("Error in evaluating deceptive trap k: Number of parameters is not a multiple of k.\n");
    exit( 0 );
  }

  m      = number_of_parameters / k;
  result = 0.0;
  for( i = 0; i < m; i++ )
  {
    u = 0;
    for( j = 0; j < k; j++ )
      u += parameters[i+m*j];

    if( u == k )
      result += 1.0;
    else
      result += ((double) (k-1-u))/((double) k);
  }

  *objective_value  = result;
  *constraint_value = 0;
}


int **adf_subfunctions_indices, *adf_subfunctions_indices_lengths, number_of_adf_subfunctions;
double **adf_subfunctions_values;
void adfFunctionProblemInitialization()
{
  if( !adfReadInstanceFromFile() )
  {
    printf("\n");
    printf("Error: couldn't find an adf instance.");
    printf("\n\n");

    exit( 0 );
  }
}

short adfReadInstanceFromFile()
{
  char    c, string[1000], substring[1000], filename[1000];
  int     i, j, k, q, r, t, *indices, number_of_indices,
          number_of_variables, index, number_of_values, power_of_two;
  double *values;
  FILE   *file;

  if (number_of_parameters == 25)
    sprintf( filename, "problem_data/nk-s1/N25K5S1.txt" );
  else if (number_of_parameters == 50)
    sprintf( filename, "problem_data/nk-s1/N50K5S1.txt" );

  file = fopen( filename, "r" );
  if( file == NULL )
    return( 0 );

  /* Number of variables */
  c = fgetc( file );
  k = 0;
  while( c != ' ' )
  {
    string[k] = (char) c;
    c      = fgetc( file );
    k++;
  }
  string[k] = '\0';

  number_of_variables = atoi( string );

  if( number_of_variables != number_of_parameters )
  {
    printf("Error during reading of adf instance (%s):\n",filename);
    printf("  Read number of variables: %d\n", number_of_variables);
    printf("  Doesn't match number of parameters on command line: %d\n", number_of_parameters);
    exit( 1 );
  }

  /* Number of subfunctions */
  string[k] = '\0';
  c = fgetc( file );
  k = 0;
  while( c != '\n' )
  {
    string[k] = (char) c;
    c      = fgetc( file );
    k++;
  }
  string[k] = '\0';
  number_of_adf_subfunctions = atoi( string );

  adf_subfunctions_indices         = (int **) Malloc( number_of_adf_subfunctions*sizeof( int * ) );
  adf_subfunctions_indices_lengths = (int *) Malloc( number_of_adf_subfunctions*sizeof( int ) );
  adf_subfunctions_values          = (double **) Malloc( number_of_adf_subfunctions*sizeof( double * ) );

  /* Subfunctions */
  t = 0;
  c = fgetc( file );
  k = 0;
  while( c != '\n' && c != EOF )
  {
    string[k] = (char) c;
    c      = fgetc( file );
    k++;
  }
  string[k] = '\0';
  while( k > 0 )
  {
    number_of_indices = 0;
    number_of_values  = 1;
    for( i = 0; i < k; i++ )
    {
      if( string[i] == ' ' )
      {
        number_of_indices++;
        number_of_values *= 2;
      }
    }
    number_of_indices++;
    number_of_values *= 2;

    indices = (int *) Malloc( number_of_indices*sizeof( int ) );
    i       = 0;
    j       = 0;
    while( j < k )
    {
      q = 0;
      while( (string[j] != ' ') && (j < k) )
      {
        substring[q] = string[j];
        q++;
        j++;
      }
      substring[q] = '\0';
      j++;

      indices[i] = atoi( substring );
      i++;
    }

    values = (double *) Malloc( number_of_values*sizeof( double ) );
    for( i = 0; i < number_of_values; i++ )
    {
      c = fgetc( file );
      k = 0;
      while( c != '\n' && c != EOF )
      {
        string[k] = (char) c;
        c      = fgetc( file );
        k++;
      }
      string[k] = '\0';

      q = 0;
      j = 0;
      while( (string[j] != ' ') && (j < k) )
      {
        substring[q] = string[j];
        q++;
        j++;
      }
      substring[q] = '\0';
      j++;

      index        = 0;
      power_of_two = 1;
      for( r = number_of_indices; r >= 1; r-- )
      {
        index += (substring[r]=='0'?0:1)*power_of_two;
        power_of_two *= 2;
      }

      q = 0;
      while( (string[j] != ' ') && (j < k) )
      {
        substring[q] = string[j];
        q++;
        j++;
      }
      substring[q] = '\0';
      j++;

      values[index] = atof( substring );
    }

    adf_subfunctions_indices[t]         = indices;
    adf_subfunctions_indices_lengths[t] = number_of_indices;
    adf_subfunctions_values[t]          = values;
    t++;

    c = fgetc( file );
    k = 0;
    while( c != '\n' && c != EOF )
    {
      string[k] = (char) c;
      c      = fgetc( file );
      k++;
    }
    string[k] = '\0';
  }
  fclose( file );

  return( 1 );
}

void adfFunctionProblemNoitazilaitini()
{
  int i;

  for( i = 0; i < number_of_adf_subfunctions; i++ )
  {
    free( adf_subfunctions_indices[i] );
    free( adf_subfunctions_values[i] );
  }
  free( adf_subfunctions_indices );
  free( adf_subfunctions_indices_lengths );
  free( adf_subfunctions_values );
}

void adfFunctionProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  int     i, j, index, power_of_two;
  double  result;

  result = 0.0;
  for( i = 0; i < number_of_adf_subfunctions; i++ )
  {
    index        = 0;
    power_of_two = 1;
    for( j = adf_subfunctions_indices_lengths[i]-1; j >= 0; j-- )
    {
      index += (parameters[adf_subfunctions_indices[i][j]] == 1) ? power_of_two : 0;
      power_of_two *= 2;
    }

    result += adf_subfunctions_values[i][index];
  }

  *objective_value  = result;
  *constraint_value = 0;
}


void hiffProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  char   same;
  int    i, j, block_size;
  double result;

  result     = 0.0;
  block_size = 1;
  while( block_size <= number_of_parameters )
  {
    for( i = 0; i < number_of_parameters; i += block_size )
    {
      same = 1;
      for( j = 0; j < block_size; j++ )
      {
        if (i + j >= number_of_parameters)
          continue;
        
        if( parameters[i+j] != parameters[i] )
        {
          same = 0;
          break;
        }
      }
      if( same )
        result += block_size;
    }
    block_size *= 2;
  }

  *objective_value  = result;
  *constraint_value = 0;
}

void htrap3ProblemEvaluation( char *parameters, double *objective_value, double *constraint_value )
{
  double result = 0.0;

  for (int i = 0; i < number_of_parameters; i += 3)
  {
    if (i + 2 >= number_of_parameters)
      break;
    int u = (int)parameters[i] + (int)parameters[i + 1] + (int)parameters[i + 2];
    if (u == 0)
      result += 0.9;
    else if (u == 1)
      result += 0.8;
    else if (u == 2)
      result += 0.0;
    else if (u == 3)
      result += 1;
     
  }

  *objective_value = result;
  *constraint_value = 0;
}

//SECTION core functions

//adding to archive of evaluated solutions
void addToEvaluated(char *parameters, double objective_value)
{
  if (number_of_evaluations >= max_evaluated_solutions)
    return;

  for (int i = 0; i < number_of_parameters; ++i)
  {
      evaluated_solutions[number_of_evaluations][i] = parameters[i];
  }
  evaluated_archive[number_of_evaluations] = objective_value;
}

//adding to archive of evaluated solutions which are random
void addToRandomEvaluated(char *parameters, double objective_value)
{
  if (evaluated_random_archive_size >= max_evaluated_solutions)
    return;

  for (int i = 0; i < number_of_parameters; ++i)
  {
      evaluated_random_archive[evaluated_random_archive_size][i] = parameters[i];
  }
  evaluated_random_archive_values[evaluated_random_archive_size] = objective_value;

  evaluated_random_archive_size++;
}

//check whether the solution is already evalauted, if yes return corresponding fitness values
archiveRecord checkAlreadyEvaluated(char *parameters)
{
  int start = 0;
  archiveRecord res;
  res.found = false;

  for (int i = start; i < number_of_evaluations; ++i)
  {
    bool matched = true;
    for (int j = 0; j < number_of_parameters; ++j)
    {
      if (evaluated_solutions[i][j] != parameters[j])
      {
        matched = false;
        break;
      }
    }
    if (matched) 
    {
      res.found = true;
      res.value = evaluated_archive[i];
      return res;
    }
  }
  
  return res;
}


bool realEvaluation(int problem_index, int gomea_index, char *parameters, double *objective_value, double *constraint_value, int number_of_touched_parameters, int *touched_parameters_indices, char *parameters_before, double objective_value_before, double constraint_value_before)
{
  short same;
  int   i, j;

  if (verbose)
    printf("REAL EVALUATION TRIES TO PERFORM!!! %ld\n", number_of_evaluations);

  /* Count the evaluation */
  archiveRecord record = checkAlreadyEvaluated(parameters);
  bool is_found = record.found;
  double found_value = record.value;
  
  if (is_found && number_of_evaluations)
  {
    *objective_value = found_value;
    *constraint_value = 0;
    if (verbose)   
      printf("FOUND ALREADY EVALUATED! value = %f", found_value);

    save_new_evaluation(parameters, *objective_value, gomea_index, true);

    /* Update gomeawise elitist solution */
    if( (number_of_evaluations == 1) || betterFitness( *objective_value, *constraint_value, gomeawise_elitist_solution_objective_value[gomea_index], elitist_solution_constraint_value ) )
    {
      gomeawise_elitist_solution_objective_value[gomea_index] = *objective_value;
      if (verbose)   
        printf("UPDATED GOMEA ELITIST %lf\n", gomeawise_elitist_solution_objective_value[gomea_index]);
      updated_solutions_count_when_updated_elitist[gomea_index] = updated_solutions_count[gomea_index];
    }

    return 1;
  }

  if (verbose)
    printf("NUMBER OF EVALUATIONS = %ld\n", number_of_evaluations);
  
  switch( problem_index )
  {
    case  0: onemaxFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;
    case  1: deceptiveTrap4TightEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;
    case  2: deceptiveTrap4LooseEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;
    case  3: deceptiveTrap5TightEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;
    case  4: deceptiveTrap5LooseEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;
    case  5: adfFunctionProblemEvaluation( parameters, objective_value, constraint_value ); break;    
    case  6: deceptiveTrapKTightEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 3 ); break;   
    case  7: deceptiveTrapKLooseEncodingFunctionProblemEvaluation( parameters, objective_value, constraint_value, 3 ); break;   
    case  8: hiffProblemEvaluation( parameters, objective_value, constraint_value ); break;
  }
  
  addToEvaluated(parameters, *objective_value);
  save_new_evaluation(parameters, *objective_value, gomea_index, false);
  number_of_evaluations++;
  
  /* Check the VTR */
  if( !vosostr_hit_status )
  {
    if( vtr_exists > 0 )
    {
      if( ((*constraint_value) == 0) && ((*objective_value) >= vtr)  )
      {
        vosostr_hit_status = 1;
      }
    }
  }

  /* Check the SOSTR */
  if( !vosostr_hit_status )
  {
    if( sostr_exists )
    {
      for( i = 0; i < number_of_solutions_in_sostr; i++ )
      {
        same = 1;
        for( j = 0; j < number_of_parameters; j++ )
        {
          if( parameters[j] != sostr[i][j] )
          {
            same = 0;
            break;
          }
        }
        if( same )
        {
          vosostr_hit_status = 1;
          break;
        }
      }
    }
  }

  /* Check the VOSOSTR */
  if( vosostr_hit_status == 1 )
  {
    vosostr_hit_status                     = 2;
    vosostr_hitting_time                   = getMilliSecondsRunningAfterInit();
    vosostr_number_of_evaluations          = number_of_evaluations;
    vosostr_number_of_bit_flip_evaluations = number_of_bit_flip_evaluations;
  }

  /* Update gomeawise elitist solution */
  if( (number_of_evaluations == 1) || betterFitness( *objective_value, *constraint_value, gomeawise_elitist_solution_objective_value[gomea_index], elitist_solution_constraint_value ) )
  {
    gomeawise_elitist_solution_objective_value[gomea_index] = *objective_value;
    if (verbose)  
      printf("UPDATED GOMEA ELITIST %lf\n", gomeawise_elitist_solution_objective_value[gomea_index]);
    updated_solutions_count_when_updated_elitist[gomea_index] = updated_solutions_count[gomea_index];
  }

  if( (number_of_evaluations == 1) || betterFitness( *objective_value, *constraint_value, elitist_solution_objective_value, elitist_solution_constraint_value ) )
    updateRealElitist(gomea_index, parameters, *objective_value, *constraint_value);

  /* Exit early, depending on VOSOSTR status */
  if( vosostr_hit_status != 0 || number_of_evaluations == maximum_number_of_evaluations)
  {
    if (verbose)
      printf("VTR HIT");
    exit(0);
  }
  
  /* first model training*/
  if (number_of_evaluations >= base_population_size && previous_training_evaluation == -1)
  {
    previous_training_evaluation=number_of_evaluations;
    generateSolutionsToGetQuality(-1);

    updateSurrogateValuesAndElitist(0);
  } 

  return 0;
}

void generateSolutionsToGetQuality(double model_score)
{
  generateRandomSolutions(WARMUP_THRESHOLD - 10 - number_of_evaluations);

  while(model_score < 0.9 && number_of_evaluations + 10 <= WARMUP_THRESHOLD*2)
  {
    generateRandomSolutions(10);
  
    model_score = call_function_train_model(0);
  }

  model_quality = model_score;

  if (model_quality < 0.9)
    mixed_populations_mode = true;
  else
    mixed_populations_mode = false;
}

void updateRealElitist(int gomea_index, char *parameters, double objective_value, double constraint_value)
{
  for( int i = 0; i < number_of_parameters; i++ )
      elitist_solution[i] = parameters[i];

    elitist_solution_objective_value                = objective_value;
    elitist_solution_constraint_value               = constraint_value;
    elitist_solution_hitting_time                   = getMilliSecondsRunningAfterInit();
    elitist_solution_number_of_evaluations          = number_of_evaluations;
    elitist_solution_number_of_bit_flip_evaluations = number_of_bit_flip_evaluations;

    writeRunningTime( filename_elitist_solution_hitting_time );
    writeElitistEvaluations(filename_elitist_solutions, elitist_solution_objective_value, elitist_solution_constraint_value);
    writeElitistSolution();
    if (verbose)
      printf("UPDATED GLOBAL ELITIST %lf\n", elitist_solution_objective_value);

    surrogate_evaluations_when_updated_elitist[gomea_index] = number_of_surrogate_evaluations;
    updated_solutions_count_when_updated_elitist[gomea_index] = updated_solutions_count[gomea_index];
}


//calling Python function which saves the solution along with its fitness values
void save_new_evaluation(char *parameters, double objective_value, int gomea_index, bool check)
{    
  PyObject* pyParameters = PyList_New(number_of_parameters);
  if (pyParameters == NULL) {
  printf("ERROR creating args array");
  exit(-1);
  }

  for (int i = 0; i < number_of_parameters; ++i)
    PyList_SetItem(pyParameters, i, PyFloat_FromDouble((double)parameters[i]));
  PyObject* pyObjective = PyFloat_FromDouble(objective_value);
  PyObject *pyGomeaIndex = Py_BuildValue("i", gomea_index);
  PyObject *pyCheck = Py_BuildValue("i", check);

  PyObject *pyArgs = PyTuple_Pack(4, pyParameters, pyObjective, pyGomeaIndex, pyCheck);
  PyObject* pyResult = PyObject_CallObject(function_save, pyArgs);
  
  if (pyResult == NULL) {
    printf("ERROR getting result from python function SAVE");
    exit(-1);
  }
}

//calling Python function which performs the model training
double call_function_train_model(int gomea_index)
{
  PyObject *pyArgs = Py_BuildValue("i", gomea_index);
  pyArgs = PyTuple_Pack(1, pyArgs);

  PyObject* pyResult = PyObject_CallObject(function_train_model, pyArgs);
  if (pyResult == NULL) {
    printf("ERROR getting result from python function TRAIN");
    exit(-1);
  }

  double model_quality_current = PyFloat_AsDouble(pyResult);
  if (verbose)
    printf ("MODEL QUALITY %lf\n", model_quality_current);
  previous_training_evaluation = number_of_evaluations;

  return model_quality_current;
}

//calling Python function which performs the fitness prediction by model and returns it
void getFitnessPredictionsFromModel(char *parameters, double *mean_pred, int gomea_index)
{
  number_of_surrogate_evaluations++;

  PyObject* pyParameters = PyList_New(number_of_parameters);
  if (pyParameters == NULL) {
  printf("ERROR creating args array");
  exit(-1);
  }

  for (int i = 0; i < number_of_parameters; ++i)
    PyList_SetItem(pyParameters, i, PyFloat_FromDouble((double)parameters[i]));

  PyObject *pyGomeaIndex = Py_BuildValue("i", gomea_index);

  PyObject *pyArgs = PyTuple_Pack(2, pyParameters, pyGomeaIndex);
  PyObject* pyResult = PyObject_CallObject(function_evaluate, pyArgs);
  PyObject *py_mean_pred = PyTuple_GetItem(pyResult, 0);
  
  *mean_pred = PyFloat_AsDouble(py_mean_pred);
}

void expensiveProblemEvaluation( int problem_index, int gomea_index, char *parameters, double *objective_value, double *constraint_value, int number_of_touched_parameters, int *touched_parameters_indices, char *parameters_before, double objective_value_before, double constraint_value_before, int surrogate_allowed, bool *is_surrogate_used )
{
  bool doRealEvaluation;
  
  //real evaluation
  if (previous_training_evaluation == -1 || surrogate_allowed == 0)
  {
    realEvaluation(problem_index, gomea_index, parameters, objective_value, constraint_value, number_of_touched_parameters, touched_parameters_indices, parameters_before, objective_value_before, constraint_value_before);
    *is_surrogate_used = false;

    if (previous_training_evaluation == -1)
      save_new_evaluation(parameters, *objective_value, -1, true);
  }
  else if (surrogate_allowed == 1) //try to use surrogate evaluation
  {
    double surrogate_fitness;
    getFitnessPredictionsFromModel(parameters, &surrogate_fitness, gomea_index);
   
    doRealEvaluation = false;

    if (surrogate_fitness > surrogate_elitist_solution_objective_value[gomea_index])
    {
      surrogate_elitist_solution_objective_value[gomea_index] = surrogate_fitness;
      doRealEvaluation = true;
      if (verbose)
        printf("Surrogate elitist=%lf | current surrogate fitness=%lf\n", surrogate_elitist_solution_objective_value[gomea_index], surrogate_fitness);    
    }
    else
    {
      int cnt = population_sizes[gomea_index];
      if (updated_solutions_count[gomea_index] - updated_solutions_count_when_updated_elitist[gomea_index] > cnt) 
      {
        double lower_bound = random_solutions_min;
        double delta = (surrogate_elitist_solution_objective_value[gomea_index]  - lower_bound) / (surrogate_fitness - lower_bound);
        //printf("bound %lf %lf\n", lower_bound, x);
        if (delta >= max_delta_parameter)
          doRealEvaluation = false;
        else
          doRealEvaluation = true;
      
      if (verbose)
        printf("Surrogate elitist=%lf | current surrogate fitness=%lf\n", surrogate_elitist_solution_objective_value[gomea_index], surrogate_fitness);    
      }
    }

    if (doRealEvaluation)
    {
      realEvaluation(problem_index, gomea_index, parameters, objective_value, constraint_value, number_of_touched_parameters, touched_parameters_indices, parameters_before, objective_value_before, constraint_value_before);      
      *is_surrogate_used = false;    
    }
    else
      *is_surrogate_used = true;
    
   
    *objective_value = surrogate_fitness;
    *constraint_value = 0;   
  }
  else if (surrogate_allowed == 2) //try to use surrogate evaluation
  {
    double surrogate_fitness;
    getFitnessPredictionsFromModel(parameters, &surrogate_fitness, gomea_index);
    *objective_value = surrogate_fitness;
    
    *is_surrogate_used = true;
    *constraint_value = 0;    

    if (surrogate_fitness > surrogate_elitist_solution_objective_value[gomea_index]) //updating local surrogate elitist
      surrogate_elitist_solution_objective_value[gomea_index] = surrogate_fitness;     
  }
}


/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Initialize -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Performs initialization for a single GOMEA.
 */
void initializeNewGOMEA()
{
  if( number_of_GOMEAs == 0 )
  {
    populations                          = (char ***) Malloc( maximum_number_of_GOMEAs*sizeof( char ** ) );
    objective_values                     = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    real_objective_values                     = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    not_surrogate_objective_values       = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    constraint_values                    = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    offsprings                           = (char ***) Malloc( maximum_number_of_GOMEAs*sizeof( char ** ) );
    objective_values_offsprings          = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    constraint_values_offsprings         = (double **) Malloc( maximum_number_of_GOMEAs*sizeof( double * ) );
    average_objective_values             = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );
    average_constraint_values            = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );
    terminated                           = (char *) Malloc( maximum_number_of_GOMEAs*sizeof( char ) );
    MI_matrices                          = (double ***) Malloc( maximum_number_of_GOMEAs*sizeof( double ** ) );
    FOSs                                 = (int ***) Malloc( maximum_number_of_GOMEAs*sizeof( int ** ) );
    FOSs_number_of_indices               = (int **) Malloc( maximum_number_of_GOMEAs*sizeof( int * ) );
    FOSs_length                          = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
    objective_values_best_of_generation  = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );
    constraint_values_best_of_generation = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );
    surrogate_elitist_solution_objective_value = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );
    gomeawise_elitist_solution_objective_value = (double *) Malloc( maximum_number_of_GOMEAs*sizeof( double ) );

    population_sizes[number_of_GOMEAs] = base_population_size;

    for (int i = 0; i < maximum_number_of_GOMEAs; ++i)
      gomeawise_elitist_solution_objective_value[i] = -1e+308;
  }
  else
  {
    population_sizes[number_of_GOMEAs] = 2*population_sizes[number_of_GOMEAs-1];
  }

  terminated[number_of_GOMEAs]              = 0;
  no_improvement_stretchs[number_of_GOMEAs] = 0;
  FOSs[number_of_GOMEAs]                    = NULL;
  
  if (number_of_GOMEAs > 0)
  {
    call_function_train_model(number_of_GOMEAs);
  }

  initializeNewGOMEAMemory();
  initializeNewGOMEAPopulationAndFitnessValues(number_of_GOMEAs);
  updateSurrogateValuesAndElitist(number_of_GOMEAs);
  number_of_GOMEAs++;
}

//generating random solutions and saving them to corresponding archive
void generateRandomSolutions(int number_of_solutions)
{
  char *solution = (char*)Malloc( number_of_parameters * sizeof(char));
  for (int i = 0; i < number_of_solutions; ++i)
  {
    for( int j = 0; j < number_of_parameters; j++ )
        solution[j] = (randomInt( 2 ) == 1) ? 1 : 0;
    
    double objective_value, constraint_value;
    bool is_surrogate_used;
    expensiveProblemEvaluation( problem_index, 0, solution, &objective_value, &constraint_value, 0, NULL, NULL, 0, 0, 0, &is_surrogate_used );
    addToRandomEvaluated(solution, objective_value);
    save_new_evaluation(solution, objective_value, -1, true);

    if (number_of_evaluations < WARMUP_THRESHOLD)
    {
      if (objective_value < random_solutions_min)
        random_solutions_min = objective_value;
    }
  }
}

/**
 * Initializes the memory for a single GOMEA.
 */
void initializeNewGOMEAMemory()
{
  int i;

  populations[number_of_GOMEAs]                  = (char **) Malloc( population_sizes[number_of_GOMEAs]*sizeof( char * ) );
  objective_values[number_of_GOMEAs]             = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );
  real_objective_values[number_of_GOMEAs]             = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );
  not_surrogate_objective_values[number_of_GOMEAs]             = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );
  constraint_values[number_of_GOMEAs]            = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );
  offsprings[number_of_GOMEAs]                   = (char **) Malloc( population_sizes[number_of_GOMEAs]*sizeof( char * ) );
  objective_values_offsprings[number_of_GOMEAs]  = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );
  constraint_values_offsprings[number_of_GOMEAs] = (double *) Malloc( population_sizes[number_of_GOMEAs]*sizeof( double ) );

  for( i = 0; i < population_sizes[number_of_GOMEAs]; i++ )
    populations[number_of_GOMEAs][i] = (char *) Malloc( number_of_parameters*sizeof( char ) );

  for( i = 0; i < population_sizes[number_of_GOMEAs]; i++ )
    offsprings[number_of_GOMEAs][i] = (char *) Malloc( number_of_parameters*sizeof( char ) );

  MI_matrices[number_of_GOMEAs] = (double **) Malloc( number_of_parameters*sizeof( double * ) );
  for( i = 0; i < number_of_parameters; i++ )
    (MI_matrices[number_of_GOMEAs])[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );

  FOSs[number_of_GOMEAs] = NULL;
}

/**
 * Initializes the population and the objective values by randomly
 * generation n solutions.
 */
void initializeNewGOMEAPopulationAndFitnessValues(int gomea_index)
{
  int    i, j;
  double obj, con;

  objective_values_best_of_generation[number_of_GOMEAs]  = -1e+308;
  constraint_values_best_of_generation[number_of_GOMEAs] = 1e+308;
  for( i = 0; i < population_sizes[number_of_GOMEAs]; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
      populations[number_of_GOMEAs][i][j] = (randomInt( 2 ) == 1) ? 1 : 0;

    bool is_surrogate_used = 0;
    expensiveProblemEvaluation( problem_index, gomea_index, populations[number_of_GOMEAs][i], &obj, &con, 0, NULL, NULL, 0, 0, 2, &is_surrogate_used );
    
    objective_values[number_of_GOMEAs][i]  = obj;
    constraint_values[number_of_GOMEAs][i] = con;

    if( betterFitness( objective_values[number_of_GOMEAs][i], constraint_values[number_of_GOMEAs][i], objective_values_best_of_generation[number_of_GOMEAs], constraint_values_best_of_generation[number_of_GOMEAs] ) )
    {
      objective_values_best_of_generation[number_of_GOMEAs]  = objective_values[number_of_GOMEAs][i];
      constraint_values_best_of_generation[number_of_GOMEAs] = constraint_values[number_of_GOMEAs][i];
    }
  }
}

/**
 * Checks to see if files exists with values and solutions to reach.
 */
void initializeValueAndSetOfSolutionsToReach()
{
  vtr_exists = 0;
  if( initializeValueToReach() )
    vtr_exists = 1;

  sostr_exists = 0;
  if( initializeSetOfSolutionsToReach() )
    sostr_exists = 1;
}

/**
 * Attempts to read the value to reach.
 */
short initializeValueToReach()
{
  char  c, string[100000], filename[1000];
  int   i;
  FILE *file;

  sprintf( filename, "%s/vtr.txt",  (char * )folder_name);
  file = fopen( filename, "r" );
  if( file == NULL )
    return( 0 );

  i    = 0;
  c    = fgetc( file );
  while( (c != '\n') && (c != EOF) && (c != ' ') )
  {
    string[i] = (char) c;
    c         = fgetc( file );
    i++;
  }
  string[i] = '\0';
  sscanf( string, "%le", &vtr);

  fclose( file );

  return( 1 );
}

/**
 * Attempts to read assets of solutions to reach.
 */
short initializeSetOfSolutionsToReach()
{
  char  c, string[100000], filename[1000];
  int   i, j;
  FILE *file;

  number_of_solutions_in_sostr = 0;

  sprintf( filename, "sostr.txt" );
  file = fopen( filename, "r" );
  if( file == NULL )
    return( 0 );

  do
  {
    c = fgetc( file );
    if( (c == '0') || (c == '1') )
    {
      number_of_solutions_in_sostr++;
      do
      {
        c = fgetc( file );
      } while( (c == '0') || (c == '1') );
    }
  }
  while( c != EOF );
  fclose( file );

  sostr = (char **) Malloc( number_of_solutions_in_sostr*sizeof( char * ) );
  for( i = 0; i < number_of_solutions_in_sostr; i++ )
    sostr[i] = (char *) Malloc( number_of_parameters*sizeof( char ) );

  file = fopen( filename, "r" );
  i    = 0;
  do
  {
    c = fgetc( file );
    if( c == EOF )
      break;
    if( !((c == '0') || (c == '1')) )
      continue;
    j = 0;
    while( (c != '\n') && (c != EOF) && (c != ' ') )
    {
      string[j] = (char) c;
      c         = fgetc( file );
      j++;
    }
    if( j != number_of_parameters )
    {
      printf("Error while reading %s: the number of parameters in at least one of the solutions does not match the number of parameters on the commandline (%d != %d)\n",filename,j,number_of_parameters);
      exit(0);
    }
    for( j = 0; j < number_of_parameters; j++ )
      sostr[i][j] = string[j] == '0' ? 0 : 1;
  }
  while( c != EOF );

  fclose( file );

  return( 1 );
}

/**
 * Initializes the pseudo-random number generator.
 */
void initializeRandomNumberGenerator()
{
  struct timeval tv;
  struct tm *timep;

  while( random_seed_changing == 0 )
  {
    gettimeofday( &tv, NULL );
    timep = localtime (&tv.tv_sec);
    random_seed_changing = timep->tm_hour * 3600 * 1000 + timep->tm_min * 60 * 1000 + timep->tm_sec * 1000 + tv.tv_usec / 1000;
  }

  random_seed = random_seed_changing;
}

void initializeProblem( int index )
{
  switch( index )
  {
    case  0: break;
    case  1: break;
    case  2: break;
    case  3: break;
    case  4: break;
    case  5: adfFunctionProblemInitialization(); break;
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*=-=-=-=-=-=-=-=-=-=-= Section Survivor Selection =-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Determines the solutions that finally survive the generation (offspring only).
 */
void selectFinalSurvivorsSpecificGOMEA( int gomea_index )
{
  int    i, j;
  double objective_values_best_of_generation_before, constraint_values_best_of_generation_before;

  objective_values_best_of_generation_before  = objective_values_best_of_generation[gomea_index];
  constraint_values_best_of_generation_before = constraint_values_best_of_generation[gomea_index];

  for( i = 0; i < population_sizes[gomea_index]; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
      populations[gomea_index][i][j] = offsprings[gomea_index][i][j];
    objective_values[gomea_index][i]  = objective_values_offsprings[gomea_index][i];
    constraint_values[gomea_index][i] = constraint_values_offsprings[gomea_index][i];

    //printf("OBJ VALUES %d %lf %lf\n", i, objective_values[gomea_index][i], objective_values_best_of_generation[gomea_index]);

    if( betterFitness( objective_values[gomea_index][i], constraint_values[gomea_index][i], objective_values_best_of_generation[gomea_index], constraint_values_best_of_generation[gomea_index] ) )
    {
      objective_values_best_of_generation[gomea_index]  = objective_values[gomea_index][i];
      constraint_values_best_of_generation[gomea_index] = constraint_values[gomea_index][i];
    }
  }

  if( !betterFitness( objective_values_best_of_generation[gomea_index], constraint_values_best_of_generation[gomea_index], objective_values_best_of_generation_before, constraint_values_best_of_generation_before ) )
    no_improvement_stretchs[gomea_index]++;
  else
    no_improvement_stretchs[gomea_index] = 0;
}

/**
 * Returns 1 if x is better than y, 0 otherwise.
 * x is not better than y unless:
 * - x and y are both infeasible and x has a smaller sum of constraint violations, or
 * - x is feasible and y is not, or
 * - x and y are both feasible and x has a larger objective value than y
 */
char betterFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y )
{
  char result;

  result = 0;

  if( constraint_value_x > 0 ) /* x is infeasible */
  {
    if( constraint_value_y > 0 ) /* Both are infeasible */
    {
      if( constraint_value_x < constraint_value_y )
       result = 1; 
    }
  }
  else /* x is feasible */
  {
    if( constraint_value_y > 0 ) /* x is feasible and y is not */
      result = 1;
    else /* Both are feasible */
    {
      if( objective_value_x > objective_value_y )
        result = 1;
    }
  }
  //printf("better fitness %d, %lf %lf %lf %lf\n", result, objective_value_x, objective_value_y, constraint_value_x, constraint_value_y);
  return( result );
}

/**
 * Returns 1 if x is equally preferable to y, 0 otherwise.
 */
char equalFitness( double objective_value_x, double constraint_value_x, double objective_value_y, double constraint_value_y )
{
  char result;
  
  result = 0;

  if( (constraint_value_x == constraint_value_y) && (objective_value_x == objective_value_y) )
    result = 1;

  return( result );
}

void computeAverageFitnessSpecificGOMEA( int gomea_index )
{
  int i;
  bool is_surrogate_used;
  average_objective_values[gomea_index]  = 0;
  average_constraint_values[gomea_index] = 0;

  int max_ind = population_sizes[gomea_index];
  if (mixed_populations_mode)
  {
    real_solutions_part = ceil(population_sizes[gomea_index] * (0.9 - model_quality));
    if (real_solutions_part >= population_sizes[gomea_index] / 2 )
      real_solutions_part = population_sizes[gomea_index] / 2;
    max_ind = real_solutions_part;
  }

  for( i = 0; i < max_ind; i++ )
  {
    //printf(" COMPUTING AVERAGES %d %d %lf\n", GOMEA_index, i, objective_values[GOMEA_index][i]);
    if (mixed_populations_mode)
        expensiveProblemEvaluation( problem_index, gomea_index, populations[gomea_index][i], &objective_values[gomea_index][i], &constraint_values[gomea_index][i], 0, NULL, NULL, 0, 0, 0, &is_surrogate_used );
    else
      expensiveProblemEvaluation( problem_index, gomea_index, populations[gomea_index][i], &objective_values[gomea_index][i], &constraint_values[gomea_index][i], 0, NULL, NULL, 0, 0, 2, &is_surrogate_used );

    average_objective_values[gomea_index]  += objective_values[gomea_index][i];
    average_constraint_values[gomea_index] += constraint_values[gomea_index][i];

    //printf(" COMPUTING AVERAGES RECALC %d %d %lf\n", GOMEA_index, i, objective_values[GOMEA_index][i]);
    
  }
  average_objective_values[gomea_index] /= (double) (population_sizes[gomea_index]);
  average_constraint_values[gomea_index] /= (double) (population_sizes[gomea_index]);
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Output =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Writes (appends) statistics about the current generation to a
 * file named "statistics.dat".
 */

void updateSurrogateValuesAndElitist(int gomea_model_index)
{
  surrogate_elitist_solution_objective_value[gomea_model_index] = -1e+308;
  bool is_surrogate_used;

  int index = gomea_model_index;

  for (int i = 0; i < population_sizes[index]; ++i)
  {
    double obj;
    expensiveProblemEvaluation( problem_index, gomea_model_index, populations[index][i], &obj, &constraint_values[index][i], 0, NULL, NULL, 0, 0, 2, &is_surrogate_used );
    
    objective_values[index][i] = obj;
  
    if (objective_values[index][i] > surrogate_elitist_solution_objective_value[gomea_model_index])
      surrogate_elitist_solution_objective_value[gomea_model_index] = objective_values[index][i];
  }
}

void makeRealEvaluationsSpecificGomea(int gomea_index)
{
  if (terminated[gomea_index] || gomeaUpdatesCounter[gomea_index] == 0)
      return;

  int *sorted = (int *) Malloc( population_sizes[gomea_index] * sizeof( int ) );
  int *tosort = (int *) Malloc( population_sizes[gomea_index] * sizeof( int ) );
  for( int i = 0; i < population_sizes[gomea_index]; i++ )
    tosort[i] = i;

  if (population_sizes[gomea_index] == 1)
    sorted[0] = 0;
  else
    mergeSortObjectivesDecreasingWithinBounds( objective_values[gomea_index], constraint_values[gomea_index], sorted, tosort, 0, population_sizes[gomea_index]-1 );
  

  for (int i = 0; i < population_sizes[gomea_index]; ++i)
  {
    bool is_surrogate_used;
    expensiveProblemEvaluation( problem_index, gomea_index, populations[gomea_index][sorted[i]], &real_objective_values[gomea_index][sorted[i]], &constraint_values[gomea_index][sorted[i]], 0, NULL, NULL, 0, 0, 0, &is_surrogate_used );
  }
  
  free(sorted);
  free(tosort);
}

void makeRealEvaluations()
{
  for( int gomea_index = 0; gomea_index < number_of_GOMEAs; gomea_index++ )
    makeRealEvaluationsSpecificGomea(gomea_index);
}

void writeGenerationalStatistics()
{
  char    filename[1000], string[10000];
  int     i, n, gomea_index;
  double  objective_avg, objective_var, objective_best, objective_worst,
          constraint_avg, constraint_var, constraint_best, constraint_worst;
  FILE   *file;

  /* First compute the statistics */
  /* Average, best and worst */ 
  objective_avg    = 0.0;
  constraint_avg   = 0.0;
  objective_best   = objective_values[0][0];
  objective_worst  = objective_values[0][0];
  constraint_best  = constraint_values[0][0];
  constraint_worst = constraint_values[0][0];
  n                = 0;

  

  for( gomea_index = 0; gomea_index < number_of_GOMEAs; gomea_index++ )
  {
    for( i = 0; i < population_sizes[gomea_index]; i++ )
    {
      objective_avg += real_objective_values[gomea_index][i];
      constraint_avg += constraint_values[gomea_index][i];
      if( betterFitness( real_objective_values[gomea_index][i], constraint_values[gomea_index][i], objective_best, constraint_best ) )
      {
        objective_best  = real_objective_values[gomea_index][i];
        constraint_best = constraint_values[gomea_index][i];
      }
      if( betterFitness( objective_worst, constraint_worst, real_objective_values[gomea_index][i], constraint_values[gomea_index][i] ) )
      {
        objective_worst  = real_objective_values[gomea_index][i];
        constraint_worst = constraint_values[gomea_index][i];
      }
      n++;
    }
  }
  
  objective_avg = objective_avg / ((double) n);
  constraint_avg = constraint_avg / ((double) n);

  /* Variance */
  objective_var  = 0.0;
  constraint_var = 0.0;
  for( gomea_index = 0; gomea_index < number_of_GOMEAs; gomea_index++ )
  {
    for( i = 0; i < population_sizes[gomea_index]; i++ )
    {
      objective_var += (real_objective_values[gomea_index][i] - objective_avg)*(real_objective_values[gomea_index][i] - objective_avg);
      constraint_var += (constraint_values[gomea_index][i] - constraint_avg)*(constraint_values[gomea_index][i] - constraint_avg);
    }
  }
  objective_var = objective_var / ((double) n);
  constraint_var = constraint_var / ((double) n);

  if( objective_var <= 0.0 )
     objective_var = 0.0;
  if( constraint_var <= 0.0 )
     constraint_var = 0.0;

  /* Then write them */
  sprintf( filename, "statistics.dat" );
  file = NULL;
  if( number_of_generations == 0 )
  {
    file = fopen( filename, "w" );

    sprintf( string, "Generation Evaluations Average-obj. Variance-obj. Best-obj. Worst-obj. Elite-obj. Average-con. Variance-con. Best-con. Worst-con. Elite-con.\n");
    fputs( string, file );
  }
  else
    file = fopen( filename, "a" );

  sprintf( string, "%d %ld %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n", number_of_generations, number_of_evaluations, objective_avg, objective_var, objective_best, objective_worst, elitist_solution_objective_value, constraint_avg, constraint_var, constraint_best, constraint_worst, elitist_solution_constraint_value );
  fputs( string, file );

  fclose( file );
}

/**
 * Writes the solutions to various files. The filenames
 * contain the generation. If the flag is_final_generation
 * is set the generation number in the filename
 * is replaced with the word "final".
 *
 * populations_xxxxx_generation_xxxxx.dat: the populations
 * offsprings_xxxxx_generation_xxxxx.dat : the offsprings
 */
void writeGenerationalSolutions( char is_final_generation )
{
  char  filename[1000], string[10000];
  int   i, j, gomea_index;
  FILE *file;

  /* Populations */
  if( is_final_generation )
    sprintf( filename, "populations_generation_final.dat" );
  else
    sprintf( filename, "populations_generation_%05d.dat", number_of_generations );
  file = fopen( filename, "w" );

  for( gomea_index = 0; gomea_index < number_of_GOMEAs; gomea_index++ )
  {
    for( i = 0; i < population_sizes[gomea_index]; i++ )
    {
      for( j = 0; j < number_of_parameters; j++ )
      {
        sprintf( string, "%d", populations[gomea_index][i][j] );
        fputs( string, file );
      }
      sprintf( string, "     %17.10e %17.10e\n", objective_values[gomea_index][i], constraint_values[gomea_index][i] );
      fputs( string, file );
    }
  }
  fclose( file );
  
  /* Offsprings */
  if( (number_of_generations > 0) && (!is_final_generation) )
  {
    sprintf( filename, "offsprings_generation_%05d.dat", number_of_generations-1 );
    file = fopen( filename, "w" );

    for( gomea_index = 0; gomea_index < number_of_GOMEAs; gomea_index++ )
    {
      for( i = 0; i < population_sizes[gomea_index]; i++ )
      {
        for( j = 0; j < number_of_parameters; j++ )
        {
          sprintf( string, "%d", offsprings[gomea_index][i][j] );
          fputs( string, file );
        }
        sprintf( string, "     %17.10e %17.10e\n", objective_values_offsprings[gomea_index][i], constraint_values_offsprings[gomea_index][i] );
        fputs( string, file );
      }
    }

    fclose( file );
  }
}

void writeElitistEvaluationsInit( char *filename )
{
  char  string[10000];
  FILE *file;

  file = fopen( filename, "w" );
  sprintf( string, "Evaluations Objective Constraint HittingTime SurrogateEvaluations\n");
  fputs( string, file );
  fclose( file );
}

void writeElitistEvaluations( char *filename, double elitist_objective, double elitist_constraint )
{
  char  string[10000];
  FILE *file;

  file = fopen( filename, "a" );

  sprintf( string, "%ld %f %f %lf %ld\n", number_of_evaluations, elitist_objective, elitist_constraint, (double)getMilliSecondsRunningAfterInit() / 1000.0, number_of_surrogate_evaluations );

  fputs( string, file );
  fclose( file );
}

void writeRunningTime( char *filename )
{
  char  string[10000];
  FILE *file;

  file = fopen( filename, "w" );
  // sprintf( string, "# Column 1: Total number of milliseconds.\n");
  // fputs( string, file );
  // sprintf( string, "# Column 2: Total number of milliseconds after initialization.\n");
  // fputs( string, file );
  // sprintf( string, "# Column 3: Total number of evaluations.\n"); 
  // fputs( string, file );
  // sprintf( string, "# Column 4: Total number of bit-flip evaluations.\n"); 
  // fputs( string, file );
#ifdef OS_WIN
  sprintf( string, "%ld %ld %ld %I64d\n", getMilliSecondsRunning(), getMilliSecondsRunningAfterInit(), number_of_evaluations, number_of_bit_flip_evaluations );
#else
  sprintf( string, "%ld %ld %ld %lld\n", getMilliSecondsRunning(), getMilliSecondsRunningAfterInit(), number_of_evaluations, number_of_bit_flip_evaluations );
#endif
  fputs( string, file );
  fclose( file );
}

void writeElitistSolution()
{
  char       string[10000];
  int        i;
  FILE      *file;

  file = fopen(filename_elitist_solution, "w" );
  sprintf( string, "# Column 1: Solution.\n");
  fputs( string, file );
  sprintf( string, "# Column 2: Objective value.\n");
  fputs( string, file );
  sprintf( string, "# Column 3: Constraint value.\n");
  fputs( string, file );
  for( i = 0; i < number_of_parameters; i++ )
  {
    sprintf( string, "%d", elitist_solution[i] );
    fputs( string, file );
  }
  sprintf( string, "     %17.10e %17.10e\n", elitist_solution_objective_value, elitist_solution_constraint_value );
  fputs( string, file );
  fclose( file );
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Termination -=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Returns 1 if termination should be enforced at the end of a generation, 0 otherwise.
 */
char checkTermination()
{
  if( checkNumberOfEvaluationsTerminationCondition() )
    return( 1 );
  
  if( checkNumberOfMilliSecondsTerminationCondition() )
    return( 1 );

  if( checkVOSOSTRTerminationCondition() )
      return( 1 );

  return( 0 );
}

/**
 * Returns 1 if the maximum number of evaluations
 * has been reached, 0 otherwise.
 */
char checkNumberOfEvaluationsTerminationCondition()
{
  if( (maximum_number_of_evaluations >= 0) && (number_of_evaluations >= maximum_number_of_evaluations) )
    return( 1 );

  return( 0 );
}

/**
 * Returns 1 if the value-to-reach has been reached.
 */
char checkVOSOSTRTerminationCondition()
{
  if( vosostr_hit_status > 0 )
    return( 1 );

  return( 0 );
}

/**
 * Returns 1 if the maximum number of milliseconds
 * has passed, 0 otherwise.
 */
char checkNumberOfMilliSecondsTerminationCondition()
{
  if( (maximum_number_of_milliseconds >= 0) && (getMilliSecondsRunning() > maximum_number_of_milliseconds) )
    return( 1 );

  return( 0 );
}

char checkTerminationSpecificGOMEA( int GOMEA_index )
{
  int i, j;
  computeAverageFitnessSpecificGOMEA( GOMEA_index );
  
  for( i = GOMEA_index+1; i < number_of_GOMEAs; i++ )
  {
    computeAverageFitnessSpecificGOMEA( i );

    if( betterFitness( average_objective_values[i], average_constraint_values[i], average_objective_values[GOMEA_index], average_constraint_values[GOMEA_index] ) )
    {
      minimum_GOMEA_index = GOMEA_index+1;

      return( 1 );
    }
  }

  for( i = 1; i < population_sizes[GOMEA_index]; i++ )
  {
    for( j = 0; j < number_of_parameters; j++ )
    {
      if( populations[GOMEA_index][i][j] != populations[GOMEA_index][0][j] )
        return( 0 );
    }
  }

  return( 1 );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-= Section Variation -==-=-=-=-=-=-=-=-=-=-=-=-=-=*/
void generationalStepAllGOMEAsRecursiveFold( int GOMEA_index_smallest, int GOMEA_index_biggest );
void generationalStepAllGOMEAs()
{
  int GOMEA_index_smallest, GOMEA_index_biggest;

  GOMEA_index_biggest  = number_of_GOMEAs-1;
  GOMEA_index_smallest = 0;
  while( GOMEA_index_smallest <= GOMEA_index_biggest )
  {
    if( !terminated[GOMEA_index_smallest] )
      break;

    GOMEA_index_smallest++;
  }
  //printf("inidices %d--%d\n", GOMEA_index_smallest, GOMEA_index_biggest);
  generationalStepAllGOMEAsRecursiveFold( GOMEA_index_smallest, GOMEA_index_biggest );
}

void generationalStepAllGOMEAsRecursiveFold( int GOMEA_index_smallest, int GOMEA_index_biggest )
{
  int i, GOMEA_index;
  for( i = 0; i < number_of_subgenerations_per_GOMEA_factor-1; i++ )
  {
    for( GOMEA_index = GOMEA_index_smallest; GOMEA_index <= GOMEA_index_biggest; GOMEA_index++ )
    {
      if( !terminated[GOMEA_index] )
      {
        terminated[GOMEA_index] = checkTerminationSpecificGOMEA( GOMEA_index );
      }

      if( (!terminated[GOMEA_index]) && (GOMEA_index >= minimum_GOMEA_index) )
      {
        makeOffspringSpecificGOMEA( GOMEA_index );

        selectFinalSurvivorsSpecificGOMEA( GOMEA_index );

        computeAverageFitnessSpecificGOMEA( GOMEA_index );

        makeRealEvaluationsSpecificGomea( GOMEA_index );

        gomeaUpdatesCounter[GOMEA_index]++;
      }
    }

    for( GOMEA_index = GOMEA_index_smallest; GOMEA_index < GOMEA_index_biggest; GOMEA_index++ )
      generationalStepAllGOMEAsRecursiveFold( GOMEA_index_smallest, GOMEA_index );
  }
}

void makeOffspringSpecificGOMEA( int gomea_index )
{
  learnFOSSpecificGOMEA( gomea_index );

  if( print_FOSs_contents )
  {
    printf("### FOS contents for GOMEA #%02d in generation #%03d\n", gomea_index, number_of_generations);
    printFOSContentsSpecificGOMEA( gomea_index );
    printf( "###################################################\n" );
  }

  generateAndEvaluateNewSolutionsToFillOffspringSpecificGOMEA( gomea_index );

  if (number_of_evaluations >= WARMUP_THRESHOLD)
  {
      if (previous_training_evaluation == -1)
      {
        double model_score = call_function_train_model(gomea_index);
        generateSolutionsToGetQuality(model_score);
      }
      else
      {  
        call_function_train_model(gomea_index);
      }

      updateSurrogateValuesAndElitist(gomea_index);
  }

}

/**
 * Returns the name of an installed problem.
 */
char *installedFOSStructureName( int index )
{
  switch( index )
  {
    case  0: return( (char *) "Univariate" );
    case  1: return( (char *) "Linkage Tree" );
    case  2: return( (char *) "Multiscale Linkage Neighbors" );
    case  3: return( (char *) "Linkage Trees and Neighbors" );
    case  4: return( (char *) "Filtered Linkage Tree" );
    case  5: return( (char *) "Filtered Multiscale Linkage Neighbors" );
    case  6: return( (char *) "Filtered Linkage Trees and Neighbors" );
    case  7: return( (char *) "Marginal Product Model" );
  }

  return( NULL );
}

/**
 * Returns the number of FOS structures installed.
 */
int numberOfInstalledFOSStructures()
{
  static int result = -1;
  
  if( result == -1 )
  {
    result = 0;
    while( installedFOSStructureName( result ) != NULL )
      result++;
  }
  
  return( result );
}

/**
 * Selects the FOS to be learned and calls the appropriate function to do
 * the learning.
 */
void learnFOSSpecificGOMEA( int gomea_index )
{
  int i;

  if( FOSs[gomea_index] != NULL )
  {
    for( i = 0; i < FOSs_length[gomea_index]; i++ )
      free( FOSs[gomea_index][i] );
    free( FOSs[gomea_index] );
    free( FOSs_number_of_indices[gomea_index] );
  }

  switch( FOSs_structure_index )
  {
    case  0: learnUnivariateFOSSpecificGOMEA( gomea_index ); break;
    case  1: learnLTFOSSpecificGOMEA( gomea_index, 1, 0, NULL ); break;
    case  2: learnMLNFOSSpecificGOMEA( gomea_index, 1, 0, NULL ); break;
    case  3: learnLTNFOSSpecificGOMEA( gomea_index ); break;
    case  4: learnFilteredLTFOSSpecificGOMEA( gomea_index, 1 ); break;
    case  5: learnFilteredMLNFOSSpecificGOMEA( gomea_index, 1 ); break;
    case  6: learnFilteredLTNFOSSpecificGOMEA( gomea_index ); break;
    case  7: learnMPMFOSSpecificGOMEA( gomea_index ); break;
  }

  switch( FOSs_structure_index )
  {
    case  0: break;
    case  1: break;
    case  2: uniquifyFOSSpecificGOMEA( gomea_index ); break;
    case  3: uniquifyFOSSpecificGOMEA( gomea_index ); break;
    case  4: break;
    case  5: uniquifyFOSSpecificGOMEA( gomea_index ); break;
    case  6: uniquifyFOSSpecificGOMEA( gomea_index ); break;
    case  7: break;
  }
}

/**
 * Learns a univariate FOS (randomized ordering of the singletons).
 */
void learnUnivariateFOSSpecificGOMEA( int gomea_index )
{
  int i, FOSs_index, *order;

  order                               = randomPermutation( number_of_parameters );
  FOSs_length[gomea_index]            = number_of_parameters;
  FOSs[gomea_index]                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  FOSs_index                           = 0;
  for( i = 0; i < number_of_parameters; i++ )
  {
    FOSs[gomea_index][FOSs_index]                   = (int *) Malloc( 1*sizeof( int ) );
    FOSs[gomea_index][FOSs_index][0]                = order[FOSs_index];
    FOSs_number_of_indices[gomea_index][FOSs_index] = 1;
    FOSs_index++;
  }

  free( order );
}

/**
 * Learns a linkage tree FOS by means of hierarchical clustering.
 * This implementation follows the reciprocal nearest neighbor approach.
 */
int **learnLTFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices, short compute_parent_child_relations, int *number_of_parent_child_relations )
{
  char     done;
  int      i, j, r0, r1, rswap, *indices, *order,
           FOSs_index, **mpm, *mpm_number_of_indices, mpm_length,
         **mpm_new, *mpm_new_number_of_indices, mpm_new_length,
          *NN_chain, NN_chain_length, **parent_child_relations,
           PCR_index, *FOSs_index_of_mpm_element;
  double **S_matrix, mul0, mul1;

  parent_child_relations   = NULL; /* Only needed to prevent compiler warnings. */
  PCR_index                = 0;    /* Only needed to prevent compiler warnings. */
  FOSs_index_of_mpm_element = NULL; /* Only needed to prevent compiler warnings. */
  if( compute_parent_child_relations )
  {
    *number_of_parent_child_relations = number_of_parameters-1;
    parent_child_relations = (int **) Malloc( (*number_of_parent_child_relations)*sizeof( int * ) );
    for( i = 0; i < (*number_of_parent_child_relations); i++ )
      parent_child_relations[i] = (int *) Malloc( 3*sizeof( int ) );
    FOSs_index_of_mpm_element = (int *) Malloc( number_of_parameters*sizeof( int ) );
    for( i = 0; i < number_of_parameters; i++ )
      FOSs_index_of_mpm_element[i] = i;
  }

  /* Compute Mutual Information matrix */
  if( compute_MI_matrices )
    computeMIMatrixSpecificGOMEA( gomea_index );

  /* Initialize MPM to the univariate factorization */
  order                 = randomPermutation( number_of_parameters );
  mpm                   = (int **) Malloc( number_of_parameters*sizeof( int * ) );
  mpm_number_of_indices = (int *) Malloc( number_of_parameters*sizeof( int ) );
  mpm_length            = number_of_parameters;
  for( i = 0; i < number_of_parameters; i++ )
  {
    indices                  = (int *) Malloc( 1*sizeof( int ) );
    indices[0]               = order[i];
    mpm[i]                   = indices;
    mpm_number_of_indices[i] = 1;
  }
  free( order );

  /* Initialize LT to the initial MPM */
  FOSs_length[gomea_index]            = number_of_parameters+number_of_parameters-1;
  FOSs[gomea_index]                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  FOSs_index                                             = 0;
  for( i = 0; i < mpm_length; i++ )
  {
    FOSs[gomea_index][FOSs_index]                   = mpm[i];
    FOSs_number_of_indices[gomea_index][FOSs_index] = mpm_number_of_indices[i];
    FOSs_index++;
  }

  /* Initialize similarity matrix */
  S_matrix = (double **) Malloc( number_of_parameters*sizeof( double * ) );
  for( i = 0; i < number_of_parameters; i++ )
    S_matrix[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
  for( i = 0; i < mpm_length; i++ )
    for( j = 0; j < mpm_length; j++ )
      S_matrix[i][j] = MI_matrices[gomea_index][mpm[i][0]][mpm[j][0]];
  for( i = 0; i < mpm_length; i++ )
    S_matrix[i][i] = 0;

  NN_chain        = (int *) Malloc( (number_of_parameters+2)*sizeof( int ) );
  NN_chain_length = 0;
  done            = 0;
  while( !done )
  {
    if( NN_chain_length == 0 )
    {
      NN_chain[NN_chain_length] = randomInt( mpm_length );
      NN_chain_length++;
    }

    while( NN_chain_length < 3 )
    {
      NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length );
      NN_chain_length++;
    }

    while( NN_chain[NN_chain_length-3] != NN_chain[NN_chain_length-1] )
    {
      NN_chain[NN_chain_length] = determineNearestNeighbour( NN_chain[NN_chain_length-1], S_matrix, mpm_number_of_indices, mpm_length );
      if( ((S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length]] == S_matrix[NN_chain[NN_chain_length-1]][NN_chain[NN_chain_length-2]])) && (NN_chain[NN_chain_length] != NN_chain[NN_chain_length-2]) )
        NN_chain[NN_chain_length] = NN_chain[NN_chain_length-2];
      NN_chain_length++;
      if( NN_chain_length > number_of_parameters )
        break;
    }
    r0 = NN_chain[NN_chain_length-2];
    r1 = NN_chain[NN_chain_length-1];
    if( r0 > r1 )
    {
      rswap = r0;
      r0    = r1;
      r1    = rswap;
    }
    NN_chain_length -= 3;

    if( r1 < mpm_length ) /* This test is required for exceptional cases in which the nearest-neighbor ordering has changed within the chain while merging within that chain */
    {
      indices = (int *) Malloc( (mpm_number_of_indices[r0]+mpm_number_of_indices[r1])*sizeof( int ) );
  
      i = 0;
      for( j = 0; j < mpm_number_of_indices[r0]; j++ )
      {
        indices[i] = mpm[r0][j];
        i++;
      }
      for( j = 0; j < mpm_number_of_indices[r1]; j++ )
      {
        indices[i] = mpm[r1][j];
        i++;
      }
    
      if( compute_parent_child_relations )
      {
        parent_child_relations[PCR_index][0] = FOSs_index;
        parent_child_relations[PCR_index][1] = FOSs_index_of_mpm_element[r0];
        parent_child_relations[PCR_index][2] = FOSs_index_of_mpm_element[r1];
        FOSs_index_of_mpm_element[r0]         = FOSs_index;
        FOSs_index_of_mpm_element[r1]         = FOSs_index_of_mpm_element[mpm_length-1];
        PCR_index++;
      }
      FOSs[gomea_index][FOSs_index]                   = indices;
      FOSs_number_of_indices[gomea_index][FOSs_index] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
      FOSs_index++;
  
      mul0 = ((double) mpm_number_of_indices[r0])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
      mul1 = ((double) mpm_number_of_indices[r1])/((double) mpm_number_of_indices[r0]+mpm_number_of_indices[r1]);
      for( i = 0; i < mpm_length; i++ )
      {
        if( (i != r0) && (i != r1) )
        {
          S_matrix[i][r0] = mul0*S_matrix[i][r0] + mul1*S_matrix[i][r1];
          S_matrix[r0][i] = S_matrix[i][r0];
        }
      }
  
      mpm_new                   = (int **) Malloc( (mpm_length-1)*sizeof( int * ) );
      mpm_new_number_of_indices = (int *) Malloc( (mpm_length-1)*sizeof( int ) );
      mpm_new_length            = mpm_length-1;
      for( i = 0; i < mpm_new_length; i++ )
      {
        mpm_new[i]                   = mpm[i];
        mpm_new_number_of_indices[i] = mpm_number_of_indices[i];
      }
  
      mpm_new[r0]                   = indices;
      mpm_new_number_of_indices[r0] = mpm_number_of_indices[r0]+mpm_number_of_indices[r1];
      if( r1 < mpm_length-1 )
      {
        mpm_new[r1]                   = mpm[mpm_length-1];
        mpm_new_number_of_indices[r1] = mpm_number_of_indices[mpm_length-1];
  
        for( i = 0; i < r1; i++ )
        {
          S_matrix[i][r1] = S_matrix[i][mpm_length-1];
          S_matrix[r1][i] = S_matrix[i][r1];
        }
  
        for( j = r1+1; j < mpm_new_length; j++ )
        {
          S_matrix[r1][j] = S_matrix[j][mpm_length-1];
          S_matrix[j][r1] = S_matrix[r1][j];
        }
      }
  
      for( i = 0; i < NN_chain_length; i++ )
      {
        if( NN_chain[i] == mpm_length-1 )
        {
          NN_chain[i] = r1;
          break;
        }
      }
  
      free( mpm );
      free( mpm_number_of_indices );
      mpm                   = mpm_new;
      mpm_number_of_indices = mpm_new_number_of_indices;
      mpm_length            = mpm_new_length;
  
      if( mpm_length == 1 )
        done = 1;
    }
  }

  free( NN_chain );

  free( mpm_new );
  free( mpm_number_of_indices );

  for( i = 0; i < number_of_parameters; i++ )
    free( S_matrix[i] );
  free( S_matrix );

  free( FOSs_index_of_mpm_element );

  return( parent_child_relations );
}

/**
 * Determines nearest neighbour according to similarity values.
 */
int determineNearestNeighbour( int index, double **S_matrix, int *mpm_number_of_indices, int mpm_length )
{
  int i, result;

  result = 0;
  if( result == index )
    result++;
  for( i = 1; i < mpm_length; i++ )
  {
    if( ((S_matrix[index][i] > S_matrix[index][result]) || ((S_matrix[index][i] == S_matrix[index][result]) && (mpm_number_of_indices[i] < mpm_number_of_indices[result]))) && (i != index) )
      result = i;
  }

  return( result );
}

/**
 * Learns a multiscale linkage neighbors FOS.
 */
int **learnMLNFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices, short compute_parent_child_relations, int *number_of_parent_child_relations )
{
  int    i, j, k, k2, k3, q, q2, **neighbors, ***buckets, **bucket_sizes, bucket_index, number_of_buckets,
         PCR_index, **parent_child_relations, **parent_child_relations_temp, number_of_parent_child_relations_temp, parent_child_relations_temp_size;
  double MI_max, MI_ratio;

  parent_child_relations           = NULL; /* Only needed to prevent compiler warnings. */
  parent_child_relations_temp      = NULL; /* Only needed to prevent compiler warnings. */
  PCR_index                        = 0;    /* Only needed to prevent compiler warnings. */
  parent_child_relations_temp_size = 0;    /* Only needed to prevent compiler warnings. */
  number_of_buckets                = 1+sqrt( population_sizes[gomea_index] );
  if( compute_parent_child_relations )
  {
    number_of_parent_child_relations_temp = 0;
    parent_child_relations_temp_size      = number_of_parameters*number_of_buckets;
    parent_child_relations_temp           = (int **) Malloc( (parent_child_relations_temp_size)*sizeof( int * ) );
    for( i = 0; i < parent_child_relations_temp_size; i++ )
      parent_child_relations_temp[i] = (int *) Malloc( 3*sizeof( int ) );
  }

  /* Compute Mutual Information matrix */
  if( compute_MI_matrices )
    computeMIMatrixSpecificGOMEA( gomea_index );

  /* Create a random ordering of neighbors for each variable */
  neighbors = (int **) Malloc( number_of_parameters*sizeof( int * ) );
  for( i = 0; i < number_of_parameters; i++ )
    neighbors[i] = randomPermutation( number_of_parameters );

  /* Determine for each variable i a particular ordering: a bucket sort without sorted buckets */
  buckets           = (int ***) Malloc( number_of_parameters*sizeof( int ** ) );
  bucket_sizes      = (int **) Malloc( number_of_parameters*sizeof( int * ) );
  for( i = 0; i < number_of_parameters; i++ )
  {
    buckets[i]      = NULL;
    bucket_sizes[i] = NULL;

    buckets[i]      = (int **) Malloc( number_of_buckets*sizeof( int * ) );
    bucket_sizes[i] = (int *) Malloc( number_of_buckets*sizeof( int ) );

    MI_max = 0;
    for( j = 0; j < number_of_parameters; j++ )
    {
      if( neighbors[i][j] != i )
      {
        if( MI_matrices[gomea_index][i][neighbors[i][j]] > MI_max )
          MI_max = MI_matrices[gomea_index][i][neighbors[i][j]];
      }
    }
    MI_max = MI_max > 1 ? 1 : MI_max;

    for( bucket_index = 0; bucket_index < number_of_buckets; bucket_index++ )
      bucket_sizes[i][bucket_index] = 0;
    for( j = 0; j < number_of_parameters; j++ )
    {
      if( neighbors[i][j] == i )
        bucket_index = number_of_buckets-1;
      else
      {
        MI_ratio = 1;
        if( MI_max > 0 )
          MI_ratio = MI_matrices[gomea_index][i][neighbors[i][j]]/MI_max;
        bucket_index = MI_ratio*((double) (number_of_buckets-1));
        bucket_index = bucket_index >= (number_of_buckets-1) ? (number_of_buckets-1)-1 : bucket_index;
        bucket_index = bucket_index <= 0 ? 0 : bucket_index;
      }
      bucket_sizes[i][bucket_index]++;
    }

    for( bucket_index = 0; bucket_index < number_of_buckets; bucket_index++ )
    {
      buckets[i][bucket_index] = NULL;
      if( bucket_sizes[i][bucket_index] > 0 )
        buckets[i][bucket_index] = (int *) Malloc( bucket_sizes[i][bucket_index]*sizeof( int ) );
    }

    for( bucket_index = 0; bucket_index < number_of_buckets; bucket_index++ )
      bucket_sizes[i][bucket_index] = 0;
    for( j = 0; j < number_of_parameters; j++ )
    {
      if( neighbors[i][j] == i )
        bucket_index = number_of_buckets-1;
      else
      {
        MI_ratio = 1;
        if( MI_max > 0 )
          MI_ratio = MI_matrices[gomea_index][i][neighbors[i][j]]/MI_max;
        bucket_index = MI_ratio*((double) (number_of_buckets-1));
        bucket_index = bucket_index >= (number_of_buckets-1) ? (number_of_buckets-1)-1 : bucket_index;
        bucket_index = bucket_index <= 0 ? 0 : bucket_index;
      }
      buckets[i][bucket_index][bucket_sizes[i][bucket_index]] = neighbors[i][j];
      bucket_sizes[i][bucket_index]++;
    }
  }

  FOSs_length[gomea_index] = 0;
  for( i = 0; i < number_of_parameters; i++ )
  {
    if( number_of_parameters == 1 )
      FOSs_length[gomea_index]++;
    else
    {
      if( bucket_sizes[i][number_of_buckets-1] > 1 )
        FOSs_length[gomea_index]++;
      q  = bucket_sizes[i][number_of_buckets-1];
      q2 = q;
      for( j = number_of_buckets-2; j >= 0; j-- )
      {
        q2 += bucket_sizes[i][j];
        if( ((q2 >= (2*q)) || (q2 == number_of_parameters)) && (q2 <= (number_of_parameters/2)) )
        {
          q = q2;
          if( compute_parent_child_relations )
          {
            parent_child_relations_temp[PCR_index][0] = FOSs_length[gomea_index];
            parent_child_relations_temp[PCR_index][1] = FOSs_length[gomea_index]-1;
            parent_child_relations_temp[PCR_index][2] = -1;
            PCR_index++;
          }
          FOSs_length[gomea_index]++;
        }
        if( q2 == number_of_parameters )
          break;
      }
    }
  }

  /* Create the multiscale set of neighbors for each variable */
  FOSs[gomea_index]                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  
  FOSs_length[gomea_index] = 0;
  for( i = 0; i < number_of_parameters; i++ )
  {
    if( number_of_parameters == 1 )
    {
      FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]] = 1;
      FOSs[gomea_index][FOSs_length[gomea_index]]                   = (int *) Malloc( FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]]*sizeof( int ) );
      FOSs[gomea_index][FOSs_length[gomea_index]][0]                = neighbors[i][0];
      FOSs_length[gomea_index]++;
    }
    else
    {
      if( bucket_sizes[i][number_of_buckets-1] > 1 )
      {
        FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]] = bucket_sizes[i][number_of_buckets-1];
        FOSs[gomea_index][FOSs_length[gomea_index]]                   = (int *) Malloc( FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]]*sizeof( int ) );
        for( k = 0; k < bucket_sizes[i][number_of_buckets-1]; k++ )
          FOSs[gomea_index][FOSs_length[gomea_index]][k] = buckets[i][number_of_buckets-1][k];
        FOSs_length[gomea_index]++;
      }
      q  = bucket_sizes[i][number_of_buckets-1];
      q2 = q;
      for( j = number_of_buckets-2; j >= 0; j-- )
      {
        q2 += bucket_sizes[i][j];
        if( ((q2 >= (2*q)) || (q2 == number_of_parameters)) && (q2 <= (number_of_parameters/2)) )
        {
          q = q2;
          FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]] = q;
          FOSs[gomea_index][FOSs_length[gomea_index]]                   = (int *) Malloc( FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]]*sizeof( int ) );
          k                                 = 0;
          for( k2 = number_of_buckets-1; k2 >= j; k2-- )
          {
            for( k3 = 0; k3 < bucket_sizes[i][k2]; k3++ )
            {
              FOSs[gomea_index][FOSs_length[gomea_index]][k] = buckets[i][k2][k3];
              k++;
            }
          }
          FOSs_length[gomea_index]++;
        }
        if( q2 == number_of_parameters )
          break;
      }
    }
  }

  for( i = 0; i < number_of_parameters; i++ )
  {
    if( buckets[i] != NULL )
    {
      for( j = 0; j < number_of_buckets; j++ )
      {
        if( buckets[i][j] != NULL )
        {
          free( buckets[i][j] );
        }
      }
      free( buckets[i] );
      free( bucket_sizes[i] );
    }
  }
  free( bucket_sizes );
  free( buckets );

  for( i = 0; i < number_of_parameters; i++ )
  {
    if( neighbors[i] != NULL )
    {
      free( neighbors[i] );
    }
  }
  free( neighbors );

  if( compute_parent_child_relations )
  {
    number_of_parent_child_relations_temp = PCR_index;
    *number_of_parent_child_relations     = number_of_parent_child_relations_temp;

    parent_child_relations = (int **) Malloc( (*number_of_parent_child_relations)*sizeof( int * ) );
    for( i = 0; i < (*number_of_parent_child_relations); i++ )
    {
      parent_child_relations[i] = (int *) Malloc( 3*sizeof( int ) );
      for( j = 0; j < 3; j++ )
        parent_child_relations[i][j] = parent_child_relations_temp[i][j];
    }
    for( i = 0; i < parent_child_relations_temp_size; i++ )
      free( parent_child_relations_temp[i] );
    free( parent_child_relations_temp );
  }
  return( parent_child_relations );
}


/**
 * Learns a multiscale linkage neighbors FOS.
 */
void learnLTNFOSSpecificGOMEA( int gomea_index )
{
  learnLTNFOSWithOrWithoutFilteringSpecificGOMEA( gomea_index, 0 );
}

/**
 * Learns a multiscale linkage neighbors FOS with or without filtering. The actual
 * work is done here.
 */
void learnLTNFOSWithOrWithoutFilteringSpecificGOMEA( int gomea_index, short use_filtering )
{
  int i, j, **LT_FOS, **MLN_FOS, *LT_FOS_number_of_indices, *MLN_FOS_number_of_indices, LT_FOS_length, MLN_FOS_length;

  /* Learn the LT FOS and create a backup copy */
  if( use_filtering )
    learnFilteredLTFOSSpecificGOMEA( gomea_index, 1 );
  else
    learnLTFOSSpecificGOMEA( gomea_index, 1, 0, NULL );
  
  LT_FOS_length            = FOSs_length[gomea_index];
  LT_FOS_number_of_indices = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  LT_FOS                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    LT_FOS_number_of_indices[i] = FOSs_number_of_indices[gomea_index][i];
    LT_FOS[i] = (int *) Malloc( FOSs_number_of_indices[gomea_index][i]*sizeof( int ) );
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      LT_FOS[i][j] = FOSs[gomea_index][i][j];
  }

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    free( FOSs[gomea_index][i] );
  free( FOSs[gomea_index] );
  free( FOSs_number_of_indices[gomea_index] );

  /* Learn the MLN FOS and create a backup copy */
  if( use_filtering )
    learnFilteredMLNFOSSpecificGOMEA( gomea_index, 0 );
  else
    learnMLNFOSSpecificGOMEA( gomea_index, 0, 0, NULL );

  MLN_FOS_length            = FOSs_length[gomea_index];
  MLN_FOS_number_of_indices = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  MLN_FOS                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    MLN_FOS_number_of_indices[i] = FOSs_number_of_indices[gomea_index][i];
    MLN_FOS[i] = (int *) Malloc( FOSs_number_of_indices[gomea_index][i]*sizeof( int ) );
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      MLN_FOS[i][j] = FOSs[gomea_index][i][j];
  }

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    free( FOSs[gomea_index][i] );
  free( FOSs[gomea_index] );
  free( FOSs_number_of_indices[gomea_index] );

  /* Construct the LTN FOS: join the LT FOS and the MLN FOS */
  FOSs_length[gomea_index]            = LT_FOS_length + MLN_FOS_length; 
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( (LT_FOS_length + MLN_FOS_length)*sizeof( int ) );
  FOSs[gomea_index]                   = (int **) Malloc( (LT_FOS_length + MLN_FOS_length)*sizeof( int * ) );
  for( i = 0; i < LT_FOS_length; i++ )
  {
    FOSs_number_of_indices[gomea_index][i] = LT_FOS_number_of_indices[i];
    FOSs[gomea_index][i] = (int *) Malloc( LT_FOS_number_of_indices[i]*sizeof( int ) );
    for( j = 0; j < LT_FOS_number_of_indices[i]; j++ )
      FOSs[gomea_index][i][j] = LT_FOS[i][j];
  }
  for( i = 0; i < MLN_FOS_length; i++ )
  {
    FOSs_number_of_indices[gomea_index][LT_FOS_length+i] = MLN_FOS_number_of_indices[i];
    FOSs[gomea_index][LT_FOS_length+i] = (int *) Malloc( MLN_FOS_number_of_indices[i]*sizeof( int ) );
    for( j = 0; j < MLN_FOS_number_of_indices[i]; j++ )
      FOSs[gomea_index][LT_FOS_length+i][j] = MLN_FOS[i][j];
  }

  /* Free up backup memory */
  for( i = 0; i < LT_FOS_length; i++ )
    free( LT_FOS[i] );
  free( LT_FOS_number_of_indices );
  free( LT_FOS );
  for( i = 0; i < MLN_FOS_length; i++ )
    free( MLN_FOS[i] );
  free( MLN_FOS_number_of_indices );
  free( MLN_FOS );
}

void learnFilteredLTFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices )
{
  int i, **parent_child_relations, number_of_parent_child_relations;

  parent_child_relations = learnLTFOSSpecificGOMEA( gomea_index, compute_MI_matrices, 1, &number_of_parent_child_relations );

  filterParentChildRelationsAndCreateNewFOSSpecificGOMEA( gomea_index, parent_child_relations, number_of_parent_child_relations );

  for( i = 0; i < number_of_parent_child_relations; i++ )
    free( parent_child_relations[i] );
  free( parent_child_relations );
}

void learnFilteredMLNFOSSpecificGOMEA( int gomea_index, short compute_MI_matrices )
{
  int i, **parent_child_relations, number_of_parent_child_relations;

  parent_child_relations = learnMLNFOSSpecificGOMEA( gomea_index, compute_MI_matrices, 1, &number_of_parent_child_relations );

  filterParentChildRelationsAndCreateNewFOSSpecificGOMEA( gomea_index, parent_child_relations, number_of_parent_child_relations );

  for( i = 0; i < number_of_parent_child_relations; i++ )
    free( parent_child_relations[i] );
  free( parent_child_relations );
}

/**
 * Learns a multiscale linkage neighbors FOS.
 */
void learnFilteredLTNFOSSpecificGOMEA( int gomea_index )
{
  learnLTNFOSWithOrWithoutFilteringSpecificGOMEA( gomea_index, 1 );
}

void filterParentChildRelationsAndCreateNewFOSSpecificGOMEA( int gomea_index, int **parent_child_relations, int number_of_parent_child_relations )
{
  char   *FOS_element_accepted, *linkage_strength_computed;
  int     i, j, parent_index, child0_index, child1_index, **FOS, *FOS_number_of_indices, FOS_length;
  double *linkage_strength;

  FOS_element_accepted = (char *) Malloc( FOSs_length[gomea_index]*sizeof( char ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    FOS_element_accepted[i] = 1;

  linkage_strength_computed = (char *) Malloc( FOSs_length[gomea_index]*sizeof( char ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    linkage_strength_computed[i] = 0;

  linkage_strength = (double *) Malloc( FOSs_length[gomea_index]*sizeof( double ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    linkage_strength[i] = 0;

  for( i = 0; i < number_of_parent_child_relations; i++ )
  {
    parent_index = parent_child_relations[i][0];
    child0_index = parent_child_relations[i][1];
    child1_index = parent_child_relations[i][2];

    if( !linkage_strength_computed[parent_index] )
    {
      linkage_strength[parent_index]          = computeLinkageStrengthSpecificGOMEA( gomea_index, FOSs[gomea_index][parent_index], FOSs_number_of_indices[gomea_index][parent_index] );
      linkage_strength_computed[parent_index] = 1;
    }
    if( child0_index != -1 )
    {
      if( !linkage_strength_computed[child0_index] )
      {
        linkage_strength[child0_index]          = computeLinkageStrengthSpecificGOMEA( gomea_index, FOSs[gomea_index][child0_index], FOSs_number_of_indices[gomea_index][child0_index] );
        linkage_strength_computed[child0_index] = 1;
      }
    }
    if( child1_index != -1 )
    {
      if( !linkage_strength_computed[child1_index] )
      {
        linkage_strength[child1_index]          = computeLinkageStrengthSpecificGOMEA( gomea_index, FOSs[gomea_index][child1_index], FOSs_number_of_indices[gomea_index][child1_index] );
        linkage_strength_computed[child1_index] = 1;
      }
    }

    /* Remove each child if it has the same linkage strength as its parent */
    if( child0_index != -1 )
    {
      if( ((linkage_strength[parent_index] >= (1.0-1e-100)*linkage_strength[child0_index])) && ((linkage_strength[parent_index] <= (1.0+1e-100)*linkage_strength[child0_index])) )
        FOS_element_accepted[child0_index] = 0;
    }
    if( child1_index != -1 )
    {
      if( ((linkage_strength[parent_index] >= (1.0-1e-100)*linkage_strength[child1_index])) && ((linkage_strength[parent_index] <= (1.0+1e-100)*linkage_strength[child1_index])) )
        FOS_element_accepted[child1_index] = 0;
    }
  }

  /* Create a backup copy of the FOS */
  FOS_length            = FOSs_length[gomea_index];
  FOS_number_of_indices = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  FOS                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    FOS_number_of_indices[i] = FOSs_number_of_indices[gomea_index][i];
    FOS[i] = (int *) Malloc( FOSs_number_of_indices[gomea_index][i]*sizeof( int ) );
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      FOS[i][j] = FOSs[gomea_index][i][j];
  }

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    free( FOSs[gomea_index][i] );
  free( FOSs[gomea_index] );
  free( FOSs_number_of_indices[gomea_index] );

  /* Apply filters */
  FOSs_length[gomea_index] = 0;
  for( i = 0; i < FOS_length; i++ )
    if( FOS_element_accepted[i] )
      FOSs_length[gomea_index]++;
 
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  FOSs[gomea_index]                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_length[gomea_index]            = 0;
  for( i = 0; i < FOS_length; i++ )
  {
    if( FOS_element_accepted[i] )
    {
      FOSs_number_of_indices[gomea_index][FOSs_length[gomea_index]] = FOS_number_of_indices[i];
      FOSs[gomea_index][FOSs_length[gomea_index]]                   = (int *) Malloc( FOS_number_of_indices[i]*sizeof( int ) );
      for( j = 0; j < FOS_number_of_indices[i]; j++ )
        FOSs[gomea_index][FOSs_length[gomea_index]][j] = FOS[i][j];
      FOSs_length[gomea_index]++;
    }
  }

  /* Free up backup memory */
  for( i = 0; i < FOS_length; i++ )
    free( FOS[i] );
  free( FOS_number_of_indices );
  free( FOS );

  free( linkage_strength );
  free( linkage_strength_computed );
  free( FOS_element_accepted );
}

double computeLinkageStrengthSpecificGOMEA( int gomea_index, int *variables, int number_of_variables )
{
  int    i, j, n;
  double result;

  result = MI_matrices[gomea_index][variables[0]][variables[0]];
  if( number_of_variables > 1 )
  {
    result = 0;
    n      = 0;
    for( i = 0; i < number_of_variables; i++ )
    {
      for( j = i+1; j < number_of_variables; j++ )
      {
        result += MI_matrices[gomea_index][variables[i]][variables[j]];
        n++;
      }
    }
    result /= (double) n;
  }

  return( result );
}

void computeMIMatrixSpecificGOMEA( int gomea_index )
{
  int    i, j, k, *indices, factor_size;
  double p, *cumulative_probabilities;

  /* Compute joint entropy matrix */
  for( i = 0; i < number_of_parameters; i++ )
  {
    for( j = i+1; j < number_of_parameters; j++ )
    {
      indices                  = (int *) Malloc( 2*sizeof( int ) );
      indices[0]               = i;
      indices[1]               = j;
      cumulative_probabilities = estimateParametersForSingleBinaryMarginalSpecificGOMEA( gomea_index, indices, 2, &factor_size );

      MI_matrices[gomea_index][i][j] = 0.0;
      for( k = 0; k < factor_size; k++ )
      {
        if( k == 0 )
          p = cumulative_probabilities[k];
        else
          p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
        if( p > 0 )
          MI_matrices[gomea_index][i][j] += -p*log2(p);
      }

      MI_matrices[gomea_index][j][i] = MI_matrices[gomea_index][i][j];

      free( indices );
      free( cumulative_probabilities );
    }
    indices                  = (int *) Malloc( 1*sizeof( int ) );
    indices[0]               = i;
    cumulative_probabilities = estimateParametersForSingleBinaryMarginalSpecificGOMEA( gomea_index, indices, 1, &factor_size );

    MI_matrices[gomea_index][i][i] = 0.0;
    for( k = 0; k < factor_size; k++ )
    {
      if( k == 0 )
        p = cumulative_probabilities[k];
      else
        p = cumulative_probabilities[k]-cumulative_probabilities[k-1];
      if( p > 0 )
       MI_matrices[gomea_index][i][i] += -p*log2(p);
    }

    free( indices );
    free( cumulative_probabilities );
  }

  /* Then transform into mutual information matrix MI(X,Y)=H(X)+H(Y)-H(X,Y) */
  for( i = 0; i < number_of_parameters; i++ )
    for( j = i+1; j < number_of_parameters; j++ )
    {
      MI_matrices[gomea_index][i][j] = MI_matrices[gomea_index][i][i] + MI_matrices[gomea_index][j][j] - MI_matrices[gomea_index][i][j];
      MI_matrices[gomea_index][j][i] = MI_matrices[gomea_index][i][j];
    }
}

/**
 * Estimates the cumulative probability distribution of a
 * single binary marginal.
 */
double *estimateParametersForSingleBinaryMarginalSpecificGOMEA( int gomea_index, int *indices, int number_of_indices, int *factor_size )
{
  int     i, j, index, power_of_two;
  double *result;

  *factor_size = (int) pow( 2, number_of_indices );
  result       = (double *) Malloc( (*factor_size)*sizeof( double ) );

  for( i = 0; i < (*factor_size); i++ )
    result[i] = 0.0;

  for( i = 0; i < population_sizes[gomea_index]; i++ )
  {
    index        = 0;
    power_of_two = 1;
    for( j = number_of_indices-1; j >= 0; j-- )
    {
      index += populations[gomea_index][i][indices[j]] ? power_of_two : 0;
      power_of_two *= 2;
    }

    result[index] += 1.0;
  }

  for( i = 0; i < (*factor_size); i++ )
    result[i] /= (double) population_sizes[gomea_index];

  for( i = 1; i < (*factor_size); i++ )
    result[i] += result[i-1];

  result[(*factor_size)-1] = 1.0;

  return( result );
}

int cmpfun (const void * a, const void * b)
{
   double x = *(double*)a, y = *(double*)b;
   if (x < y)
    return 1;
   if (x > y)
    return -1;
   return 0;
}


/**
 * Ensures that every FOS element is unique, i.e. there are no
 * duplicate linkage subsets.
 */
void uniquifyFOSSpecificGOMEA( int gomea_index )
{
  short *FOSs_subset_is_duplicate;
  int    i, j, k, q, *sorted, **FOSs_new, *FOSs_number_of_indices_new, FOSs_length_new;

  /* First analyze which subsets are duplicates */
  FOSs_subset_is_duplicate = (short *) Malloc( FOSs_length[gomea_index]*sizeof( short ) );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    FOSs_subset_is_duplicate[i] = 0;

  sorted = mergeSortIntegersDecreasing( FOSs_number_of_indices[gomea_index], FOSs_length[gomea_index] );

  i = 0;
  while( i < FOSs_length[gomea_index] )
  {
    /* Determine stretch of FOS elements that have the same length */
    j = i+1;
    while( (j < FOSs_length[gomea_index]) && (FOSs_number_of_indices[gomea_index][sorted[i]] == FOSs_number_of_indices[gomea_index][sorted[j]]) )
      j++;

    /* Check inside stretch for duplicates */
    for( k = i; k < j-1; k++ )
    {
      if( FOSs_subset_is_duplicate[sorted[k]] )
        continue;

      for( q = k+1; q < j; q++ )
      {
        if( FOSs_subset_is_duplicate[sorted[q]] )
          continue;

        if( linkageSubsetsOfSameLengthAreDuplicates( FOSs[gomea_index][sorted[k]], FOSs[gomea_index][sorted[q]], FOSs_number_of_indices[gomea_index][sorted[k]] ) )
          FOSs_subset_is_duplicate[sorted[q]] = 1;
      }
    }
    i = j;
  }

  /* Then re-create the FOS without the duplicate sets */
  FOSs_length_new = 0;
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    if( !FOSs_subset_is_duplicate[i] )
      FOSs_length_new++;
  }

  FOSs_new                   = (int **) Malloc( FOSs_length_new*sizeof( int * ) );
  FOSs_number_of_indices_new = (int *) Malloc( FOSs_length_new*sizeof( int ) );

  j = 0;
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    if( !FOSs_subset_is_duplicate[i] )
    {
      FOSs_new[j] = (int *) Malloc( FOSs_number_of_indices[gomea_index][i]*sizeof( int ) );
      for( k = 0; k < FOSs_number_of_indices[gomea_index][i]; k++ )
        FOSs_new[j][k] = FOSs[gomea_index][i][k];

      FOSs_number_of_indices_new[j] = FOSs_number_of_indices[gomea_index][i];

      j++;
    }
  }

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
    free( FOSs[gomea_index][i] );
  free( FOSs[gomea_index] );
  free( FOSs_number_of_indices[gomea_index] );
  FOSs[gomea_index]                   = FOSs_new;
  FOSs_number_of_indices[gomea_index] = FOSs_number_of_indices_new;
  FOSs_length[gomea_index]            = FOSs_length_new;

  free( sorted );
  free( FOSs_subset_is_duplicate );
}

short linkageSubsetsOfSameLengthAreDuplicates( int *linkageSubset0, int *linkageSubset1, int length )
{
  short result, *linkageSubset0AsBitString, *linkageSubset1AsBitString;
  int   i, *sorted0, *sorted1;

  result = 0;
  if( length == 1 )
  {
    if( linkageSubset0[0] == linkageSubset1[0] )
      result = 1;
  }
  else if( length == number_of_parameters )
  {
    result = 1;
  }
  else if( length <= (number_of_parameters/2) )
  {
    sorted0 = mergeSortIntegersDecreasing( linkageSubset0, length );
    sorted1 = mergeSortIntegersDecreasing( linkageSubset1, length );
    result = 1;
    for( i = 0; i < length; i++ )
    {
      if( linkageSubset0[sorted0[i]] != linkageSubset1[sorted1[i]] )
      {
        result = 0;
        break;
      }
    }
    free( sorted0 );
    free( sorted1 );
  }
  else
  {
    linkageSubset0AsBitString = (short *) Malloc( number_of_parameters*sizeof( short ) );
    linkageSubset1AsBitString = (short *) Malloc( number_of_parameters*sizeof( short ) );
    for( i = 0; i < number_of_parameters; i++ )
    {
      linkageSubset0AsBitString[i] = 0;
      linkageSubset1AsBitString[i] = 0;
    }
    for( i = 0; i < length; i++ )
    {
      linkageSubset0AsBitString[linkageSubset0[i]] = 1;
      linkageSubset1AsBitString[linkageSubset1[i]] = 1;
    }
    result = 1;
    for( i = 0; i < number_of_parameters; i++ )
    {
      if( linkageSubset0AsBitString[i] != linkageSubset1AsBitString[i] )
      {
        result = 0;
        break;
      }
    }
    free( linkageSubset0AsBitString );
    free( linkageSubset1AsBitString );
  }

  return( result );
}

/**
 * Learns a marginal product model FOS by means of iterative merging and MDL.
 */
void learnMPMFOSSpecificGOMEA( int gomea_index )
{
  int      i, j, mpm_length;
  binmarg *mpm;

  mpm = learnMPMSpecificGOMEA( gomea_index, &mpm_length );

  FOSs_length[gomea_index]            = mpm_length;
  FOSs[gomea_index]                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_number_of_indices[gomea_index] = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  for( i = 0; i < mpm_length; i++ )
  {
    FOSs[gomea_index][i] = (int *) Malloc( mpm[i].number_of_indices*sizeof( int ) );
    for( j = 0; j < mpm[i].number_of_indices; j++ )
      FOSs[gomea_index][i][j] = mpm[i].indices[j];
    FOSs_number_of_indices[gomea_index][i] = mpm[i].number_of_indices;
  }

  for( i = 0; i < mpm_length; i++ )
  {
    free( mpm[i].indices );
    free( mpm[i].cumulative_probabilities );
  }
  free( mpm );
}

binmarg *learnMPMSpecificGOMEA( int gomea_index, int *mpm_length )
{
  char     done;
  int      i, j, a, b, c, a_max, b_max, *order, number_of_indices, *indices;
  double **metric_reduction_after_merge, metric_reduction_after_merge_max;
  binmarg *MPM_new, *result;

  /* Initialize MPM to the univariate factorization */
  order  = randomPermutation( number_of_parameters );
  result = (binmarg *) Malloc( number_of_parameters*sizeof( binmarg ) );
  (*mpm_length) = number_of_parameters;
  for( i = 0; i < number_of_parameters; i++ )
  {
    result[i].indices           = (int *) Malloc( 1*sizeof( int ) );
    (result[i].indices)[0]      = order[i];
    result[i].number_of_indices = 1;
    estimateParametersForSingleBinaryMarginalInBinMargTypeSpecificGOMEA( gomea_index, &(result[i]) );
  }
  free( order );

  /* Initialize the metric change */
  metric_reduction_after_merge = (double **) Malloc( number_of_parameters*sizeof( double * ) );
  for( i = 0; i < number_of_parameters; i++ )
    metric_reduction_after_merge[i] = (double *) Malloc( number_of_parameters*sizeof( double ) );
  for( i = 0; i < number_of_parameters; i++ )
    for( j = i+1; j < number_of_parameters; j++ )
      metric_reduction_after_merge[i][j] = computeMetricReductionAfterMerge( gomea_index, result[i], result[j] );

  /* Find a better MPM with a greedy algorithm */
  done = 0;
  while( !done )
  {
    /* Find largest change in metric, go over candidates in random order */
    order = randomPermutation( (*mpm_length) );

    a = order[0];
    b = order[1];
    if( a > b )
    {
      c = a;
      a = b;
      b = c;
    }
    a_max = a;
    b_max = b;
    metric_reduction_after_merge_max = metric_reduction_after_merge[a_max][b_max];
    for( i = 0; i < (*mpm_length); i++ )
    {
      for( j = i+1; j < (*mpm_length); j++ )
      {
        a = order[i];
        b = order[j];
        if( a > b )
        {
          c = a;
          a = b;
          b = c;
        }

        if( metric_reduction_after_merge[a][b] > metric_reduction_after_merge[a_max][b_max] )
        {
          a_max = a;
          b_max = b;
          metric_reduction_after_merge_max = metric_reduction_after_merge[a][b];
        }
      }
    }

    free( order );

    /* Execute merge if largest change is positive */
    if( metric_reduction_after_merge_max >= 0 )
    {
      number_of_indices = result[a_max].number_of_indices+result[b_max].number_of_indices;
      indices           = (int *) Malloc( number_of_indices*sizeof( int ) );
  
      i = 0;
      for( j = 0; j < result[a_max].number_of_indices; j++ )
      {
        indices[i] = result[a_max].indices[j];
        i++;
      }
      for( j = 0; j < result[b_max].number_of_indices; j++ )
      {
        indices[i] = result[b_max].indices[j];
        i++;
      }

      free( result[a_max].indices );
      free( result[a_max].cumulative_probabilities );
      free( result[b_max].indices );
      free( result[b_max].cumulative_probabilities );

      MPM_new = (binmarg *) Malloc( ((*mpm_length)-1)*sizeof( binmarg ) );
      for( i = 0; i < (*mpm_length)-1; i++ )
        MPM_new[i] = result[i];

      MPM_new[a_max].indices           = indices;
      MPM_new[a_max].number_of_indices = number_of_indices;
      estimateParametersForSingleBinaryMarginalInBinMargTypeSpecificGOMEA( gomea_index, &(MPM_new[a_max]) );

      if( b_max < (*mpm_length)-1 )
      {
        MPM_new[b_max] = result[(*mpm_length)-1];

        for( i = 0; i < b_max; i++ )
          metric_reduction_after_merge[i][b_max] = metric_reduction_after_merge[i][(*mpm_length)-1];
 
        for( j = b_max+1; j < (*mpm_length)-1; j++ )
          metric_reduction_after_merge[b_max][j] = metric_reduction_after_merge[j][(*mpm_length)-1];
      }

      for( i = 0; i < a_max; i++ )
        metric_reduction_after_merge[i][a_max] = computeMetricReductionAfterMerge( gomea_index, MPM_new[i], MPM_new[a_max] );

      for( j = a_max+1; j < (*mpm_length)-1; j++ )
        metric_reduction_after_merge[a_max][j] = computeMetricReductionAfterMerge( gomea_index, MPM_new[a_max], MPM_new[j] );

      free( result );
      result = MPM_new;
      (*mpm_length)--;
    }
    else
      done = 1;
  }

  for( i = 0; i < number_of_parameters; i++ )
    free( metric_reduction_after_merge[i] );
  free( metric_reduction_after_merge );

  return( result );
}

void estimateParametersForSingleBinaryMarginalInBinMargTypeSpecificGOMEA( int gomea_index, binmarg *binary_marginal )
{
  int i, j, index, power_of_two;

  binary_marginal->number_of_cumulative_probabilities = (int) pow( 2, binary_marginal->number_of_indices );
  binary_marginal->cumulative_probabilities           = (double *) Malloc( binary_marginal->number_of_cumulative_probabilities*sizeof( double ) );

  for( i = 0; i < binary_marginal->number_of_cumulative_probabilities; i++ )
    (binary_marginal->cumulative_probabilities)[i] = 0.0;

  for( i = 0; i < population_sizes[gomea_index]; i++ )
  {
    index        = 0;
    power_of_two = 1;
    for( j = binary_marginal->number_of_indices-1; j >= 0; j-- )
    {
      index += populations[gomea_index][i][(binary_marginal->indices)[j]] ? power_of_two : 0;
      power_of_two *= 2;
    }

    (binary_marginal->cumulative_probabilities)[index] += 1.0;
  }

  for( i = 0; i < binary_marginal->number_of_cumulative_probabilities; i++ )
    (binary_marginal->cumulative_probabilities)[i] /= (double) population_sizes[gomea_index];

  for( i = 1; i < binary_marginal->number_of_cumulative_probabilities; i++ )
    (binary_marginal->cumulative_probabilities)[i] += (binary_marginal->cumulative_probabilities)[i-1];

  (binary_marginal->cumulative_probabilities)[binary_marginal->number_of_cumulative_probabilities-1] = 1.0;
}

double computeMetricReductionAfterMerge( int gomea_index, binmarg binary_marginal_0, binmarg binary_marginal_1 )
{
  int     i, j;
  double  result, p, cpcx, cpcy, cpcxy, mcx, mcy, mcxy;
  binmarg binary_marginal_2;

  binary_marginal_2.number_of_indices = binary_marginal_0.number_of_indices+binary_marginal_1.number_of_indices;
  binary_marginal_2.indices           = (int *) Malloc( (binary_marginal_2.number_of_indices)*sizeof( int ) );

  i = 0;
  for( j = 0; j < binary_marginal_0.number_of_indices; j++ )
  {
    binary_marginal_2.indices[i] = binary_marginal_0.indices[j];
    i++;
  }
  for( j = 0; j < binary_marginal_1.number_of_indices; j++ )
  {
    binary_marginal_2.indices[i] = binary_marginal_1.indices[j];
    i++;
  }

  estimateParametersForSingleBinaryMarginalInBinMargTypeSpecificGOMEA( gomea_index, &binary_marginal_2 );

  cpcx = 0.0;
  for( i = 0; i < binary_marginal_0.number_of_cumulative_probabilities; i++ )
  {
    if( i == 0 )
    p = binary_marginal_0.cumulative_probabilities[i];
    else
    p = binary_marginal_0.cumulative_probabilities[i]-binary_marginal_0.cumulative_probabilities[i-1];
    if( p > 0 )
    cpcx += -p*log2(p);
  }
  cpcx *= population_sizes[gomea_index];

  cpcy = 0.0;
  for( i = 0; i < binary_marginal_1.number_of_cumulative_probabilities; i++ )
  {
    if( i == 0 )
    p = binary_marginal_1.cumulative_probabilities[i];
    else
    p = binary_marginal_1.cumulative_probabilities[i]-binary_marginal_1.cumulative_probabilities[i-1];
    if( p > 0 )
    cpcy += -p*log2(p);
  }
  cpcy *= population_sizes[gomea_index];

  cpcxy = 0.0;
  for( i = 0; i < binary_marginal_2.number_of_cumulative_probabilities; i++ )
  {
    if( i == 0 )
    p = binary_marginal_2.cumulative_probabilities[i];
    else
    p = binary_marginal_2.cumulative_probabilities[i]-binary_marginal_2.cumulative_probabilities[i-1];
    if( p > 0 )
    cpcxy += -p*log2(p);
  }
  cpcxy *= population_sizes[gomea_index];

  mcx  = log2(population_sizes[gomea_index]+1)*(binary_marginal_0.number_of_cumulative_probabilities-1);
  mcy  = log2(population_sizes[gomea_index]+1)*(binary_marginal_1.number_of_cumulative_probabilities-1);
  mcxy = log2(population_sizes[gomea_index]+1)*(binary_marginal_2.number_of_cumulative_probabilities-1);

  result = mcx + mcy - mcxy + cpcx + cpcy - cpcxy;

  free( binary_marginal_2.indices );
  free( binary_marginal_2.cumulative_probabilities );

  return( result );
}

void printFOSContentsSpecificGOMEA( int gomea_index )
{
  int i, j;

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    printf( "# [" );
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
    {
      printf( "%d",FOSs[gomea_index][i][j] );
      if( j < FOSs_number_of_indices[gomea_index][i]-1 )
        printf( " " );
    }
    printf( "]\n" );
  }
  fflush( stdout );
}


/**
 * Computes the two-log of x.
 */
double math_log_two = log(2.0);
double log2( double x )
{
  return( log(x) / math_log_two );
}

/**
 * Generates new solutions.
 */
void generateAndEvaluateNewSolutionsToFillOffspringSpecificGOMEA( int gomea_index )
{
  char  *solution;
  int    i, j;
  double obj, con;

  for( i = 0; i < population_sizes[gomea_index]; i++ )
  {
    solution = generateAndEvaluateNewSolutionSpecificGOMEA( gomea_index, i%(population_sizes[gomea_index]), i, &obj, &con );

    for( j = 0; j < number_of_parameters; j++ )
      offsprings[gomea_index][i][j] = solution[j];

    objective_values_offsprings[gomea_index][i] = obj;
    constraint_values_offsprings[gomea_index][i] = con;

    free( solution );
  }
}

void printSolution(char *parameters)
{
  for (int i = 0; i < number_of_parameters; ++i)
    printf("%d:%d ", i, parameters[i]);
  printf("\n");
}
/**
 * Performs Genepool Optimal Mixing (for one solution in the population).
 */
char *generateAndEvaluateNewSolutionSpecificGOMEA( int gomea_index, int parent_index, int offspring_index, double *obj, double *con )
{
  char   *result, *backup, *backup_for_ilse, donor_parameters_are_the_same;
  short   solution_has_changed;
  int     i, j, j_ilse_best, index, parameter_index_for_ilse[1];
  double  obj_backup=0, con_backup=0, obj_backup_for_ilse, con_backup_for_ilse, obj_ilse_best, con_ilse_best;
  bool is_surrogate_used;

  if (mixed_populations_mode)
  {
    real_solutions_part = ceil(population_sizes[gomea_index] * (0.9 - model_quality));
    if (real_solutions_part >= population_sizes[gomea_index] / 2 )
      real_solutions_part = population_sizes[gomea_index] / 2;

    if (verbose)
      printf ("real_solutions_part %d ", real_solutions_part);
  }

  solution_has_changed = 0;

  result = (char *) Malloc( number_of_parameters*sizeof( char ) );
  for( i = 0; i < number_of_parameters; i++ )
    result[i] = populations[gomea_index][parent_index][i];
  
  if (mixed_populations_mode && parent_index < real_solutions_part)
    expensiveProblemEvaluation( problem_index, gomea_index, result, &objective_values[gomea_index][parent_index], &constraint_values[gomea_index][parent_index], FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 0, &is_surrogate_used );
  else
    expensiveProblemEvaluation( problem_index, gomea_index, result, &objective_values[gomea_index][parent_index], &constraint_values[gomea_index][parent_index], FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 2, &is_surrogate_used );
       
  *obj = objective_values[gomea_index][parent_index];
  *con = constraint_values[gomea_index][parent_index];

  if (verbose)
    printf("FITNESS BEFORE CHANGES %lf\n", *obj);

  backup = (char *) Malloc( number_of_parameters*sizeof( char ) );
  for( i = 0; i < number_of_parameters; i++ )
    backup[i] = result[i];

  backup_for_ilse = NULL; /* Only needed to prevent compiler warnings. */
  if( use_ilse )
  {
    backup_for_ilse = (char *) Malloc( number_of_parameters*sizeof( char ) );
    for( i = 0; i < number_of_parameters; i++ )
      backup_for_ilse[i] = result[i];
  }

  obj_backup = *obj;
  con_backup = *con;

  /* Phase 1: optimal mixing with random donors */
  shuffleFOSSpecificGOMEA( gomea_index );
  if( use_ilse )
    shuffleFOSSubsetsSpecificGOMEA( gomea_index );
  
  is_surrogate_used = false;
  bool real_already_used = false;

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    if( FOSs_number_of_indices[gomea_index][i] == number_of_parameters )
      continue;

    if (!mixed_populations_mode || offspring_index >= real_solutions_part)
      index = randomInt(population_sizes[gomea_index]);
    else
      index = randomInt(real_solutions_part);
      
    /* Convert index to binary representation and set factor variables. */
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
    {
      result[FOSs[gomea_index][i][j]] = populations[gomea_index][index][FOSs[gomea_index][i][j]];
    }

    /* Test if the change is for the better */
    donor_parameters_are_the_same = 1;
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
    {
      if( backup[FOSs[gomea_index][i][j]] != result[FOSs[gomea_index][i][j]] )
      {
        donor_parameters_are_the_same = 0;
        break;
      }
    }

    if( !donor_parameters_are_the_same )
    {
      if( !use_ilse )
      {
        if (!mixed_populations_mode)
        {
          if (!real_already_used)
            expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 1, &is_surrogate_used );
          else
            expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 2, &is_surrogate_used );
          
          if (!is_surrogate_used)
            real_already_used = true;
        }
        else
        {
          if (offspring_index < real_solutions_part)
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 0, &is_surrogate_used );
          else
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 1, &is_surrogate_used );
          
        }
      }
      else
      {
        for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
          result[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];

        j_ilse_best         = 0;
        obj_ilse_best       = *obj;
        con_ilse_best       = *con;
        obj_backup_for_ilse = *obj;
        con_backup_for_ilse = *con;
        for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
        {
          if( result[FOSs[gomea_index][i][j]] != populations[gomea_index][index][FOSs[gomea_index][i][j]] )
          {
            result[FOSs[gomea_index][i][j]] = populations[gomea_index][index][FOSs[gomea_index][i][j]];
            parameter_index_for_ilse[0] = FOSs[gomea_index][i][j];
            expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, 1, parameter_index_for_ilse, backup_for_ilse, obj_backup_for_ilse, con_backup_for_ilse, 1, &is_surrogate_used );
          }
          if( (j == 0) || betterFitness( *obj, *con, obj_ilse_best, con_ilse_best ) || equalFitness( *obj, *con, obj_ilse_best, con_ilse_best ) )
          {
            j_ilse_best   = j;
            obj_ilse_best = *obj;
            con_ilse_best = *con;
          }
          backup_for_ilse[FOSs[gomea_index][i][j]] = populations[gomea_index][index][FOSs[gomea_index][i][j]];
          obj_backup_for_ilse = *obj;
          con_backup_for_ilse = *con;
        }
        for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
        {
          result[FOSs[gomea_index][i][j]]          = backup[FOSs[gomea_index][i][j]];
          backup_for_ilse[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];
        }
        for( j = 0; j <= j_ilse_best; j++ )
          result[FOSs[gomea_index][i][j]] = populations[gomea_index][index][FOSs[gomea_index][i][j]];
        *obj = obj_ilse_best;
        *con = con_ilse_best;
      }

      bool acceptChange;
      acceptChange = betterFitness( *obj, *con, obj_backup, con_backup ) || equalFitness( *obj, *con, obj_backup, con_backup );

      if ( acceptChange )
      {
        for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
          backup[FOSs[gomea_index][i][j]] = result[FOSs[gomea_index][i][j]];
        if( use_ilse )
        {
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
            backup_for_ilse[FOSs[gomea_index][i][j]] = result[FOSs[gomea_index][i][j]];
        }

        obj_backup = *obj;
        con_backup = *con;

        solution_has_changed = 1;
      }
      else
      {
        for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
          result[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];

        *obj = obj_backup;
        *con = con_backup;
      }
    }
  }

  real_already_used = false;
  
  /* Phase 2 (Forced Improvement): optimal mixing with elitist solution */
  if( (!solution_has_changed) || (no_improvement_stretchs[gomea_index] > (1+(log(population_sizes[gomea_index])/log(10)))) )
  {
    shuffleFOSSpecificGOMEA( gomea_index );
    if( use_ilse )
      shuffleFOSSubsetsSpecificGOMEA( gomea_index );

    solution_has_changed = 0;
    for( i = 0; i < FOSs_length[gomea_index]; i++ )
    {
      for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      {
        result[FOSs[gomea_index][i][j]] = elitist_solution[FOSs[gomea_index][i][j]];
      }

      /* Test if the change is for the better */
      donor_parameters_are_the_same = 1;
      for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      {
        if( backup[FOSs[gomea_index][i][j]] != result[FOSs[gomea_index][i][j]] )
        {
          donor_parameters_are_the_same = 0;
          break;
        }
      }
      
      if( !donor_parameters_are_the_same )
      {
        if( !use_ilse )
        {
          if (!mixed_populations_mode)
          {
            if (!real_already_used)
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 1, &is_surrogate_used  );
            else
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 2, &is_surrogate_used  );
          
            if (!is_surrogate_used)
              real_already_used = true;
          }
          else
          {
            if (offspring_index < real_solutions_part)
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 0, &is_surrogate_used  );
            else
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, FOSs_number_of_indices[gomea_index][i], FOSs[gomea_index][i], backup, obj_backup, con_backup, 1, &is_surrogate_used  );

          }
        }

        else
        {
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
            result[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];

          j_ilse_best         = 0;
          obj_ilse_best       = *obj;
          con_ilse_best       = *con;
          obj_backup_for_ilse = *obj;
          con_backup_for_ilse = *con;
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
          {
            if( result[FOSs[gomea_index][i][j]] != elitist_solution[FOSs[gomea_index][i][j]] )
            {
              result[FOSs[gomea_index][i][j]] = elitist_solution[FOSs[gomea_index][i][j]];
              parameter_index_for_ilse[0] = FOSs[gomea_index][i][j];
              expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, 1, parameter_index_for_ilse, backup_for_ilse, obj_backup_for_ilse, con_backup_for_ilse, 1, &is_surrogate_used  );
            }
            if( (j == 0) || betterFitness( *obj, *con, obj_ilse_best, con_ilse_best ) || equalFitness( *obj, *con, obj_ilse_best, con_ilse_best ) )
            {
              j_ilse_best   = j;
              obj_ilse_best = *obj;
              con_ilse_best = *con;
            }
            backup_for_ilse[FOSs[gomea_index][i][j]] = elitist_solution[FOSs[gomea_index][i][j]];
            obj_backup_for_ilse = *obj;
            con_backup_for_ilse = *con;
          }
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
          {
            result[FOSs[gomea_index][i][j]]          = backup[FOSs[gomea_index][i][j]];
            backup_for_ilse[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];
          }
          for( j = 0; j <= j_ilse_best; j++ )
            result[FOSs[gomea_index][i][j]] = elitist_solution[FOSs[gomea_index][i][j]];
          *obj = obj_ilse_best;
          *con = con_ilse_best;
        }

        char acceptChange;
        acceptChange = betterFitness( *obj, *con, obj_backup, con_backup );
        if (verbose)  
          printf("FORCED IMPROVEMENT %lf %lf\n", *obj, obj_backup);

        if( acceptChange )
        {
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
            backup[FOSs[gomea_index][i][j]] = result[FOSs[gomea_index][i][j]];
          if( use_ilse )
          {
            for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
              backup_for_ilse[FOSs[gomea_index][i][j]] = result[FOSs[gomea_index][i][j]];
          }

          obj_backup = *obj;
          con_backup = *con;

          solution_has_changed = 1;
        }
        else
        {
          for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
            result[FOSs[gomea_index][i][j]] = backup[FOSs[gomea_index][i][j]];

          *obj = obj_backup;
          *con = con_backup;
        }
      }
      if( solution_has_changed )
        break;
    }

    if( !solution_has_changed )
    {
      if( betterFitness( elitist_solution_objective_value, elitist_solution_constraint_value, *obj, *con ) )
        solution_has_changed = 1;

      //in case surrogate elitist of solution larger then elitists surrogate real evaluation is performed
      expensiveProblemEvaluation( problem_index, gomea_index, backup, &obj_backup, con, 1, parameter_index_for_ilse, backup_for_ilse, obj_backup_for_ilse, con_backup_for_ilse, 0, &is_surrogate_used  );      
      
      for( i = 0; i < number_of_parameters; i++ )
        result[i] = elitist_solution[i];
      *obj = elitist_solution_objective_value;
      *con = elitist_solution_constraint_value;

      expensiveProblemEvaluation( problem_index, gomea_index, result, obj, con, 1, parameter_index_for_ilse, backup_for_ilse, obj_backup_for_ilse, con_backup_for_ilse, 2, &is_surrogate_used  );
    }
  }

  free( backup );
  if( use_ilse )
    free( backup_for_ilse );
  
  updated_solutions_count[gomea_index]++;
  if (verbose)
    printf("FITNESS AFTER CHANGES %d %d %lf %d %d\n", gomea_index, gomeaUpdatesCounter[gomea_index], *obj, updated_solutions_count[gomea_index], updated_solutions_count_when_updated_elitist[gomea_index]);
      
  return( result );
}

/**
 * Shuffles the FOS (ordering), but not the contents
 * of the linkage subsets themselves.
 */
void shuffleFOSSpecificGOMEA( int gomea_index )
{
  int i, *order, **FOSs_new, *FOSs_number_of_indices_new;

  FOSs_new                   = (int **) Malloc( FOSs_length[gomea_index]*sizeof( int * ) );
  FOSs_number_of_indices_new = (int *) Malloc( FOSs_length[gomea_index]*sizeof( int ) );
  order                     = randomPermutation( FOSs_length[gomea_index] );
  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    FOSs_new[i]                   = FOSs[gomea_index][order[i]];
    FOSs_number_of_indices_new[i] = FOSs_number_of_indices[gomea_index][order[i]];
  }
  free( FOSs[gomea_index] );
  free( FOSs_number_of_indices[gomea_index] );
  FOSs[gomea_index]                   = FOSs_new;
  FOSs_number_of_indices[gomea_index] = FOSs_number_of_indices_new;

  free( order );
}

/**
 * Shuffles the linkage subsets (ordering) in the FOS, but not the FOS itself.
 */
void shuffleFOSSubsetsSpecificGOMEA( int gomea_index )
{
  int i, j, *order, *FOSs_subset_new;

  for( i = 0; i < FOSs_length[gomea_index]; i++ )
  {
    order = randomPermutation( FOSs_number_of_indices[gomea_index][i] );

    FOSs_subset_new = (int *) Malloc( FOSs_number_of_indices[gomea_index][i]*sizeof( int ) );
    for( j = 0; j < FOSs_number_of_indices[gomea_index][i]; j++ )
      FOSs_subset_new[j] = FOSs[gomea_index][i][order[j]];
    free( FOSs[gomea_index][i] );
    FOSs[gomea_index][i] = FOSs_subset_new;

    free( order );
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=- Section Ezilaitini -=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * Undoes GOMEA initializations.
 */
void ezilaitiniAllGOMEAs()
{
  int i;

  for( i = 0; i < number_of_GOMEAs; i++ )
    ezilaitiniSpecificGOMEA( i );

  free( FOSs_length );
  free( FOSs_number_of_indices );
  free( FOSs );
  free( MI_matrices );
  free( populations );
  free( objective_values );
  free( real_objective_values);
  free( not_surrogate_objective_values ) ;
  free( constraint_values );
  free( average_objective_values );
  free( average_constraint_values );
  free( terminated );
  free( offsprings );
  free( objective_values_offsprings );
  free( constraint_values_offsprings );
  free( objective_values_best_of_generation );
  free( constraint_values_best_of_generation );
}

void ezilaitiniSpecificGOMEA( int gomea_index )
{
  int i;

  if( FOSs[gomea_index] != NULL )
  {
    for( i = 0; i < FOSs_length[gomea_index]; i++ )
      free( FOSs[gomea_index][i] );
    free( FOSs[gomea_index] );
    free( FOSs_number_of_indices[gomea_index] );
  }

  for( i = 0; i < number_of_parameters; i++ )
    free( MI_matrices[gomea_index][i] );
  free( MI_matrices[gomea_index] );

  ezilaitiniSpecificGOMEAMemoryForPopulationAndOffspring( gomea_index );
}

/**
 * Initializes the memory for a single GOMEA instance, for the population only.
 */
void ezilaitiniSpecificGOMEAMemoryForPopulationAndOffspring( int gomea_index )
{
  int i;

  for( i = 0; i < population_sizes[gomea_index]; i++ )
    free( offsprings[gomea_index][i] );

  for( i = 0; i < population_sizes[gomea_index]; i++ )
    free( populations[gomea_index][i] );

  free( populations[gomea_index] );
  free( objective_values[gomea_index] );
  free( constraint_values[gomea_index] );
  free( offsprings[gomea_index] );
  free( objective_values_offsprings[gomea_index] );
  free( constraint_values_offsprings[gomea_index] );
}

void ezilaitiniValueAndSetOfSolutionsToReach()
{
  int i;

  for( i = 0; i < number_of_solutions_in_sostr; i++ )
    free( sostr[i] );
  free( sostr );
}

void ezilaitiniProblem( int index )
{
  switch( index )
  {
    case  0: break;
    case  1: break;
    case  2: break;
    case  3: break;
    case  4: break;
    case  5: adfFunctionProblemNoitazilaitini(); break;
  }
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Time -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
long getMilliSecondsRunning()
{
  return( getMilliSecondsRunningSinceTimeStamp( timestamp_start ) );
}

long getMilliSecondsRunningAfterInit()
{
  return( getMilliSecondsRunningSinceTimeStamp( timestamp_start_after_init ) );
}

long getMilliSecondsRunningSinceTimeStamp( long timestamp )
{
  long timestamp_now, difference;

  timestamp_now = getCurrentTimeStampInMilliSeconds();

  difference = timestamp_now-timestamp;

  return( difference );
}

long getCurrentTimeStampInMilliSeconds()
{
  struct timeval tv;
  long   result;

  gettimeofday( &tv, NULL );
  result = (tv.tv_sec * 1000) + (tv.tv_usec / 1000);

  return( result );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Run -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Initializes python modules and functions, the VOSOSTR, the random number generator and the problem and runs the GOMEA.
 */
void run()
{
  timestamp_start = getCurrentTimeStampInMilliSeconds();

  sprintf(filename_elitist_solution_hitting_time, "%s/elitist_solution_hitting_time.dat", (char * )folder_name);
  sprintf(filename_elitist_solutions, "%s/elitist_solutions.dat", (char * )folder_name);
  sprintf(filename_elitist_solution, "%s/elitist_solution.dat", (char * )folder_name);
  sprintf(filename_solutions, "%s/solutions.dat", (char * )folder_name);

  char pwd[1000], module_name[1000];
  getcwd( pwd, 1000 );

  //init all Python functions
  Py_Initialize();
  sprintf(module_name, "import sys; sys.path.insert(0, \"%s/utils\")", pwd);
  
  PyRun_SimpleString (module_name);
  printf("%s\n", module_name);
    
  module = PyImport_ImportModule("cs_gomea_python_functions");
  if (module == NULL) {
    printf("ERROR importing module");
    exit(-1);
    } 
  
  PyObject *function_init = PyObject_GetAttrString(module,(char*)"function_init");
  if (function_init == NULL) {
    printf("ERROR getting function init");
    exit(-1);
    } 

  PyObject *pyArgs = Py_BuildValue("s", folder_name);
  PyObject *pyArgs2 = PyTuple_Pack(1, pyArgs);
  PyObject* pyResult = PyObject_CallObject(function_init, pyArgs2);
  if (pyResult == NULL) {
    printf("ERROR getting result from function init");
    exit(-1);
    } 
  
  PyObject *function_set_gpu = PyObject_GetAttrString(module,(char*)"set_gpu_device");
  if (function_set_gpu == NULL) {
    printf("ERROR getting function set gpu");
    exit(-1);
    } 

  pyArgs = Py_BuildValue("i", gpu_device);
  PyObject *pyArgs_time = Py_BuildValue("i", maximum_number_of_milliseconds / 1000);
  
  pyArgs2 = PyTuple_Pack(2, pyArgs, pyArgs_time);
  pyResult = PyObject_CallObject(function_set_gpu, pyArgs2);
  if (pyResult == NULL) {
    printf("ERROR getting result from function set gpu");
    exit(-1);
    } 

  function_evaluate = PyObject_GetAttrString(module,(char*)"function_get_fitness_pairwise");    
  if (function_evaluate == NULL) {
    printf("ERROR getting function evaluate");
    exit(-1);
    } 

  function_save = PyObject_GetAttrString(module,(char*)"function_save");
  if (function_save == NULL) {
    printf("ERROR getting function save");
    exit(-1);
    } 
  
  function_train_model = PyObject_GetAttrString(module,(char*)"train_model_pairwise_regression");
  
  if (function_train_model == NULL) {
    printf("ERROR getting function function_train_model");
    exit(-1);
    } 

  function_reset_file = PyObject_GetAttrString(module,(char*)"function_reset_file");
  if (function_reset_file == NULL) {
    printf("ERROR getting function reset file");
    exit(-1);
    } 

  pyArgs = NULL;
  pyResult = PyObject_CallObject(function_reset_file, pyArgs);
  if (pyResult == NULL) {
      printf("ERROR getting result from python function RESET_FILE");
      exit(-1);
  }
  
  writeElitistEvaluationsInit(filename_elitist_solutions);

  initializeValueAndSetOfSolutionsToReach();

  initializeRandomNumberGenerator();

  initializeProblem( problem_index );

  if( print_verbose_overview )
    printVerboseOverview();

  timestamp_start_after_init = getCurrentTimeStampInMilliSeconds();

  multiPopGOMEA();

  writeRunningTime( (char *) "total_running_time.dat" );

  ezilaitiniProblem( problem_index );

  ezilaitiniValueAndSetOfSolutionsToReach();

  Py_Finalize();
}

void multiPopGOMEA()
{
  maximum_number_of_GOMEAs                  = 20;
  number_of_subgenerations_per_GOMEA_factor = 4;
  base_population_size                      = 8;

  number_of_GOMEAs               = 0;
  number_of_generations          = 0;
  number_of_evaluations          = 0;
  previous_training_evaluation   = -1;
  number_of_bit_flip_evaluations = 0;
  minimum_GOMEA_index            = 0;

  population_sizes        = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
  no_improvement_stretchs = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
  elitist_solution        = (char *) Malloc( number_of_parameters*sizeof( char ) );

  evaluated_solutions                  = (char **) Malloc( max_evaluated_solutions*sizeof( char * ) );
  evaluated_archive                    = (double *) Malloc( max_evaluated_solutions*sizeof( double ) );
  evaluated_random_archive             = (char **) Malloc( max_evaluated_solutions*sizeof( char * ) );
  evaluated_random_archive_values      = (double *) Malloc( max_evaluated_solutions*sizeof( double ) );

  surrogate_evaluated_solutions                  = (char **) Malloc( max_evaluated_solutions*sizeof( char * ) );
  surrogate_evaluated_archive                    = (double *) Malloc( max_evaluated_solutions*sizeof( double ) );
  sorted_evaluated_archive                    = (double *) Malloc( max_evaluated_solutions*sizeof( double ) );

  surrogate_evaluations_when_updated_elitist = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
  updated_solutions_count = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
  updated_solutions_count_when_updated_elitist = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );

  gomeaUpdatesCounter = (int *) Malloc( maximum_number_of_GOMEAs*sizeof( int ) );
  for (int i = 0; i < maximum_number_of_GOMEAs; ++i)
  {
    gomeaUpdatesCounter[i] = 0;
    updated_solutions_count[i] = 0;
    updated_solutions_count_when_updated_elitist[i] = 0;
    surrogate_evaluations_when_updated_elitist[i] = 0;
  }

  for( int i = 0; i < max_evaluated_solutions; i++ )
    evaluated_solutions[i] = (char *) Malloc( number_of_parameters*sizeof( char ) );
  for( int i = 0; i < max_evaluated_solutions; i++ )
    evaluated_random_archive[i] = (char *) Malloc( number_of_parameters*sizeof( char ) );
  for( int i = 0; i < max_evaluated_solutions; i++ )
    surrogate_evaluated_solutions[i] = (char *) Malloc( number_of_parameters*sizeof( char ) );

  
  while( !checkTermination() )
  {
    if( number_of_GOMEAs < maximum_number_of_GOMEAs )
      initializeNewGOMEA();

    if( write_generational_statistics )
      writeGenerationalStatistics();

    if( write_generational_solutions )
      writeGenerationalSolutions( 0 );

    if (number_of_GOMEAs == 0)
      makeRealEvaluations();

    generationalStepAllGOMEAs();

    number_of_generations++;
  }

  if( write_generational_statistics )
    writeGenerationalStatistics();

  if( write_generational_solutions )
    writeGenerationalSolutions( 1 );

  ezilaitiniAllGOMEAs();

  free( elitist_solution );
  free( no_improvement_stretchs );
  free( population_sizes );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/





/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=- Section Main -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/**
 * The main function:
 * - interpret parameters on the command line
 * - run the algorithm with the interpreted parameters
 */
int main( int argc, char **argv )
{
  interpretCommandLine( argc, argv );

  run();

  return( 0 );
}
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
