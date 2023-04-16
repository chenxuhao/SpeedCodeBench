#include <stdio.h>
#include <stdlib.h>
#include <sys/times.h>
#include <unistd.h>
#include <sys/stat.h>
#include "config.h"
#include "lbm_1d_array.h"

#define N_DISTR_FUNCS FLAGS

typedef struct {
  float timeScale;
  clock_t tickStart, tickStop;
  struct tms timeStart, timeStop;
} MAIN_Time;

typedef enum {NOTHING = 0, COMPARE, STORE} MAIN_Action;
typedef enum {LDC = 0, CHANNEL} MAIN_SimType;

typedef struct {
  int nTimeSteps;
  char* resultFilename;
  MAIN_Action action;
  MAIN_SimType simType;
  char* obstacleFilename;
} MAIN_Param;

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param);
void MAIN_printInfo( const MAIN_Param* param );
void MAIN_initialize( const MAIN_Param* param );
void MAIN_finalize( const MAIN_Param* param );
void MAIN_startClock( MAIN_Time* time );
void MAIN_stopClock( MAIN_Time* time, const MAIN_Param* param );

static LBM_GridPtr srcGrid, dstGrid;

void LBM_allocateGrid( float** ptr );
void LBM_freeGrid( float** ptr );
void LBM_initializeGrid( LBM_Grid grid );
void LBM_initializeSpecialCellsForLDC( LBM_Grid grid );
void LBM_loadObstacleFile( LBM_Grid grid, const char* filename );
void LBM_initializeSpecialCellsForChannel( LBM_Grid grid );
void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 );
void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid );
void LBM_handleInOutFlow( LBM_Grid srcGrid );
void LBM_showGridStatistics( LBM_Grid Grid );
void LBM_storeVelocityField( LBM_Grid grid, const char* filename, const BOOL binary );
void LBM_compareVelocityField( LBM_Grid grid, const char* filename, const BOOL binary );
#ifdef USE_GPU
void CUDA_LBM_allocateGrid( float** ptr );
void CUDA_LBM_freeGrid( float** ptr );
void CUDA_LBM_initializeGrid( float** d_grid, float** h_grid );
void CUDA_LBM_getDeviceGrid( float** d_grid, float** h_grid );
#endif

int main( int nArgs, char* arg[] ) {
  MAIN_Param param;
  MAIN_Time time;
  int t;
  MAIN_parseCommandLine( nArgs, arg, &param);
  MAIN_printInfo( &param );
  MAIN_initialize( &param );
  MAIN_startClock( &time );
  for( t = 1; t <= param.nTimeSteps; t++ ) {
    if( param.simType == CHANNEL ) {
      LBM_handleInOutFlow( *srcGrid );
    }
    LBM_performStreamCollide( *srcGrid, *dstGrid );
    LBM_swapGrids( &srcGrid, &dstGrid );
    if( (t & 63) == 0 ) {
      printf( "timestep: %i\n", t );
      //LBM_showGridStatistics( *srcGrid );
    }
  }
  MAIN_stopClock( &time, &param );
  MAIN_finalize( &param );
  return 0;
}

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param) {
  struct stat fileStat;
  if ( nArgs < 2 ) {
    printf( "syntax: lbm <time steps>\n" );
    exit( 1 );
  }
  param->nTimeSteps = atoi( arg[1] );
  param->resultFilename = arg[2];
  if ( arg[3] != NULL ) {
    param->obstacleFilename = arg[3];
    if( stat( param->obstacleFilename, &fileStat ) != 0 ) {
      printf( "MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
          param->obstacleFilename );
      exit( 1 );
    }
    if( fileStat.st_size != SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z ) {
      printf( "MAIN_parseCommandLine:\n"
          "\tsize of file '%s' is %i bytes\n"
          "\texpected size is %i bytes\n",
          param->obstacleFilename, (int) fileStat.st_size,
          SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z );
      exit( 1 );
    }
  } else param->obstacleFilename = NULL;
  param->action         = STORE;
  param->simType        = LDC;
}

void MAIN_printInfo( const MAIN_Param* param ) {
  const char actionString[3][32] = {"nothing", "compare", "store"};
  const char simTypeString[3][32] = {"lid-driven cavity", "channel flow"};
  printf( "MAIN_printInfo:\n"
      "\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
      "\tnTimeSteps     : %i\n"
      "\tresult file    : %s\n"
      "\taction         : %s\n"
      "\tsimulation type: %s\n"
      "\tobstacle file  : %s\n\n",
      SIZE_X, SIZE_Y, SIZE_Z, 1e-6*SIZE_X*SIZE_Y*SIZE_Z,
      param->nTimeSteps, param->resultFilename, 
      actionString[param->action], simTypeString[param->simType],
      (param->obstacleFilename == NULL) ? "<none>" :
      param->obstacleFilename );
}

void MAIN_initialize( const MAIN_Param* param ) {
  LBM_allocateGrid( (float**) &srcGrid );
  LBM_allocateGrid( (float**) &dstGrid );
  LBM_initializeGrid( *srcGrid );
  LBM_initializeGrid( *dstGrid );
  if( param->obstacleFilename != NULL ) {
    LBM_loadObstacleFile( *srcGrid, param->obstacleFilename );
    LBM_loadObstacleFile( *dstGrid, param->obstacleFilename );
  }
  if( param->simType == CHANNEL ) {
    LBM_initializeSpecialCellsForChannel( *srcGrid );
    LBM_initializeSpecialCellsForChannel( *dstGrid );
  }
  else {
    LBM_initializeSpecialCellsForLDC( *srcGrid );
    LBM_initializeSpecialCellsForLDC( *dstGrid );
  }
#ifdef USE_GPU
  //Setup DEVICE datastructures
  CUDA_LBM_allocateGrid( (float**) &CUDA_srcGrid );
  CUDA_LBM_allocateGrid( (float**) &CUDA_dstGrid );
  CUDA_LBM_initializeGrid( (float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid );
  CUDA_LBM_initializeGrid( (float**)&CUDA_dstGrid, (float**)&TEMP_dstGrid );
#endif
  LBM_showGridStatistics( *srcGrid );
}

void MAIN_finalize( const MAIN_Param* param ) {
  LBM_showGridStatistics( *srcGrid );
  if( param->action == COMPARE )
    LBM_compareVelocityField( *srcGrid, param->resultFilename, TRUE );
  if( param->action == STORE )
    LBM_storeVelocityField( *srcGrid, param->resultFilename, TRUE );
  LBM_freeGrid( (float**) &srcGrid );
  LBM_freeGrid( (float**) &dstGrid );
}

void MAIN_startClock( MAIN_Time* time ) {
  time->timeScale = 1.0 / sysconf( _SC_CLK_TCK );
  time->tickStart = times( &(time->timeStart) );
}

void MAIN_stopClock( MAIN_Time* time, const MAIN_Param* param ) {
  time->tickStop = times( &(time->timeStop) );
  printf( "MAIN_stopClock:\n"
      "\tusr: %7.2f sys: %7.2f tot: %7.2f wct: %7.2f MLUPS: %5.2f\n\n",
      (time->timeStop.tms_utime - time->timeStart.tms_utime) * time->timeScale,
      (time->timeStop.tms_stime - time->timeStart.tms_stime) * time->timeScale,
      (time->timeStop.tms_utime - time->timeStart.tms_utime +
       time->timeStop.tms_stime - time->timeStart.tms_stime) * time->timeScale,
      (time->tickStop           - time->tickStart          ) * time->timeScale,
      1.0e-6 * SIZE_X * SIZE_Y * SIZE_Z * param->nTimeSteps /
      (time->tickStop           - time->tickStart          ) / time->timeScale );
}
