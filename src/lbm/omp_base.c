#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "lbm_1d_array.h"

#define DFL1 (1.0/ 3.0)
#define DFL2 (1.0/18.0)
#define DFL3 (1.0/36.0)

void LBM_allocateGrid( float** ptr ) {
  const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES,
        size   = sizeof( LBM_Grid ) + 2*margin*sizeof( float );
  *ptr = malloc( size );
  if( ! *ptr ) {
    printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
        size / (1024.0*1024.0) );
    exit( 1 );
  }
#if !defined(SPEC_CPU)
  printf( "LBM_allocateGrid: allocated %.1f MByte\n",
      size / (1024.0*1024.0) );
#endif
  *ptr += margin;
}

void LBM_freeGrid( float** ptr ) {
  const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;
  free( *ptr-margin );
  *ptr = NULL;
}

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR
	/*voption indep*/
    #pragma omp parallel for
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = DFL1;
		LOCAL( grid, N  ) = DFL2;
		LOCAL( grid, S  ) = DFL2;
		LOCAL( grid, E  ) = DFL2;
		LOCAL( grid, W  ) = DFL2;
		LOCAL( grid, T  ) = DFL2;
		LOCAL( grid, B  ) = DFL2;
		LOCAL( grid, NE ) = DFL3;
		LOCAL( grid, NW ) = DFL3;
		LOCAL( grid, SE ) = DFL3;
		LOCAL( grid, SW ) = DFL3;
		LOCAL( grid, NT ) = DFL3;
		LOCAL( grid, NB ) = DFL3;
		LOCAL( grid, ST ) = DFL3;
		LOCAL( grid, SB ) = DFL3;
		LOCAL( grid, ET ) = DFL3;
		LOCAL( grid, EB ) = DFL3;
		LOCAL( grid, WT ) = DFL3;
		LOCAL( grid, WB ) = DFL3;

		CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 ) {
  LBM_GridPtr aux = *grid1;
  *grid1 = *grid2;
  *grid2 = aux;
}

/*############################################################################*/

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
  int x,  y,  z;
  FILE* file = fopen( filename, "rb" );
  for( z = 0; z < SIZE_Z; z++ ) {
    for( y = 0; y < SIZE_Y; y++ ) {
      for( x = 0; x < SIZE_X; x++ ) {
        if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
      }
      fgetc( file );
    }
    fgetc( file );
  }

  fclose( file );
}

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
  int x,  y,  z;
  /*voption indep*/
  #pragma omp parallel for private( x, y )
  for( z = -2; z < SIZE_Z+2; z++ ) {
    for( y = 0; y < SIZE_Y; y++ ) {
      for( x = 0; x < SIZE_X; x++ ) {
        if( x == 0 || x == SIZE_X-1 ||
            y == 0 || y == SIZE_Y-1 ||
            z == 0 || z == SIZE_Z-1 ) {
          SET_FLAG( grid, x, y, z, OBSTACLE );
        }
        else {
          if( (z == 1 || z == SIZE_Z-2) &&
              x > 1 && x < SIZE_X-2 &&
              y > 1 && y < SIZE_Y-2 ) {
            SET_FLAG( grid, x, y, z, ACCEL );
          }
        }
      }
    }
  }
}

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
  int x,  y,  z;
  /*voption indep*/
  #pragma omp parallel for private( x, y )
  for( z = -2; z < SIZE_Z+2; z++ ) {
    for( y = 0; y < SIZE_Y; y++ ) {
      for( x = 0; x < SIZE_X; x++ ) {
        if( x == 0 || x == SIZE_X-1 ||
            y == 0 || y == SIZE_Y-1 ) {
          SET_FLAG( grid, x, y, z, OBSTACLE );

          if( (z == 0 || z == SIZE_Z-1) &&
              ! TEST_FLAG( grid, x, y, z, OBSTACLE ))
            SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
        }
      }
    }
  }
}

void LBM_performStreamCollide( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR
	float ux, uy, uz, u2, rho;
	/*voption indep*/
    #pragma omp parallel for private( ux, uy, uz, u2, rho )
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			DST_C ( dstGrid ) = SRC_C ( srcGrid );
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			continue;
		}

		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		      + SRC_WB( srcGrid );

		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005f;
			uy = 0.002f;
			uz = 0.000f;
		}

		u2 = 1.5f * (ux*ux + uy*uy + uz*uz);
		DST_C ( dstGrid ) = (1.0f-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0f                                 - u2);

		DST_N ( dstGrid ) = (1.0f-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uy*(4.5f*uy       + 3.0f) - u2);
		DST_S ( dstGrid ) = (1.0f-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uy*(4.5f*uy       - 3.0f) - u2);
		DST_E ( dstGrid ) = (1.0f-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       ux*(4.5f*ux       + 3.0f) - u2);
		DST_W ( dstGrid ) = (1.0f-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       ux*(4.5f*ux       - 3.0f) - u2);
		DST_T ( dstGrid ) = (1.0f-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uz*(4.5f*uz       + 3.0f) - u2);
		DST_B ( dstGrid ) = (1.0f-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0f +       uz*(4.5f*uz       - 3.0f) - u2);

		DST_NE( dstGrid ) = (1.0f-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux+uy)*(4.5f*(+ux+uy) + 3.0f) - u2);
		DST_NW( dstGrid ) = (1.0f-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux+uy)*(4.5f*(-ux+uy) + 3.0f) - u2);
		DST_SE( dstGrid ) = (1.0f-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux-uy)*(4.5f*(+ux-uy) + 3.0f) - u2);
		DST_SW( dstGrid ) = (1.0f-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux-uy)*(4.5f*(-ux-uy) + 3.0f) - u2);
		DST_NT( dstGrid ) = (1.0f-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+uy+uz)*(4.5f*(+uy+uz) + 3.0f) - u2);
		DST_NB( dstGrid ) = (1.0f-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+uy-uz)*(4.5f*(+uy-uz) + 3.0f) - u2);
		DST_ST( dstGrid ) = (1.0f-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-uy+uz)*(4.5f*(-uy+uz) + 3.0f) - u2);
		DST_SB( dstGrid ) = (1.0f-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-uy-uz)*(4.5f*(-uy-uz) + 3.0f) - u2);
		DST_ET( dstGrid ) = (1.0f-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux+uz)*(4.5f*(+ux+uz) + 3.0f) - u2);
		DST_EB( dstGrid ) = (1.0f-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (+ux-uz)*(4.5f*(+ux-uz) + 3.0f) - u2);
		DST_WT( dstGrid ) = (1.0f-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux+uz)*(4.5f*(-ux+uz) + 3.0f) - u2);
		DST_WB( dstGrid ) = (1.0f-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0f + (-ux-uz)*(4.5f*(-ux-uz) + 3.0f) - u2);
	SWEEP_END
}

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	float ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/
    #pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, ux2, uy2, uz2, rho2, u2, px, py )
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );
		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0*rho1 - rho2;

		px = (SWEEP_X / (0.5*(SIZE_X-1))) - 1.0;
		py = (SWEEP_Y / (0.5*(SIZE_Y-1))) - 1.0;
		ux = 0.00;
		uy = 0.00;
		uz = 0.01 * (1.0-px*px) * (1.0-py*py);

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END

	/* outflow */
	/*voption indep*/
    #pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, ux2, uy2, uz2, rho2, u2, px, py )
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
		uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
		uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0;

		ux = 2*ux1 - ux2;
		uy = 2*uy1 - uy2;
		uz = 2*uz1 - uz2;

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END
}

