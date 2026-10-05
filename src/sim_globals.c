#include <stdlib.h>

#include "sim_globals.h"

int g_nx, g_ny, g_nz;
float g_lx, g_ly, g_lz;
float g_dx, g_dy, g_dz;
int g_max_cell;

float g_shift_x, g_shift_y, g_shift_z;
float g_sub_z;
float g_cell_lo_x, g_cell_hi_x;

int g_nbx, g_nby, g_nbz, g_nbins;
float g_bin_ox, g_bin_oy, g_bin_oz;

float g_chem_ox, g_chem_oy, g_chem_oz;
float g_chem_lx, g_chem_ly, g_chem_lz;

int g_rank, g_nprocs;
float g_x_lo, g_x_hi;
int g_x_neighbour_left;
int g_x_neighbour_right;
int g_xcut[REBAL_MAX_PROCS + 1];
int g_x_imin, g_x_imax;
int g_min_w;

float ***chem1, ***chem2, ***chem_diff;
int ***grid, ***grid1, ***grid2, ***grid3, ***grid4;

double g_output_io = 0.0, g_output_comm = 0.0;

float RNG(void)
{
    return (1.0 * (rand() % RAND_MAX)) / (1.0 * RAND_MAX);
}
