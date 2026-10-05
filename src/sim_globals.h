#ifndef SIM_GLOBALS_H
#define SIM_GLOBALS_H

// element slots reserved per cell: element k of cell ci lives at ci * MAX_ELE + k
#define MAX_ELE 20

// cell cycle
#define C1_MEAN 180.0 // slow-dividing doubles in 180 days
#define C1_STD 40.0 // standard deviation
#define C2_MEAN 4.0 // fast-dividing doubles in 4 days
#define C2_STD 1.0 // standard deviation
#define MAX_GENER 2
#define R_DIFF 0.01f // probability that a slow-dividing cell gives birth to a fast-dividing daughter

// domain sizing
#define g_dx_TARGET 1.0f
#define PAD 5.0f
#define CELL_GROWTH 2.0f
// room to expand into, as a fraction of the cell span on each side
#define DOMAIN_PAD_FRAC 0.5f
// the chem grid is spaced at 1.0, so it tracks the cells only and never the
// padded bin box, whose cube would run to gigabytes
#define CHEM_MARGIN 8.0f

// bin grid
#define BIN_SIZE 12.0f
// slack added around the cells when the bin grid is refitted, so growth is rare
#define BIN_FIT_MARGIN (2.0f * BIN_SIZE)
// how many times a rebuild may refit the grid before giving up
#define BIN_FIT_TRIES 4

#define BC 0
#define RATIO 1

// iteration schedule
#define GROWTH_FR 1000
#define DIVISION_FR 1000
#define OUTPUT_FR 100
#define MAX_ITERATIONS 1001
#define MAX_CHEM_ITER 1000
#define MAX_DEATH 60000

// X-split decomposition and rebalancing
#define REBAL_MAX_PROCS 64
// ghost band width in X, equal to the movement cell-center cull radius, so a
// rank sees every partner that could act on a cell it owns
#define GHOST_DEPTH_F 20.0f
#define REBALANCE_FR 100 // iterations between rebalances
#define REBALANCE_MAX_STEP 8
// proportional rule only. the GRU has its own copy, RNN_HEUR_TOL/GAIN in
// rnn/rnn_balancer.c, tied to its training
#define REBALANCE_TOL 0.02
#define REBALANCE_GAIN 0.5
// 0 = the proportional rule in main.c
// 1 = the GRU in rnn/rnn_balancer.c
#define REBAL_CONTROLLER 1
#define REBAL_RNN_WEIGHTS "rnn/rnn_weights.txt" // scanned once at startup

// globals, defined in sim_globals.c and initialized by main()

// chem grid size and spacing
extern int g_nx, g_ny, g_nz;
extern float g_lx, g_ly, g_lz;
extern float g_dx, g_dy, g_dz;
extern int g_max_cell;

// grids index from 0, so the cloud is shifted at load and shifted back on output
extern float g_shift_x, g_shift_y, g_shift_z;
// IC's own floor, carried into the movement kernel
extern float g_sub_z;
// occupied x range after the shift, excluding padding
extern float g_cell_lo_x, g_cell_hi_x;

// bin grid. lower corner of bin (0,0,0); the grid is only (L/BIN_SIZE)^3 ints,
// so it is refitted around the cells whenever they drift out of it
extern int g_nbx, g_nby, g_nbz, g_nbins;
extern float g_bin_ox, g_bin_oy, g_bin_oz;

// chem grid corner and extent, kept independent of the bin box
extern float g_chem_ox, g_chem_oy, g_chem_oz;
extern float g_chem_lx, g_chem_ly, g_chem_lz;

// MPI X-split decomposition state
extern int g_rank, g_nprocs;
extern float g_x_lo, g_x_hi; // this rank's X band, [x_lo, x_hi)
extern int g_x_neighbour_left; // rank at smaller x, or MPI_PROC_NULL
extern int g_x_neighbour_right; // rank at larger x, or MPI_PROC_NULL
extern int g_xcut[REBAL_MAX_PROCS + 1];
extern int g_x_imin, g_x_imax; // this rank's chem slice range
extern int g_min_w; // the halo safety floor on a band, in chem slices

// host-side 3D chem fields and per-type cell counts, indexed [i][j][k]
extern float ***chem1, ***chem2, ***chem_diff;
extern int ***grid, ***grid1, ***grid2, ***grid3, ***grid4;

// wall time spent in output_mpi, split into file I/O and MPI
extern double g_output_io, g_output_comm;

// uniform float in [0, 1) from the process rand() stream
float RNG(void);

#endif // SIM_GLOBALS_H
