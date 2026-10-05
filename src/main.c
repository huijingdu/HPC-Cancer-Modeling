#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#if 0
#ifdef cl_khr_fp64
    #error "1Double precision floating point not supported by OpenCL implementation."
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #error "2Double precision floating point not supported by OpenCL implementation."
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "3Double precision floating point not supported by OpenCL implementation."
#endif
#endif

#include "mpi_assert.h"
#include "sim_globals.h"
#include "sim_types.h"
#include "domain.h"
#include "io.h"
#include "biology.h"
#include "sim.h"

// reports elements sitting outside the bin grid. the grid refits itself during the
// run, so this is a note that the initial padding was tight, not a failure
static void verifyDomain(float * x, float * y, float * z, int max_ele_no) {
    int i;
    int lo_x = 0, hi_x = 0, lo_y = 0, hi_y = 0, lo_z = 0, hi_z = 0;

    float hx = g_bin_ox + g_nbx * BIN_SIZE;
    float hy = g_bin_oy + g_nby * BIN_SIZE;
    float hz = g_bin_oz + g_nbz * BIN_SIZE;

    for (i = 0; i < max_ele_no; ++i) {
        if (x[i] < g_bin_ox) lo_x++;
        if (x[i] > hx) hi_x++;
        if (y[i] < g_bin_oy) lo_y++;
        if (y[i] > hy) hi_y++;
        if (z[i] < g_bin_oz) lo_z++;
        if (z[i] > hz) hi_z++;
    }

    if (lo_x || hi_x)
        fprintf(stderr, "WARNING %d elements below %.2f and %d above %.2f in x\n", lo_x, g_bin_ox, hi_x, hx);
    if (lo_y || hi_y)
        fprintf(stderr, "WARNING %d elements below %.2f and %d above %.2f in y\n", lo_y, g_bin_oy, hi_y, hy);
    if (lo_z || hi_z)
        fprintf(stderr, "WARNING %d elements below %.2f and %d above %.2f in z\n", lo_z, g_bin_oz, hi_z, hz);
}

int main (int argc, char * argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);

    int i, j, k;

    char * InName = argv[1];
    int sniff_cell_no;
    float mx, my, mz, nx_min, ny_min, nz_min;
    sniff_ic(InName, &sniff_cell_no, &mx, &my, &mz, &nx_min, &ny_min, &nz_min);

    // size from the real extent, not the max alone, and shift the cloud off zero
    float span_x = mx - nx_min;
    float span_y = my - ny_min;
    float span_z = mz - nz_min;

    float pad_x = span_x * DOMAIN_PAD_FRAC;
    float pad_y = span_y * DOMAIN_PAD_FRAC;
    float pad_z = span_z * DOMAIN_PAD_FRAC;

    g_shift_x = pad_x - nx_min;
    g_shift_y = pad_y - ny_min;
    g_shift_z = pad_z - nz_min;
    g_sub_z = nz_min + g_shift_z; // the IC floor

    g_lx = span_x + 2.0f * pad_x;
    g_ly = span_y + 2.0f * pad_y;
    g_lz = span_z + 2.0f * pad_z;

    g_cell_lo_x = pad_x;
    g_cell_hi_x = pad_x + span_x;

    // the chem grid wraps the cells only. at a spacing of 1.0 its element count is
    // the cube of the extent, so folding the bin padding in here costs gigabytes
    g_chem_ox = pad_x - CHEM_MARGIN;
    g_chem_oy = pad_y - CHEM_MARGIN;
    g_chem_oz = pad_z - CHEM_MARGIN;
    g_chem_lx = span_x + 2.0f * CHEM_MARGIN;
    g_chem_ly = span_y + 2.0f * CHEM_MARGIN;
    g_chem_lz = span_z + 2.0f * CHEM_MARGIN;

    g_nx = (int)ceilf(g_chem_lx / g_dx_TARGET);
    g_ny = (int)ceilf(g_chem_ly / g_dx_TARGET);
    g_nz = (int)ceilf(g_chem_lz / g_dx_TARGET);
    g_dx = g_chem_lx / g_nx;
    g_dy = g_chem_ly / g_ny;
    g_dz = g_chem_lz / g_nz;

    // the bin grid starts on the padded box and is refitted from there as cells move
    fit_bin_grid(0.0f, g_lx, 0.0f, g_ly, 0.0f, g_lz);

    // 1-D Cartesian topology over X. periods are all zero, so the ranks at the
    // two ends of the box get MPI_PROC_NULL and never wrap around.
    {
        int dims[3] = {g_nprocs, 1, 1};
        int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart);
        MPI_Cart_shift(cart, 0, 1, &g_x_neighbour_left, &g_x_neighbour_right);
        MPI_Comm_free(&cart);
    }

    if (g_rank == 0) printf("[mpi] %d rank(s) over occupied [%.2f, %.2f) of lx %.2f\n",
                            g_nprocs, g_cell_lo_x, g_cell_hi_x, g_lx);
    if (g_rank == 0) printf("[grid] bins %dx%dx%d = %d, chem %dx%dx%d = %d\n",
                            g_nbx, g_nby, g_nbz, g_nbins, g_nx, g_ny, g_nz, g_nx*g_ny*g_nz);

    g_max_cell = (int)(sniff_cell_no * CELL_GROWTH);

    int max_ele_no = g_max_cell * MAX_ELE;
    int ele_no = 0;
    int cell_no = 0;
    float dt = 0.1;

    int n = 100;

    ele_arrays * e = (ele_arrays *)malloc(sizeof(ele_arrays));
    cell_arrays * c = (cell_arrays *)malloc(sizeof(cell_arrays));
    chem_grid * ch = (chem_grid *)malloc(sizeof(chem_grid));

    ele_alloc(e, max_ele_no);
    cell_alloc(c, g_max_cell);
    chem_alloc(ch, g_nx, g_ny, g_nz);

    chem1 = (float ***)malloc(g_nx*sizeof(float**));
    for(i = 0; i < g_nx; i++)
    {
        chem1[i] = (float **)malloc(g_ny*sizeof(float*));
        for(j = 0; j < g_ny; j++)
            chem1[i][j] = (float *)malloc(g_nz*sizeof(float));
    }
    chem2 = (float ***)malloc(g_nx*sizeof(float**));
    for(i = 0; i < g_nx; i++)
    {
        chem2[i] = (float **)malloc(g_ny*sizeof(float*));
        for(j = 0; j < g_ny; j++)
            chem2[i][j] = (float *)malloc(g_nz*sizeof(float));
    }
    chem_diff = (float ***)malloc(g_nx*sizeof(float**));
    for(i = 0; i < g_nx; i++)
    {
        chem_diff[i] = (float **)malloc(g_ny*sizeof(float*));
        for(j = 0; j < g_ny; j++)
            chem_diff[i][j] = (float *)malloc(g_nz*sizeof(float));
    }
    grid = (int ***)malloc(g_nx*sizeof(int**));
    for(i = 0; i < g_nx; i++)
    {
        grid[i] = (int **)malloc(g_ny*sizeof(int*));
        for(j = 0; j < g_ny; j++)
            grid[i][j] = (int *)malloc(g_nz*sizeof(int));
    }
    grid1 = (int ***)malloc(g_nx*sizeof(int**));
    for(i = 0; i < g_nx; i++)
    {
        grid1[i] = (int **)malloc(g_ny*sizeof(int*));
        for(j = 0; j < g_ny; j++)
            grid1[i][j] = (int *)malloc(g_nz*sizeof(int));
    }
    grid2 = (int ***)malloc(g_nx*sizeof(int**));
    for(i = 0; i < g_nx; i++)
    {
        grid2[i] = (int **)malloc(g_ny*sizeof(int*));
        for(j = 0; j < g_ny; j++)
            grid2[i][j] = (int *)malloc(g_nz*sizeof(int));
    }
    grid3 = (int ***)malloc(g_nx*sizeof(int**));
    for(i = 0; i < g_nx; i++)
    {
        grid3[i] = (int **)malloc(g_ny*sizeof(int*));
        for(j = 0; j < g_ny; j++)
            grid3[i][j] = (int *)malloc(g_nz*sizeof(int));
    }
    grid4 = (int ***)malloc(g_nx*sizeof(int**));
    for(i = 0; i < g_nx; i++)
    {
        grid4[i] = (int **)malloc(g_ny*sizeof(int*));
        for(j = 0; j < g_ny; j++)
            grid4[i][j] = (int *)malloc(g_nz*sizeof(int));
    }

	for(i=0;i<max_ele_no;i++)
    {
        e->x[i] = e->y[i] = e->z[i] = 0.0f;
        e->id[i] = e->ele_type[i] = 0;
    }
    for(i = 0; i < g_max_cell; i++)
    {
        c->ele_per_cell[i] = 0;
        c->cell_type[i] = 0;
        c->cclock[i] = 0;
        c->cycle[i] = 0;
    }
    
    InitialReader(InName, e->x, e->y, e->z, e->id, c->cell_type, e->ele_type, &cell_no, &ele_no, c->ele_per_cell, c->cclock, c->cycle);

    // shift into the padded domain before anything bins or partitions on position
    for (int ci = 0; ci < cell_no; ci++)
    {
        for (j = 0; j < c->ele_per_cell[ci]; j++)
        {
            k = ci * MAX_ELE + j;
            e->x[k] += g_shift_x;
            e->y[k] += g_shift_y;
            e->z[k] += g_shift_z;
        }
    }

    assign_x_cuts_balanced(e->x, e->y, e->z, c->ele_per_cell, cell_no);

    printf("[rank %d] x band [%.2f, %.2f) chem slices [%d, %d) left=%d right=%d\n",
           g_rank, g_x_lo, g_x_hi, g_x_imin, g_x_imax,
           g_x_neighbour_left, g_x_neighbour_right);

    // every rank reads the whole IC, then drops the cells outside its own X band.
    // the two end bands are open ended, so nothing drifting into the padding is lost.
    // arrays are compacted in place so element blocks stay MAX_ELE strided, and
    // id[] is renumbered because cellcenter.cl matches on (id[k] - 1) == cell slot.
    {
        int kept_cell = 0;
        int kept_ele = 0;
        for (int ci = 0; ci < cell_no; ci++)
        {
            int n = c->ele_per_cell[ci];
            if (n == 0) continue;

            float sum_x = 0.0f;
            for (j = 0; j < n; j++) sum_x += e->x[ci * MAX_ELE + j];
            float mean_x = sum_x / (float)n;

            int keep = 1;
            if (g_rank > 0 && mean_x < g_x_lo) keep = 0;
            if (g_rank < g_nprocs - 1 && mean_x >= g_x_hi) keep = 0;
            if (!keep) continue;

            for (j = 0; j < n; j++)
            {
                int src = ci * MAX_ELE + j;
                int dst = kept_cell * MAX_ELE + j;
                e->x[dst] = e->x[src];
                e->y[dst] = e->y[src];
                e->z[dst] = e->z[src];
                e->ele_type[dst] = e->ele_type[src];
                e->id[dst] = kept_cell + 1;
            }
            c->ele_per_cell[kept_cell] = n;
            c->cell_type[kept_cell] = c->cell_type[ci];
            c->cclock[kept_cell] = c->cclock[ci];
            c->cycle[kept_cell] = c->cycle[ci];

            kept_cell++;
            kept_ele += n;
        }

        // clear the vacated tail so no stale cell can be read back as live
        for (int ci = kept_cell; ci < cell_no; ci++)
        {
            c->ele_per_cell[ci] = 0;
            c->cell_type[ci] = 0;
            c->cclock[ci] = 0;
            c->cycle[ci] = 0;
        }
        for (int ele = kept_cell * MAX_ELE; ele < cell_no * MAX_ELE; ele++)
        {
            e->x[ele] = e->y[ele] = e->z[ele] = 0.0f;
            e->id[ele] = e->ele_type[ele] = 0;
        }

        cell_no = kept_cell;
        ele_no = kept_ele;
    }

    printf("[rank %d] after compaction: cell_no=%d ele_no=%d\n", g_rank, cell_no, ele_no);

    initial_chem_paras(e->x, e->y, e->z, e->id, c->cell_type, e->ele_type, cell_no, c->ele_per_cell);

    // Do the OpenCL calculation
    double t_setup = 0.0, t_loop = 0.0, t_comm = 0.0, t_io = 0.0;
    runCL(e, c, ch, dt,
          max_ele_no, ele_no, cell_no,
          &t_setup, &t_loop, &t_comm, &t_io);

    // compute is the loop with the measured comm and io taken out
    printf("[TIMING] IC=%s rank=%d cells=%d setup=%.2f loop=%.2f comm=%.2f io=%.2f "
           "compute=%.2f wall=%.2f s\n",
           InName, g_rank, cell_no, t_setup, t_loop, t_comm, t_io,
           t_loop - t_comm - t_io, t_setup + t_loop);
    fflush(stdout);


    // Verify if the elements remained within the domain
    verifyDomain(e->x, e->y, e->z, max_ele_no);

    // Free up memory
    ele_free(e);
    cell_free(c);
    chem_free(ch);

    MPI_Finalize();

    return 0;
}
