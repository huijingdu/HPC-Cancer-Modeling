#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "mpi_assert.h"
#include "sim_globals.h"
#include "sim_types.h"
#include "sim.h"
#include "domain.h"
#include "halo.h"
#include "io.h"
#include "ocl_setup.h"

#if REBALANCE_FR > 0 && REBAL_CONTROLLER == 1
    #include "rnn/rnn_balancer.h"
#endif

#pragma mark -
#pragma mark Main OpenCL Routine
int runCL(ele_arrays * e, cell_arrays * c, chem_grid * ch, float dt,
          int max_ele_no, int ele_no, int cell_no,
          double *out_setup, double *out_loop, double *out_comm, double *out_io)
{
    double t_func_start = MPI_Wtime();

    // t_comm accumulates the wall time spent inside MPI calls in the loop
    double t_comm = 0.0;
    double t_comm0 = 0.0;

    int flag_print = 0;
	
    ocl_ctx ocl = {0};

    cl_int err = 0;
    size_t buffer_size;
	
    // alias arrays for consistancy
    float * x = e->x, * y = e->y, * z = e->z;
    float * xF = e->xF, * yF = e->yF, * zF = e->zF;
    float * xo = e->xo, * yo = e->yo, * zo = e->zo;
    int * id = e->id, * ele_type = e->ele_type;
    unsigned int * rng_x = e->rng_x, * rng_c = e->rng_c;

    float * xc = c->xc, * yc = c->yc, * zc = c->zc;
    float * OVOL1 = c->OVOL1, * OVOL2 = c->OVOL2;
    int * cell_type = c->cell_type, * ele_per_cell = c->ele_per_cell;
    int * cclock = c->cclock, * cycle = c->cycle;
    int * flag = c->flag, * flag_gener = c->flag_gener, * flag_death = c->flag_death;

    float * vx = ch->vx, * vy = ch->vy, * vz = ch->vz;
    float * chem1_gpu = ch->chem1_gpu, * chem2_gpu = ch->chem2_gpu;
    float * chem11_gpu = ch->chem11_gpu, * chem22_gpu = ch->chem22_gpu;
    int * grid_gpu = ch->grid_gpu;
    int * grid1_gpu = ch->grid1_gpu, * grid2_gpu = ch->grid2_gpu;
    int * grid3_gpu = ch->grid3_gpu, * grid4_gpu = ch->grid4_gpu;

    // fixed seed so a run is reproducible, offset per rank so the bands do not
    // draw identical streams
    srand(42 + g_rank);

    for(int i = 0; i < max_ele_no; i++)
    {
        xF[i] = yF[i] = zF[i] = 0.0f;
        xo[i] = yo[i] = zo[i] = 0.0f;
    }
    for(int i = 0; i < g_max_cell; i++)
    {
        xc[i] = yc[i] = zc[i] = 0.0;
        flag[i] = flag_death[i] = flag_gener[i] = 0;
        OVOL1[i] = 0.9;
        OVOL2[i] = 0.9;
    }

    for(int i = 0; i < max_ele_no; i++)
    {
        rng_x[i] = rand();
        rng_c[i] = rand();
    }

    ocl_bufs bufs = {0};

    int scan_nblocks = 0;

    ocl_init(&ocl);
    cl_device_id device_id = ocl.device;
    cl_context context = ocl.context;
    cl_command_queue cmd_queue = ocl.queue;
	
#pragma mark Program and Kernel Creation 
    ocl_build_kernels(&ocl);
    cl_kernel * kernel = ocl.kernel;
    size_t scan_wg = ocl.scan_wg;

    scan_nblocks = (g_nbins + (int)scan_wg - 1) / (int)scan_wg;
    if (g_rank == 0) printf("[bin] scan work group %d, %d group(s) over %d bins\n",
                            (int)scan_wg, scan_nblocks, g_nbins);
    
    ocl_alloc_buffers(&bufs, &ocl, e, c, ch, max_ele_no, scan_nblocks);
    cl_mem x_mem = bufs.x_mem, y_mem = bufs.y_mem, z_mem = bufs.z_mem;
    cl_mem xc_mem = bufs.xc_mem, yc_mem = bufs.yc_mem, zc_mem = bufs.zc_mem;
    cl_mem id_mem = bufs.id_mem, cell_type_mem = bufs.cell_type_mem, ele_type_mem = bufs.ele_type_mem;
    cl_mem xF_mem = bufs.xF_mem, yF_mem = bufs.yF_mem, zF_mem = bufs.zF_mem;
    cl_mem ele_per_cell_mem = bufs.ele_per_cell_mem;
    cl_mem flag_mem = bufs.flag_mem, flag_death_mem = bufs.flag_death_mem, flag_gener_mem = bufs.flag_gener_mem;
    cl_mem chem1_gpu_mem = bufs.chem1_gpu_mem, chem2_gpu_mem = bufs.chem2_gpu_mem;
    cl_mem chem11_gpu_mem = bufs.chem11_gpu_mem, chem22_gpu_mem = bufs.chem22_gpu_mem;
    cl_mem grid_gpu_mem = bufs.grid_gpu_mem, grid1_gpu_mem = bufs.grid1_gpu_mem, grid2_gpu_mem = bufs.grid2_gpu_mem,
           grid3_gpu_mem = bufs.grid3_gpu_mem, grid4_gpu_mem = bufs.grid4_gpu_mem;
    cl_mem rng_x_mem = bufs.rng_x_mem, rng_c_mem = bufs.rng_c_mem;
    cl_mem o1_mem = bufs.o1_mem, o2_mem = bufs.o2_mem;
    cl_mem vx_mem = bufs.vx_mem, vy_mem = bufs.vy_mem, vz_mem = bufs.vz_mem;
    cl_mem xo_mem = bufs.xo_mem, yo_mem = bufs.yo_mem, zo_mem = bufs.zo_mem;
    cl_mem cell_list_mem = bufs.cell_list_mem, bin_bounds_mem = bufs.bin_bounds_mem, bin_overflow_mem = bufs.bin_overflow_mem;

    ocl_set_args(&ocl, &bufs, cell_no, ele_no, dt, scan_nblocks);

#pragma mark Execution and Read

#if 0
    // update the initial chemical field
    for(int i = 0; i < g_nx; i++)    
        for(int j = 0; j < g_ny; j++)    
            for(int k = 0; k < g_nz; k++) 
    {
        int indx = i + j*g_nx + k*g_nx*g_ny;
        chem1_gpu[indx] = chem1[i][j][k];
        chem2_gpu[indx] = chem2[i][j][k];
        grid_gpu[indx] = grid[i][j][k];
        grid1_gpu[indx] = grid1[i][j][k];
        grid2_gpu[indx] = grid2[i][j][k];
        grid3_gpu[indx] = grid3[i][j][k];
        grid4_gpu[indx] = grid4[i][j][k];
    }   
    err = clEnqueueWriteBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float),
                        (void*)chem1_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float),
                        (void*)chem2_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(int),
                        (void*)grid_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid1_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(int),
                        (void*)grid1_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid2_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(int),
                        (void*)grid2_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid3_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(int),
                        (void*)grid3_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid4_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(int),
                        (void*)grid4_gpu, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
#endif

    {
    // update cell centers
    size_t global_work_size = cell_no;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_CELLCENTER], 1, NULL,
                                 &global_work_size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    clFinish(cmd_queue);

    // index to note how many empty spots in the cell array due to death
    int no_empty = 0;

    // cells this rank owns. halo cells sit above this mark and are never owned
    int local_cell_no = cell_no;

    xfer_buf halo_left, halo_right;
    xfer_alloc(&halo_left, g_max_cell, GHOST_FLOATS_PER_CELL, GHOST_INTS_PER_CELL);
    xfer_alloc(&halo_right, g_max_cell, GHOST_FLOATS_PER_CELL, GHOST_INTS_PER_CELL);

    xfer_buf mig_left, mig_right;
    xfer_alloc(&mig_left, g_max_cell, MIG_FLOATS_PER_CELL, MIG_INTS_PER_CELL);
    xfer_alloc(&mig_right, g_max_cell, MIG_FLOATS_PER_CELL, MIG_INTS_PER_CELL);

    // seed host cell centres for the first halo selection. x, y, z are already
    // valid from InitialReader
    if (local_cell_no > 0)
    {
        err = clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0, local_cell_no * sizeof(float),
                                  xc, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0, local_cell_no * sizeof(float),
                                  yc, 0, NULL, NULL);
        err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0, local_cell_no * sizeof(float),
                                  zc, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
    }

#if REBALANCE_FR > 0
    double t_local_prev = 0.0;
    double t_rebal = 0.0;
    int n_rebal = 0;
    long long n_slices_moved = 0;
#endif

    // start of time iterations
    int iterations = MAX_ITERATIONS;
#if REBALANCE_FR > 0 && REBAL_CONTROLLER == 1
    RebalRNN *rnn_ctl = (g_rank == 0 && g_nprocs > 1)
                      ? rebal_rnn_create(g_nprocs, REBAL_RNN_WEIGHTS) : NULL;
    if (g_rank == 0)
    {
        if (rnn_ctl)
            printf("[rebal] RNN controller active from %s: %d features, "
                   "max_step %d vs min_w %d%s\n",
                   REBAL_RNN_WEIGHTS, rebal_rnn_n_features(), rebal_rnn_max_step(),
                   g_min_w,
                   (rebal_rnn_max_step() < g_min_w) ? "" : "  [WARNING: step >= min_w, "
                   "migration may not be single-hop]");
        else
            printf("[rebal] RNN controller UNAVAILABLE (could not load %s), "
                   "falling back to the heuristic\n", REBAL_RNN_WEIGHTS);
        fflush(stdout);
    }
#endif

    double t_loop0 = MPI_Wtime();
    for(int iter = 0; iter < iterations; iter++)
    {
#if 0
        // start of cell updates
        if((iter+1)%GROWTH_FR == 0)
        {
            err = clEnqueueReadBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  x, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  y, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  z, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xo_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  xo, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yo_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  yo, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zo_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                                  zo, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            // map cells to a grid using their locations
            cell_in_block(x, y, z, id, cell_type, ele_type, cell_no, ele_per_cell, xo, yo, zo, vx, vy, vz);
            for(int i = 0; i < g_nx; i++)    
                for(int j = 0; j < g_ny; j++)    
                    for(int k = 0; k < g_nz; k++) 
            {
                int indx = i + j*g_nx + k*g_nx*g_ny;
                grid_gpu[indx] = grid[i][j][k];
                grid1_gpu[indx] = grid1[i][j][k];
                grid2_gpu[indx] = grid2[i][j][k];
                grid3_gpu[indx] = grid3[i][j][k];
                grid4_gpu[indx] = grid4[i][j][k];
            }   

            buffer_size = sizeof(int) * g_nx*g_ny*g_nz;
            err = clEnqueueWriteBuffer(cmd_queue, grid_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)grid_gpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, grid1_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)grid1_gpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, grid2_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)grid2_gpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, grid3_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)grid3_gpu, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, grid4_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)grid4_gpu, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            buffer_size = sizeof(float) * g_nx*g_ny*g_nz;
            err = clEnqueueWriteBuffer(cmd_queue, vx_mem, CL_TRUE, 0, buffer_size,
                                (void*)vx, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, vy_mem, CL_TRUE, 0, buffer_size,
                                (void*)vy, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, vz_mem, CL_TRUE, 0, buffer_size,
                                (void*)vz, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            /*
            // update chemical field
            size_t global_work_size = g_nx*g_ny*g_nz;
            for(int iter_chem = 0; iter_chem < MAX_CHEM_ITER; iter_chem++)
            {
                err = clSetKernelArg(kernel[K_CHEM_RK], 15, sizeof(int), &(iter_chem));
                err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_CHEM_RK], 1, NULL,
                                         &global_work_size, NULL, 0, NULL, NULL);
                assert(err == CL_SUCCESS);

                err = clEnqueueCopyBuffer(cmd_queue, chem11_gpu_mem, chem1_gpu_mem, 0, 0, g_nx*g_ny*g_nz*sizeof(float),
                                  0, 0, NULL);
                err |= clEnqueueCopyBuffer(cmd_queue, chem22_gpu_mem, chem2_gpu_mem, 0, 0, g_nx*g_ny*g_nz*sizeof(float),
                                  0, 0, NULL);
                assert(err == CL_SUCCESS);
            }
            clFinish(cmd_queue);
            */

            /*
            // update intracellular gene network
            global_work_size = cell_no;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_OVOL], 1, NULL,
                                         &global_work_size, NULL, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            clFinish(cmd_queue);
            */

            err = clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              cell_type, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, o1_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              OVOL1, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

        } // end of cell updates
#endif
        // halo exchange: cells within GHOST_DEPTH_F of a band edge go across as
        // read-only ghosts. no coordinate shift, the split is non-periodic.
        int combined_cell_no = local_cell_no;
        if (g_nprocs > 1)
        {
            int n_send_left = 0, n_send_right = 0;
            int n_recv_left = 0, n_recv_right = 0;

            for (int i = 0; i < local_cell_no; i++)
            {
                if (ele_per_cell[i] == 0) continue;
                float xci = xc[i];
                if (xci >= g_x_lo && xci < g_x_lo + GHOST_DEPTH_F)
                    halo_left.send_idx[n_send_left++] = i;
                if (xci >= g_x_hi - GHOST_DEPTH_F && xci < g_x_hi)
                    halo_right.send_idx[n_send_right++] = i;
            }

            // counts first. what I send left lands in my left neighbour's right halo,
            // so the matching receive comes from the right
            t_comm0 = MPI_Wtime();
            MPI_Sendrecv(&n_send_left, 1, MPI_INT, g_x_neighbour_left, 0,
                         &n_recv_right, 1, MPI_INT, g_x_neighbour_right, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&n_send_right, 1, MPI_INT, g_x_neighbour_right, 1,
                         &n_recv_left, 1, MPI_INT, g_x_neighbour_left, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - t_comm0;

            for (int s = 0; s < n_send_left; s++)
                halo_pack(&halo_left, s, halo_left.send_idx[s], c, e);
            for (int s = 0; s < n_send_right; s++)
                halo_pack(&halo_right, s, halo_right.send_idx[s], c, e);

            t_comm0 = MPI_Wtime();
            MPI_Sendrecv(halo_left.send_f, n_send_left * GHOST_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_left, 2,
                         halo_right.recv_f, n_recv_right * GHOST_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_right, 2,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(halo_left.send_i, n_send_left * GHOST_INTS_PER_CELL, MPI_INT, g_x_neighbour_left, 3,
                         halo_right.recv_i, n_recv_right * GHOST_INTS_PER_CELL, MPI_INT, g_x_neighbour_right, 3,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(halo_right.send_f, n_send_right * GHOST_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_right, 4,
                         halo_left.recv_f, n_recv_left * GHOST_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_left, 4,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(halo_right.send_i, n_send_right * GHOST_INTS_PER_CELL, MPI_INT, g_x_neighbour_right, 5,
                         halo_left.recv_i, n_recv_left * GHOST_INTS_PER_CELL, MPI_INT, g_x_neighbour_left, 5,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - t_comm0;

            int n_recv_total = n_recv_left + n_recv_right;
            assert(local_cell_no + n_recv_total <= g_max_cell);

            // ghosts occupy slots above local_cell_no, left arrivals then right
            for (int r = 0; r < n_recv_left; r++)
                halo_unpack(&halo_left, r, local_cell_no + r, c, e);
            for (int r = 0; r < n_recv_right; r++)
                halo_unpack(&halo_right, r, local_cell_no + n_recv_left + r, c, e);

            combined_cell_no = local_cell_no + n_recv_total;

            // push only the ghost tail; the local range is already on the device
            if (n_recv_total > 0)
            {
                size_t ele_off = (size_t)local_cell_no * MAX_ELE * sizeof(float);
                size_t ele_cnt = (size_t)n_recv_total * MAX_ELE * sizeof(float);
                size_t cell_off_f = (size_t)local_cell_no * sizeof(float);
                size_t cell_cnt_f = (size_t)n_recv_total * sizeof(float);
                size_t cell_off_i = (size_t)local_cell_no * sizeof(int);
                size_t cell_cnt_i = (size_t)n_recv_total * sizeof(int);

                err = clEnqueueWriteBuffer(cmd_queue, x_mem, CL_TRUE, ele_off, ele_cnt,
                                           x + local_cell_no * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, y_mem, CL_TRUE, ele_off, ele_cnt,
                                           y + local_cell_no * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, z_mem, CL_TRUE, ele_off, ele_cnt,
                                           z + local_cell_no * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, xc_mem, CL_TRUE, cell_off_f, cell_cnt_f,
                                           xc + local_cell_no, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, yc_mem, CL_TRUE, cell_off_f, cell_cnt_f,
                                           yc + local_cell_no, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, zc_mem, CL_TRUE, cell_off_f, cell_cnt_f,
                                           zc + local_cell_no, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, cell_off_i, cell_cnt_i,
                                           ele_per_cell + local_cell_no, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, cell_off_i, cell_cnt_i,
                                           cell_type + local_cell_no, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }
        }

        size_t gws_cells = combined_cell_no;
        int zero_int = 0;

        err = clSetKernelArg(kernel[K_BIN_COUNT], 4, sizeof(int), &combined_cell_no); // bin_count cell_no
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 4, sizeof(int), &combined_cell_no); // bin_scatter cell_no
        assert(err == CL_SUCCESS);

        // count into the current grid, and refit it if any center landed outside.
        // clamping a stray cell into an edge bin would both lose its neighbours and
        // make that bin a hotspot, so the grid moves instead
        for (int fit = 0; ; fit++)
        {
            int overflow = 0;

            err = clEnqueueFillBuffer(cmd_queue, bin_overflow_mem, &zero_int, sizeof(int),
                                      0, sizeof(int), 0, NULL, NULL);
            // zero per-bin counter (reused as the scatter cursor after the prefix sum)
            err |= clEnqueueFillBuffer(cmd_queue, bufs.bin_count_mem, &zero_int, sizeof(int),
                                       0, g_nbins * sizeof(int), 0, NULL, NULL);
            err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_BIN_COUNT], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, bin_overflow_mem, CL_TRUE, 0,
                                       sizeof(int), &overflow, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            if (overflow == 0) break;

            if (fit == BIN_FIT_TRIES)
            {
                fprintf(stderr, "[rank %d] bin grid still misses %d cells after %d refits\n",
                        g_rank, overflow, fit);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // seeding min and max from the box centre keeps the result a superset of
            // the cell extent, so a refit can only ever be too generous
            int seed_x = (int)floorf(g_bin_ox + 0.5f * g_nbx * BIN_SIZE);
            int seed_y = (int)floorf(g_bin_oy + 0.5f * g_nby * BIN_SIZE);
            int seed_z = (int)floorf(g_bin_oz + 0.5f * g_nbz * BIN_SIZE);
            int b[6] = {seed_x, seed_x, seed_y, seed_y, seed_z, seed_z};

            err = clEnqueueWriteBuffer(cmd_queue, bin_bounds_mem, CL_FALSE, 0,
                                       sizeof(b), b, 0, NULL, NULL);
            err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 4, sizeof(int), &combined_cell_no);
            err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_CELL_BOUNDS], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, bin_bounds_mem, CL_TRUE, 0,
                                       sizeof(b), b, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            int old_nbins = g_nbins;
            fit_bin_grid((float)b[0], (float)b[1] + 1.0f, (float)b[2], (float)b[3] + 1.0f,
                         (float)b[4], (float)b[5] + 1.0f);
            scan_nblocks = (g_nbins + (int)scan_wg - 1) / (int)scan_wg;

            if (g_nbins > old_nbins)
            {
                ocl_regrow_bins(&ocl, &bufs, scan_nblocks);
                printf("[REFIT] iter %d, try %d, bins %d -> %d\n", iter, fit, old_nbins, g_nbins);
                fflush(stdout);
            }
            

            ocl_set_bin_grid_args(&ocl, scan_nblocks);

        }

        // three-level scan: scan within each group, scan the group totals, then
        // shift each group by its offset. the queue is in-order, so no events
        size_t gws_scan = (size_t)scan_nblocks * scan_wg;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_BIN_BLOCK_SCAN], 1, NULL, &gws_scan, &scan_wg, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_BIN_SCAN_BLOCK_SUMS], 1, NULL, &scan_wg, &scan_wg, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_BIN_ADD_BLOCK_OFFSETS], 1, NULL, &gws_scan, &scan_wg, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[K_BIN_SCATTER], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);

        // cell movement. ghosts are integrated too and their results discarded,
        // which keeps the dispatch one contiguous range
        int combined_ele_no = combined_cell_no * MAX_ELE;
        err = clSetKernelArg(kernel[K_MOVEMENT],  5, sizeof(int), &combined_cell_no);
        err |= clSetKernelArg(kernel[K_MOVEMENT],  6, sizeof(int), &combined_ele_no);
        assert(err == CL_SUCCESS);
        // global_work_size = max_ele_no;
        global_work_size = combined_cell_no * MAX_ELE;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_MOVEMENT], 1, NULL, 
									 &global_work_size, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
        clFinish(cmd_queue);

        err = clEnqueueCopyBuffer(cmd_queue, x_mem, xo_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        assert(err == CL_SUCCESS);
        err |= clEnqueueCopyBuffer(cmd_queue, y_mem, yo_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        err |= clEnqueueCopyBuffer(cmd_queue, z_mem, zo_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        assert(err == CL_SUCCESS);
        err = clEnqueueCopyBuffer(cmd_queue, xF_mem, x_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        err |= clEnqueueCopyBuffer(cmd_queue, yF_mem, y_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        err |= clEnqueueCopyBuffer(cmd_queue, zF_mem, z_mem, 0, 0, max_ele_no*sizeof(float),
                                  0, 0, NULL);
        assert(err == CL_SUCCESS);
        clFinish(cmd_queue);


        // start of cell growth
        if((iter-1)%GROWTH_FR == 0 && local_cell_no > 0)
        {
            // owned cells only. the slots above hold this step's ghosts, which are
            // read-only copies: growing one would diverge from its owner and then be
            // overwritten by the next halo exchange
            for(int i = 0; i < local_cell_no; i++)
            {
                flag[i] = 0;
            }

            buffer_size = sizeof(int) * g_max_cell;
            err = clEnqueueWriteBuffer(cmd_queue, flag_mem, CL_TRUE, 0, buffer_size,
                                       (void*)flag, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, flag_death_mem, CL_TRUE, 0, buffer_size,
                                       (void*)flag_death, 0, NULL, NULL);
            err |= clSetKernelArg(kernel[K_GROWTH], 5, sizeof(int), &ele_no);
            assert(err == CL_SUCCESS);

            global_work_size = local_cell_no;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_GROWTH], 1, NULL,
                                         &global_work_size, NULL, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            clFinish(cmd_queue);

            err = clEnqueueReadBuffer(cmd_queue, flag_mem, CL_TRUE, 0, local_cell_no*sizeof(int),
                                      flag, 0, NULL, NULL);
            // the kernel appended an element, so the host mirror of ele_per_cell is
            // stale. the halo pack, the division gate and the type tally all read it
            err |= clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0,
                                       local_cell_no*sizeof(int), ele_per_cell, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            for(int i = 0; i < local_cell_no; i++)
            {
                if(flag[i] == 1) ele_no++;
            }
        } // end of cell growth

        // start of cell division
        if((iter-1)%DIVISION_FR == 0 && local_cell_no > 0 && local_cell_no < g_max_cell)
        {
            err = clEnqueueReadBuffer(cmd_queue, id_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                              id, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                              x, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                              y, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                              z, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                              ele_type, 0, NULL, NULL);
            // owned range only: the slots above hold this step's ghosts
            err |= clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, local_cell_no*sizeof(int),
                              ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float),
                              chem1_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float),
                              chem2_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                              xc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                              yc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                              zc, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            // age once up front
            for(int i = 0; i < local_cell_no; i++)
            {
                if(ele_per_cell[i] <= 0) continue;
                cclock[i] += DIVISION_FR;
            }

            int tmp_cell_no = local_cell_no;
            float tmpx0, tmpy0, tmpz0;
            int tmpn0, tmptype0;
            float tmpx[MAX_ELE], tmpy[MAX_ELE], tmpz[MAX_ELE];
            int tmptype[MAX_ELE];
            int id_for_new_cell;
            int search_from = 0;
            for(int i = 0; i < local_cell_no; i++)
            {
                if(ele_per_cell[i] <= 0) continue; // dead, or migrated away
                if(flag_print && cell_type[i]==2)
                    printf("cell[%d]  cell_type %d  element_per_cell %d  cclock %d  cycle %d\n",
                            i, cell_type[i], ele_per_cell[i], cclock[i], cycle[i]);
                if(cell_type[i] >= 3) continue; // cell no division
                if(ele_per_cell[i] < 16) continue;
                if(cclock[i] < cycle[i]) continue; // cell not mature

                tmpx0 = tmpy0 = tmpz0 = 0.0;
                tmpn0 = tmptype0 = 0;

                for(int j = 0; j < ele_per_cell[i]; j++)
                {
                    int k = i*MAX_ELE + j;
                    tmpx0 += x[k];
                    tmpy0 += y[k];
                    tmpz0 += z[k];
                    if(id[k] == (i+1)) tmpn0++;
                    if(ele_type[k] == 2) tmptype0++;

                    tmpx[j] = x[k];
                    tmpy[j] = y[k];
                    tmpz[j] = z[k];
                    tmptype[j] = ele_type[k];
                }

                if(tmpn0 <= 0 || tmpn0 != ele_per_cell[i] || ele_per_cell[i] < 0)
                {
                    fprintf(stderr, "[rank %d] cell %d owns %d of its %d elements\n",
                            g_rank, i, tmpn0, ele_per_cell[i]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                tmpx0 = tmpx0/tmpn0;
                tmpy0 = tmpy0/tmpn0;
                tmpz0 = tmpz0/tmpn0;

                // determine the two daughter cell types
                int ctype1 = 0, ctype2 = 0;
                // cell_lineage(i, xc[i], yc[i], zc[i], cell_type[i], &ctype1, &ctype2, chem1_gpu, chem2_gpu, OVOL1, OVOL2, flag_gener[i]);
                // cell_type[i] = (ctype1 <= ctype2)? ctype1 : ctype2; // the small type goes down
                // cell_type[id_for_new_cell] = (ctype1 >= ctype2)? ctype1 : ctype2; // the big type goes up
                if(cell_type[i] == 1) // mother is slow-dividing cell
                {
                    ctype1 = 1;
                    ctype2 = (RNG() < R_DIFF)? 2 : 1;
                }
                else // mother is fast-dividing cell
                {
                    ctype1 = 2;
                    ctype2 = 2;
                }

                // the daughter types pick the split axis. a one-sided split would
                // give the daughter a slot and no elements, so skip it instead
                int n_mother = 0;
                for(int j = 0; j < ele_per_cell[i]; j++)
                {
                    if(ctype1 == ctype2) { if(tmpx[j] > tmpx0) n_mother++; }
                    else                 { if(tmpz[j] <= tmpz0) n_mother++; }
                }
                if(n_mother == 0 || n_mother == ele_per_cell[i]) continue;

                // reuse a slot freed by death or migration, else extend the range
                id_for_new_cell = -1;
                for(int j = search_from; j < tmp_cell_no; j++)
                {
                    if(ele_per_cell[j] == 0) { id_for_new_cell = j; break; }
                }
                if(id_for_new_cell == -1)
                {
                    if(tmp_cell_no >= g_max_cell)
                    {
                        fprintf(stderr, "[rank %d] division needs slot %d, past g_max_cell %d\n",
                                g_rank, tmp_cell_no, g_max_cell);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    id_for_new_cell = tmp_cell_no;
                    tmp_cell_no++;
                }
                search_from = id_for_new_cell + 1;

                cell_type[i] = ctype1;
                cell_type[id_for_new_cell] = ctype2;

                // a reused slot still carries the previous occupant's state
                flag_gener[id_for_new_cell] = flag_gener[i];
                OVOL1[id_for_new_cell] = OVOL1[i];
                OVOL2[id_for_new_cell] = OVOL2[i];

                flag_death[i] = flag_death[id_for_new_cell] = 0;

                cclock[i] = cclock[id_for_new_cell] = 0;
                if(cell_type[i] == 1)
                    cycle[i] = (int)(1000.0f*(C1_MEAN + RNG()*C1_STD));
                else if(cell_type[i] == 2)
                    cycle[i] = (int)(1000.0f*(C2_MEAN + RNG()*C2_STD));
                else
                {
                    fprintf(stderr, "ERROR::cell division, wrong cell type = %d\n", cell_type[i]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                if(cell_type[id_for_new_cell] == 1)
                    cycle[id_for_new_cell] = (int)(1000.0f*(C1_MEAN + RNG()*C1_STD));
                else if(cell_type[id_for_new_cell] == 2)
                    cycle[id_for_new_cell] = (int)(1000.0f*(C2_MEAN + RNG()*C2_STD));
                else
                {
                    fprintf(stderr, "ERROR::cell division, wrong cell type = %d\n",
                            cell_type[id_for_new_cell]);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }

                // copy data in tmp to two daughter cells
                int tmp_ele_per_cell = ele_per_cell[i];
                ele_per_cell[i] = 0;
                ele_per_cell[id_for_new_cell] = 0;
                int count_ele_type2_1 = 0, count_ele_type2_2 = 0;
                if(ctype1 == ctype2)
                {
                  for(int j = 0; j < tmp_ele_per_cell; j++)
                  {
                    if(tmpx[j] > tmpx0)
                    {
                        int k = i*MAX_ELE + ele_per_cell[i];
                        x[k] = tmpx[j];
                        y[k] = tmpy[j];
                        z[k] = tmpz[j];
                        id[k] = i+1;
                        ele_type[k] = tmptype[j];
                        if(ele_type[k] == 2) count_ele_type2_1++;
                        ele_per_cell[i]++;
                    }
                    else
                    {
                        int k = id_for_new_cell*MAX_ELE + ele_per_cell[id_for_new_cell];
                        x[k] = tmpx[j];
                        y[k] = tmpy[j];
                        z[k] = tmpz[j];
                        id[k] = id_for_new_cell+1;
                        ele_type[k] = tmptype[j];
                        if(ele_type[k] == 2) count_ele_type2_2++;
                        ele_per_cell[id_for_new_cell]++;
                    }
                  }
                }
                else // if(ctype1 != ctype2)
                {
                  for(int j = 0; j < tmp_ele_per_cell; j++)
                  {
                    if(tmpz[j] <= tmpz0)
                    {
                        int k = i*MAX_ELE + ele_per_cell[i];
                        x[k] = tmpx[j];
                        y[k] = tmpy[j];
                        z[k] = tmpz[j];
                        id[k] = i+1;
                        ele_type[k] = tmptype[j];
                        if(ele_type[k] == 2) count_ele_type2_1++;
                        ele_per_cell[i]++;
                    }
                    else
                    {
                        int k = id_for_new_cell*MAX_ELE + ele_per_cell[id_for_new_cell];
                        x[k] = tmpx[j];
                        y[k] = tmpy[j];
                        z[k] = tmpz[j];
                        id[k] = id_for_new_cell+1;
                        ele_type[k] = tmptype[j];
                        if(ele_type[k] == 2) count_ele_type2_2++;
                        ele_per_cell[id_for_new_cell]++;
                    }
                  }
                }
                if(cell_type[i] == 1 && count_ele_type2_1 < 3)
                {
                    for(int j = 0; j < ele_per_cell[i]; j++)
                    {
                        int k = i*MAX_ELE + j;
                        if(ele_type[k] != 2)
                        {
                            ele_type[k] = 2;
                            count_ele_type2_1++;
                            if(count_ele_type2_1 == 3) break;
                        }
                    }
                }
                if(cell_type[i] != 1)
                {
                    for(int j = 0; j < ele_per_cell[i]; j++)
                    {
                        int k = i*MAX_ELE + j;
                        ele_type[k] = 1;
                    }
                }
                if(cell_type[id_for_new_cell] == 1 && count_ele_type2_2 < 3)
                {
                    for(int j = 0; j < ele_per_cell[id_for_new_cell]; j++)
                    {
                        int k = id_for_new_cell*MAX_ELE + j;
                        if(ele_type[k] != 2)
                        {
                            ele_type[k] = 2;
                            count_ele_type2_2++;
                            if(count_ele_type2_2 == 3) break;
                        }
                    }
                }
                if(cell_type[id_for_new_cell] != 1)
                {
                    for(int j = 0; j < ele_per_cell[id_for_new_cell]; j++)
                    {
                        int k = id_for_new_cell*MAX_ELE + j;
                        ele_type[k] = 1;
                    }
                }

            }
            local_cell_no = tmp_cell_no;

            err = clEnqueueWriteBuffer(cmd_queue, id_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                            (void*)id, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)x, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)y, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)z, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, local_cell_no*sizeof(int),
                            (void*)cell_type, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                            (void*)ele_type, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, local_cell_no*sizeof(int),
                            (void*)ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, flag_gener_mem, CL_TRUE, 0, local_cell_no*sizeof(int),
                            (void*)flag_gener, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, o1_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                            (void*)OVOL1, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, o2_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                            (void*)OVOL2, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        } // end of cell division

#if 0
        // start of cell death
        int FR_death = 500;
        if((iter-1)%FR_death == 0)
        {
            // update OVOL1 once since death runs more frequently
            global_work_size = cell_no;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_OVOL], 1, NULL,
                                         &global_work_size, NULL, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            clFinish(cmd_queue);

            err = clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              cell_type, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, o1_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              OVOL1, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            for(int i = 0; i < cell_no; i++)
            {
                if(ele_per_cell[i] == 0) continue;

                cclock[i] += FR_death;

                if(cell_type[i] == 1) continue;

                // type 4 cells shrink and die
                if(cell_type[i] == 4 && flag_death[i] == 1)
                {
                    if(ele_per_cell[i] > 0)
                    {
                        ele_per_cell[i]--;
                        if(ele_per_cell[i] == 0)
                        {
                            no_empty++;
                        }
                    }
                }
                // D3
                float d3 = 1.3;
                if(cell_type[i] == 4 && flag_death[i] == 0 && cclock[i] >= cycle[i]*d3) // type 4 --> death
                {
                    flag_death[i] = 1;
                }

                // D2
                float d2 = 1.5f/(1.0f + OVOL1[i]*10.0); // Ovol increase D2; decrease T2
                if(cell_type[i] == 3 && cclock[i] >= cycle[i]*d2) // type 3 --> type 4
                {
                    cell_type[i] = 4;
                    cclock[i] = 0;
                    flag_gener[i]++;
                }
            }
            err = clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)cell_type, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, flag_gener_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)flag_gener, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        } // end of cell death
#endif

        // update cell center. local only: ghost slots have the sender's id[], which
        // cellcenter.cl would reject by poisoning ele_per_cell with -1
        size_t global_work_size = local_cell_no;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[K_CELLCENTER], 1, NULL,
                                     &global_work_size, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
        clFinish(cmd_queue);

        // pull local cells back so the next step can pick its halo by centre x
        if (g_nprocs > 1 && local_cell_no > 0)
        {
            err = clEnqueueReadBuffer(cmd_queue, x_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * MAX_ELE * sizeof(float), x, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, y_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * MAX_ELE * sizeof(float), y, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, z_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * MAX_ELE * sizeof(float), z, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * sizeof(float), xc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * sizeof(float), yc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0,
                                      (size_t)local_cell_no * sizeof(float), zc, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        }

        // Dynamic rebalance
#if REBALANCE_FR > 0
        if (g_nprocs > 1 && iter > 0 && iter % REBALANCE_FR == 0)
        {
            double t_rb0 = MPI_Wtime();

            double t_local = (MPI_Wtime() - t_loop0) - (t_comm + g_output_comm) - g_output_io;
            double d_local = t_local - t_local_prev;
            t_local_prev = t_local;

            double tv[REBAL_MAX_PROCS];
            t_comm0 = MPI_Wtime();
            MPI_Allgather(&d_local, 1, MPI_DOUBLE, tv, 1, MPI_DOUBLE, MPI_COMM_WORLD);
            t_comm += MPI_Wtime() - t_comm0;

            // [REBAL] below only prints when a cut moves, which hides the
            // intervals where the controller correctly did nothing
            if (g_rank == 0)
            {
                double sum = 0.0, mx = 0.0;
                for (int r = 0; r < g_nprocs; r++)
                { sum += tv[r]; if (tv[r] > mx) mx = tv[r]; }
                double mean = sum / g_nprocs;
                printf("[RBLOG] iter=%d imb=%.4f cuts=[", iter,
                       (mean > 0.0) ? (mx - mean) / mean : 0.0);
                for (int r = 0; r <= g_nprocs; r++)
                    printf("%d%s", g_xcut[r], (r < g_nprocs) ? "," : "");
                printf("] t=[");
                for (int r = 0; r < g_nprocs; r++)
                    printf("%.4f%s", tv[r], (r < g_nprocs - 1) ? "," : "");
                printf("]\n");
                fflush(stdout);
            }

            int newcut[REBAL_MAX_PROCS + 1];
            for (int r = 0; r <= g_nprocs; r++) newcut[r] = g_xcut[r];

            if (g_rank == 0)
            {
#if REBAL_CONTROLLER == 1
              // returns cuts already legalized the same way as the rule below.
              // iter / REBALANCE_FR is the event index the phase features need
              if (rnn_ctl) {
                rebal_rnn_decide(rnn_ctl, tv, g_xcut, g_nx, g_min_w,
                                 iter / REBALANCE_FR, NULL, newcut);
              } else
#endif
                for (int b = 1; b < g_nprocs; b++)
                {
                    double lo = tv[b - 1], hi = tv[b]; // below / above cut b
                    if (lo <= 0.0 || hi <= 0.0) continue; // no signal yet

                    double mean = 0.5 * (lo + hi);
                    double imb = (lo - hi) / mean;
                    if (fabs(imb) < REBALANCE_TOL) continue; // deadband

                    // size the move from the slower side's average work per slice
                    double w_slow = (imb > 0.0) ? (double)(newcut[b] - newcut[b - 1])
                                                : (double)(newcut[b + 1] - newcut[b]);
                    double slow = (imb > 0.0) ? lo : hi;
                    double raw = REBALANCE_GAIN * w_slow * fabs(lo - hi) / (2.0 * slow);

                    int step = (int)(raw + 0.5);
                    if (step < 1) step = 1; // past the deadband, always move
                    if (step > REBALANCE_MAX_STEP) step = REBALANCE_MAX_STEP;
                    if (imb < 0.0) step = -step; // rank b is the slower one, grow b-1
					
                    int lo_limit = newcut[b - 1] + g_min_w;
                    int hi_limit = newcut[b + 1] - g_min_w;
                    if (lo_limit > hi_limit) continue; // no legal position left

                    int nc = newcut[b] - step;
                    if (nc < lo_limit) nc = lo_limit;
                    if (nc > hi_limit) nc = hi_limit;
                    newcut[b] = nc;
                }
            }
            t_comm0 = MPI_Wtime();
            MPI_Bcast(newcut, g_nprocs + 1, MPI_INT, 0, MPI_COMM_WORLD);
            t_comm += MPI_Wtime() - t_comm0;

            int moved = 0;
            for (int r = 0; r <= g_nprocs; r++) moved += abs(newcut[r] - g_xcut[r]);

            if (moved)
            {
                if (g_rank == 0)
                {
                    printf("[REBAL] iter=%d moved=%d cuts=[", iter, moved);
                    for (int r = 0; r <= g_nprocs; r++)
                        printf("%d%s", newcut[r], (r < g_nprocs) ? "," : "");
                    printf("] compute_s=[");
                    for (int r = 0; r < g_nprocs; r++)
                        printf("%.2f%s", tv[r], (r < g_nprocs - 1) ? "," : "");
                    printf("]\n");
                    fflush(stdout);
                }

                apply_x_cuts(newcut);
                n_rebal++;
                n_slices_moved += moved;
            }
            t_rebal += MPI_Wtime() - t_rb0;
        }
#endif

        // migration: a cell whose centre has left my band is handed to the neighbour.
        // end ranks keep theirs, the outer side being MPI_PROC_NULL
        if (g_nprocs > 1)
        {
            int n_mig_left = 0, n_mig_right = 0;
            for (int i = 0; i < local_cell_no; i++)
            {
                if (ele_per_cell[i] == 0) continue;
                if (xc[i] < g_x_lo && g_x_neighbour_left != MPI_PROC_NULL)
                    mig_left.send_idx[n_mig_left++] = i;
                else if (xc[i] >= g_x_hi && g_x_neighbour_right != MPI_PROC_NULL)
                    mig_right.send_idx[n_mig_right++] = i;
            }

            for (int s = 0; s < n_mig_left; s++)
                mig_pack(&mig_left, s, mig_left.send_idx[s], c, e);
            for (int s = 0; s < n_mig_right; s++)
                mig_pack(&mig_right, s, mig_right.send_idx[s], c, e);

            int n_arr_left = 0, n_arr_right = 0;
            t_comm0 = MPI_Wtime();
            MPI_Sendrecv(&n_mig_left, 1, MPI_INT, g_x_neighbour_left, 6,
                         &n_arr_right, 1, MPI_INT, g_x_neighbour_right, 6,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&n_mig_right, 1, MPI_INT, g_x_neighbour_right, 7,
                         &n_arr_left, 1, MPI_INT, g_x_neighbour_left, 7,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            MPI_Sendrecv(mig_left.send_f, n_mig_left * MIG_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_left, 8,
                         mig_right.recv_f, n_arr_right * MIG_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_right, 8,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(mig_left.send_i, n_mig_left * MIG_INTS_PER_CELL, MPI_INT, g_x_neighbour_left, 9,
                         mig_right.recv_i, n_arr_right * MIG_INTS_PER_CELL, MPI_INT, g_x_neighbour_right, 9,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(mig_right.send_f, n_mig_right * MIG_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_right, 10,
                         mig_left.recv_f, n_arr_left * MIG_FLOATS_PER_CELL, MPI_FLOAT, g_x_neighbour_left, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(mig_right.send_i, n_mig_right * MIG_INTS_PER_CELL, MPI_INT, g_x_neighbour_right, 11,
                         mig_left.recv_i, n_arr_left * MIG_INTS_PER_CELL, MPI_INT, g_x_neighbour_left, 11,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - t_comm0;

            // free the departed slots before placing arrivals, so a slot vacated this
            // step can be refilled this step
            int zero_val = 0;
            for (int s = 0; s < n_mig_left; s++)
            {
                int ci = mig_left.send_idx[s];
                ele_per_cell[ci] = 0;
                err = clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE,
                                           (size_t)ci * sizeof(int), sizeof(int), &zero_val, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }
            for (int s = 0; s < n_mig_right; s++)
            {
                int ci = mig_right.send_idx[s];
                ele_per_cell[ci] = 0;
                err = clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE,
                                           (size_t)ci * sizeof(int), sizeof(int), &zero_val, 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }

            int search_from = 0;
            for (int r = 0; r < n_arr_left + n_arr_right; r++)
            {
                xfer_buf * src = (r < n_arr_left) ? &mig_left : &mig_right;
                int rr = (r < n_arr_left) ? r : r - n_arr_left;

                // reuse a dead slot if there is one, else extend the local range
                int slot = -1;
                for (int i = search_from; i < local_cell_no; i++)
                {
                    if (ele_per_cell[i] == 0) { slot = i; break; }
                }
                if (slot == -1)
                {
                    slot = local_cell_no;
                    local_cell_no++;
                    assert(local_cell_no <= g_max_cell);
                }
                search_from = slot + 1;

                mig_unpack(src, rr, slot, c, e);

                size_t ele_off_f = (size_t)slot * MAX_ELE * sizeof(float);
                size_t ele_cnt_f = (size_t)MAX_ELE * sizeof(float);
                size_t ele_off_i = (size_t)slot * MAX_ELE * sizeof(int);
                size_t ele_cnt_i = (size_t)MAX_ELE * sizeof(int);
                size_t cell_off_f = (size_t)slot * sizeof(float);
                size_t cell_off_i = (size_t)slot * sizeof(int);

                err = clEnqueueWriteBuffer(cmd_queue, x_mem, CL_TRUE, ele_off_f, ele_cnt_f,
                                           x + slot * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, y_mem, CL_TRUE, ele_off_f, ele_cnt_f,
                                           y + slot * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, z_mem, CL_TRUE, ele_off_f, ele_cnt_f,
                                           z + slot * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, id_mem, CL_TRUE, ele_off_i, ele_cnt_i,
                                           id + slot * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, ele_type_mem, CL_TRUE, ele_off_i, ele_cnt_i,
                                           ele_type + slot * MAX_ELE, 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, xc_mem, CL_TRUE, cell_off_f, sizeof(float),
                                           &xc[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, yc_mem, CL_TRUE, cell_off_f, sizeof(float),
                                           &yc[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, zc_mem, CL_TRUE, cell_off_f, sizeof(float),
                                           &zc[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, o1_mem, CL_TRUE, cell_off_f, sizeof(float),
                                           &OVOL1[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, cell_off_i, sizeof(int),
                                           &ele_per_cell[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, cell_off_i, sizeof(int),
                                           &cell_type[slot], 0, NULL, NULL);
                err |= clEnqueueWriteBuffer(cmd_queue, flag_gener_mem, CL_TRUE, cell_off_i, sizeof(int),
                                           &flag_gener[slot], 0, NULL, NULL);
                assert(err == CL_SUCCESS);
            }
        }

#if 0
        // chem field halo. off because nothing writes chem1/chem2: chem_rk_gpu and the
        // initial-field fill are both still #if 0, so this would only shuffle zeros.
        // growth and division are now live but neither touches the chem field. turning
        // it on first needs the 3D chem field partitioned in X, one slab per rank;
        // right now every rank holds the whole field.
        {
            int nyz = g_ny * g_nz;
            float * send_lo = (float *)malloc(nyz * sizeof(float));
            float * send_hi = (float *)malloc(nyz * sizeof(float));
            float * recv_lo = (float *)malloc(nyz * sizeof(float));
            float * recv_hi = (float *)malloc(nyz * sizeof(float));

            // straight off the cut array. deriving these from the float band edges
            // instead used to go negative on rank 0, whose g_x_lo is clamped to 0
            // while g_chem_ox sits at pad_x - CHEM_MARGIN
            int i_lo = g_x_imin;
            int i_hi = g_x_imax - 1;

            for (int jy = 0; jy < g_ny; jy++)
                for (int kz = 0; kz < g_nz; kz++)
                {
                    send_lo[jy * g_nz + kz] = chem1_gpu[(i_lo * g_ny + jy) * g_nz + kz];
                    send_hi[jy * g_nz + kz] = chem1_gpu[(i_hi * g_ny + jy) * g_nz + kz];
                }

            t_comm0 = MPI_Wtime();
            MPI_Sendrecv(send_lo, nyz, MPI_FLOAT, g_x_neighbour_left, 12,
                         recv_hi, nyz, MPI_FLOAT, g_x_neighbour_right, 12,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(send_hi, nyz, MPI_FLOAT, g_x_neighbour_right, 13,
                         recv_lo, nyz, MPI_FLOAT, g_x_neighbour_left, 13,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            t_comm += MPI_Wtime() - t_comm0;

            free(send_lo);
            free(send_hi);
            free(recv_lo);
            free(recv_hi);
        }
#endif

        // start of output
        if(iter%OUTPUT_FR == 0)
        {
            err = clEnqueueReadBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  x, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  y, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  z, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                                  xc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                                  yc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0, local_cell_no*sizeof(float),
                                  zc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, id_mem, CL_TRUE, 0, max_ele_no*sizeof(int), 
								  id, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, max_ele_no*sizeof(int), 
								  ele_type, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, g_max_cell*sizeof(int), 
								  ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float), 
								  chem1_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, g_nx*g_ny*g_nz*sizeof(float), 
								  chem2_gpu, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            // local_cell_no, not combined: ghosts belong to another rank
            output_mpi(x, y, z, xc, yc, zc, id, cell_type, ele_type, local_cell_no, ele_no, ele_per_cell, iter, chem1_gpu, chem2_gpu, cclock, cycle, flag_gener, vx, vy, vz);
            for(int i = 0; i < local_cell_no; i++)
                for(int j = 0; j < ele_per_cell[i]; j++)
            {
                int k = i*MAX_ELE + j;
                
                assert(!isnan(x[k]) && !isnan(y[k]) && !isnan(z[k]));
            }
        } // end of output

        // count the number of each cell type
        if(iter%OUTPUT_FR == 0)
        {
            int count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0;
            for(int i = 0; i < local_cell_no; i++)
            {
                if(ele_per_cell[i] == 0) continue;
                count0++;
                if(cell_type[i] == 1 && ele_per_cell[i] > 0) count1++;
                if(cell_type[i] == 2 && ele_per_cell[i] > 0) count2++;
                if(cell_type[i] == 3 && ele_per_cell[i] > 0) count3++;
                if(cell_type[i] == 4 && ele_per_cell[i] > 0) count4++;
            }

            // sum across ranks: the tally is system wide, and the stop tests must be
            // reached by every rank or the rest deadlock in the next collective
            int local_counts[4] = {count1, count2, count3, count4};
            int total_counts[4];
            t_comm0 = MPI_Wtime();
            MPI_Allreduce(local_counts, total_counts, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            t_comm += MPI_Wtime() - t_comm0;
            count1 = total_counts[0];
            count2 = total_counts[1];
            count3 = total_counts[2];
            count4 = total_counts[3];

            if(g_rank == 0)
            {
                printf("%6d  %4d %4d %4d %4d\n", iter, count1, count2, count3, count4); fflush(stdout);
            }
            if(count1 == 0)
            {
                break;
            }
            if(count1 >= g_max_cell - 10 || count2 >= g_max_cell - 10 || count3 >= g_max_cell - 10|| count4 >= g_max_cell - 10)
            {
                break;
            }
        }
    }

    double t_loop_end = MPI_Wtime();

#if REBALANCE_FR > 0
    if (g_rank == 0 && g_nprocs > 1)
        printf("[REBAL] %d rebalance(s), %lld slice(s) moved, %.3f s in the rebalance path\n",
               n_rebal, n_slices_moved, t_rebal);
#if REBALANCE_FR > 0 && REBAL_CONTROLLER == 1
    rebal_rnn_destroy(rnn_ctl);
#endif
#endif

    *out_setup = t_loop0 - t_func_start;
    *out_loop = t_loop_end - t_loop0;
    *out_comm = t_comm + g_output_comm;
    *out_io = g_output_io;
    xfer_free(&halo_left);
    xfer_free(&halo_right);
    xfer_free(&mig_left);
    xfer_free(&mig_right);
	}
	
#pragma mark Teardown
	{
    	ocl_free_buffers(&bufs);
    	ocl_release(&ocl);
	}
	return CL_SUCCESS;
}
