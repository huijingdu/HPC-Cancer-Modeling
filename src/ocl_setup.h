#ifndef OCL_SETUP_H
#define OCL_SETUP_H

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "sim_types.h"

// handles all of the programs
enum { P_MOVEMENT, P_GROWTH, P_CELLCENTER, P_CHEM_RK, P_OVOL, P_BINNING, P_COUNT };

// handles all of the kernels
enum { K_MOVEMENT, K_GROWTH, K_CELLCENTER, K_CHEM_RK, K_OVOL,
       K_BIN_COUNT, K_BIN_BLOCK_SCAN, K_BIN_SCATTER, K_CELL_BOUNDS,
       K_BIN_SCAN_BLOCK_SUMS, K_BIN_ADD_BLOCK_OFFSETS, K_COUNT };

// manages variables associated with OpenCL
typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program[P_COUNT];
    cl_kernel kernel[K_COUNT];
    size_t scan_wg;
} ocl_ctx;

// manages memory for OpenCL buffers
typedef struct {
    cl_mem x_mem, y_mem, z_mem;
    cl_mem xc_mem, yc_mem, zc_mem;
    cl_mem id_mem, cell_type_mem, ele_type_mem;
    cl_mem xF_mem, yF_mem, zF_mem;
    cl_mem ele_per_cell_mem;
    cl_mem flag_mem, flag_death_mem, flag_gener_mem;
    cl_mem chem1_gpu_mem, chem2_gpu_mem;
    cl_mem chem11_gpu_mem, chem22_gpu_mem;
    cl_mem grid_gpu_mem, grid1_gpu_mem, grid2_gpu_mem, grid3_gpu_mem, grid4_gpu_mem;
    cl_mem rng_x_mem, rng_c_mem;
    cl_mem o1_mem, o2_mem;
    cl_mem vx_mem, vy_mem, vz_mem;
    cl_mem xo_mem, yo_mem, zo_mem;

    cl_mem bin_count_mem, bin_offset_mem, cell_list_mem;
    cl_mem bin_bounds_mem, bin_overflow_mem;
    cl_mem block_sums_mem; // per-group totals for the scan, plus the grand total
} ocl_bufs;

// initializes OpenCL for available hardware
void ocl_init(ocl_ctx * ocl);

// configures OpenCL kernels
void ocl_build_kernels(ocl_ctx * ocl);

// allocates memory for buffers
void ocl_alloc_buffers(ocl_bufs * b, const ocl_ctx * ocl,
                       const ele_arrays * e, const cell_arrays * c, const chem_grid * ch,
                       int max_ele_no, int scan_nblocks);

// sets arguments for bin grid kernels                       
void ocl_set_bin_grid_args(const ocl_ctx * ocl, int scan_nblocks);

// sets arguments for all kernels
void ocl_set_args(const ocl_ctx * ocl, const ocl_bufs * b,
                  int cell_no, int ele_no, float dt, int scan_nblocks);

// regrows the bins on call
void ocl_regrow_bins(const ocl_ctx * ocl, ocl_bufs * b, int scan_nblocks);

// frees memory in the buffers
void ocl_free_buffers(ocl_bufs * b);

// releases all associated memory with OpenCL
void ocl_release(ocl_ctx * ocl);

#endif // OCL_SETUP_H
