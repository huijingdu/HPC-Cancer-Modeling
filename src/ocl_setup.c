#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/stat.h>

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

#include "ocl_setup.h"
#include "sim_globals.h"
#include "mpi_assert.h"
#include "io.h"

void ocl_init(ocl_ctx * ocl) {
    cl_int err = 0;
    size_t returned_size = 0;
    cl_context context;
    cl_command_queue cmd_queue;

    cl_device_id device_id = NULL; // OpenCL device

    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, NULL, &nplat);
    assert(err == CL_SUCCESS);
    if (g_rank == 0) printf("# of Platforms = %u\n", nplat);

    cl_platform_id *plats = malloc(sizeof(cl_platform_id) * nplat);
    clGetPlatformIDs(nplat, plats, NULL);

    const cl_device_type wanted = CL_DEVICE_TYPE_GPU; // The type of requested device

    // rank to GPU binding: prefer the launcher's node local rank so that two ranks
    // sharing a node land on different devices, and fall back to the global rank
    int local_rank = g_rank;
    {
        const char * env_local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (!env_local_rank) env_local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK");
        if (!env_local_rank) env_local_rank = getenv("SLURM_LOCALID");
        if (env_local_rank) local_rank = atoi(env_local_rank);
    }

    int found = 0;
    for (cl_uint i = 0; i < nplat && !found; ++i) {
        char pname[128] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        if (g_rank == 0) printf("Platform %u: %s\n", i, pname);

        cl_uint ndev = 0;
        if (clGetDeviceIDs(plats[i], wanted, 0, NULL, &ndev) != CL_SUCCESS || ndev == 0)
            continue;

        cl_device_id * devs = malloc(sizeof(cl_device_id) * ndev);
        if (clGetDeviceIDs(plats[i], wanted, ndev, devs, NULL) == CL_SUCCESS) {
            device_id = devs[local_rank % ndev];
            found = 1;
            if (g_rank == 0) printf("[rank %d] using device %d of %u\n", g_rank, local_rank % ndev, ndev);
        }
        free(devs);
    }
    free(plats);
    assert(found);

    if (g_rank == 0) printf("err = %d\n", err);
    assert(err == CL_SUCCESS);

#pragma mark Device Information
	{
    	cl_char vendor_name[1024] = {0};
    	cl_char device_name[1024] = {0};
    	cl_char buf[1024] = {0};
    	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), 
       						  vendor_name, &returned_size);
    	err |= clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), 
							  device_name, &returned_size);
        err |= clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(buf), 
							  buf, &returned_size);
    	assert(err == CL_SUCCESS);
        if (g_rank == 0) {
            printf("Connecting to %s %s supporting ", vendor_name, device_name);
            printf("%s...\n", buf);
        }
    }
	
#pragma mark Context and Command Queue
	{
    	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    	assert(err == CL_SUCCESS);
    	cmd_queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);
	}

    ocl->device = device_id;
    ocl->context = context;
    ocl->queue = cmd_queue;
}

static char * load_program_source(const char *filename) { 
	
	struct stat statbuf;
	FILE *fh; 
	char *source; 
	int flag_return;

	fh = fopen(filename, "r");
	if (fh == 0)
		return 0; 
	
	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	flag_return = fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0'; 
    fclose(fh);
    
	return source; 
} 

static cl_program build_program(ocl_ctx * ocl, const char * filename) {
    cl_int err;
    cl_program program;

    char * program_source = load_program_source(filename);

    program = clCreateProgramWithSource(ocl->context, 1, (const char**)&program_source,
                                            NULL, &err);
    assert(err == CL_SUCCESS);

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    free(program_source);

    char build[2048] = {0};
    clGetProgramBuildInfo(program, ocl->device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "[rank %d] %s build failed (%d):\n%s\n", g_rank, filename, err, build);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    return program;
}

static cl_kernel build_kernel(cl_program program, const char * name) {
    cl_int err;

    cl_kernel kernel = clCreateKernel(program, name, &err);
    assert(err == CL_SUCCESS);

    return kernel;
}

void ocl_build_kernels(ocl_ctx * ocl) {
    cl_int err = 0;
    cl_device_id device_id = ocl->device;
    cl_program * program = ocl->program;
    cl_kernel * kernel = ocl->kernel;
    size_t scan_wg = 0;

    // create program for cell movement
    program[P_MOVEMENT] = build_program(ocl, "movement.cl");
    kernel[K_MOVEMENT] = build_kernel(program[P_MOVEMENT], "movement");

    // create program for cell growth
    program[P_GROWTH] = build_program(ocl, "growth.cl");
    kernel[K_GROWTH] = build_kernel(program[P_GROWTH], "growth");

    // create program to update cell center position
    program[P_CELLCENTER] = build_program(ocl, "cellcenter.cl");
    kernel[K_CELLCENTER] = build_kernel(program[P_CELLCENTER], "cellcenter");

    // create program to evolve chemical field
    program[P_CHEM_RK] = build_program(ocl, "chem_rk_gpu.cl");
    kernel[K_CHEM_RK] = build_kernel(program[P_CHEM_RK], "chem_rk_gpu");

    // create program to evolve intracellular gene regulatory network
    program[P_OVOL] = build_program(ocl, "ovol_fun.cl");
    kernel[K_OVOL] = build_kernel(program[P_OVOL], "ovol_fun");

    // create program for binning logic
    program[P_BINNING] = build_program(ocl, "binning.cl");

    kernel[K_BIN_COUNT] = build_kernel(program[P_BINNING], "bin_count");
    kernel[K_BIN_BLOCK_SCAN] = build_kernel(program[P_BINNING], "bin_block_scan");
    kernel[K_BIN_SCATTER] = build_kernel(program[P_BINNING], "bin_scatter");
    kernel[K_CELL_BOUNDS] = build_kernel(program[P_BINNING], "cell_bounds");
    kernel[K_BIN_SCAN_BLOCK_SUMS] = build_kernel(program[P_BINNING], "bin_scan_block_sums");
    kernel[K_BIN_ADD_BLOCK_OFFSETS] = build_kernel(program[P_BINNING], "bin_add_block_offsets");

    // largest power-of-two group the device and all three scan kernels allow
    size_t dev_max = 0;
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                            sizeof(dev_max), &dev_max, NULL);
    assert(err == CL_SUCCESS);

    const int scan_kernels[3] = {K_BIN_BLOCK_SCAN, K_BIN_SCAN_BLOCK_SUMS, K_BIN_ADD_BLOCK_OFFSETS};
    scan_wg = dev_max < 256 ? dev_max : 256;
    for (int k = 0; k < 3; k++)
    {
        size_t kmax = 0;
        err = clGetKernelWorkGroupInfo(kernel[scan_kernels[k]], device_id,
                                        CL_KERNEL_WORK_GROUP_SIZE, sizeof(kmax), &kmax, NULL);
        assert(err == CL_SUCCESS);
        if (kmax < scan_wg) scan_wg = kmax;
    }

    size_t pow2 = 1;
    while (pow2 * 2 <= scan_wg) pow2 *= 2;
    scan_wg = pow2;
    assert(scan_wg >= 1);

    ocl->scan_wg = scan_wg;

}

void ocl_alloc_buffers(ocl_bufs * b, const ocl_ctx * ocl,
                       const ele_arrays * e, const cell_arrays * c, const chem_grid * ch,
                       int max_ele_no, int scan_nblocks) {
    cl_int err = 0;
    size_t buffer_size;
    cl_context context = ocl->context;
    cl_command_queue cmd_queue = ocl->queue;

    // Bin buffer allocation (requires context)
    b->bin_count_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, g_nbins * sizeof(int), NULL, &err);
    b->bin_offset_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (g_nbins + 1) * sizeof(int), NULL, &err);
    b->cell_list_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, g_max_cell * sizeof(int), NULL, &err);
    // occupied extent and out-of-grid tally, read back once per rebuild
    b->bin_bounds_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 6 * sizeof(int), NULL, &err);
    b->bin_overflow_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);

    b->block_sums_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    (scan_nblocks + 1) * sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);

    buffer_size = sizeof(float) * max_ele_no;
    
    b->x_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->x_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->x, 0, NULL, NULL);
    b->y_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->y_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->y, 0, NULL, NULL);
    b->z_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->z_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->z, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    b->xF_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->xF_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->xF, 0, NULL, NULL);
    b->yF_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->yF_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->yF, 0, NULL, NULL);
    b->zF_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->zF_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->zF, 0, NULL, NULL);
    b->xo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->xo_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->xo, 0, NULL, NULL);
    b->yo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->yo_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->yo, 0, NULL, NULL);
    b->zo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->zo_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->zo, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    buffer_size = sizeof(int) * max_ele_no;
    b->id_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->id_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->id, 0, NULL, NULL);
    b->ele_type_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->ele_type_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->ele_type, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    
    buffer_size = sizeof(int) * g_max_cell;
    b->cell_type_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->cell_type_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->cell_type, 0, NULL, NULL);
    b->ele_per_cell_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->ele_per_cell_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->ele_per_cell, 0, NULL, NULL);
    b->flag_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->flag_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->flag, 0, NULL, NULL);
    b->flag_death_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->flag_death_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->flag_death, 0, NULL, NULL);
    b->flag_gener_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->flag_gener_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->flag_gener, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    buffer_size = sizeof(float) * g_max_cell;
    b->xc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->xc_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->xc, 0, NULL, NULL);
    b->yc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->yc_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->yc, 0, NULL, NULL);
    b->zc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->zc_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->zc, 0, NULL, NULL);
    b->o1_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->o1_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->OVOL1, 0, NULL, NULL);
    b->o2_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->o2_mem, CL_TRUE, 0, buffer_size,
                                (void*)c->OVOL2, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    buffer_size = sizeof(float) * g_nx*g_ny*g_nz;
    b->chem1_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->chem1_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->chem1_gpu, 0, NULL, NULL);
    b->chem2_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->chem2_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->chem2_gpu, 0, NULL, NULL);
    b->chem11_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->chem11_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->chem11_gpu, 0, NULL, NULL);
    b->chem22_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->chem22_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->chem22_gpu, 0, NULL, NULL);
    b->vx_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->vx_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->vx, 0, NULL, NULL);
    b->vy_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->vy_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->vy, 0, NULL, NULL);
    b->vz_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->vz_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->vz, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    buffer_size = sizeof(int) * g_nx*g_ny*g_nz;
    b->grid_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->grid_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->grid_gpu, 0, NULL, NULL);
    b->grid1_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->grid1_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->grid1_gpu, 0, NULL, NULL);
    b->grid2_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->grid2_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->grid2_gpu, 0, NULL, NULL);
    b->grid3_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->grid3_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->grid3_gpu, 0, NULL, NULL);
    b->grid4_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->grid4_gpu_mem, CL_TRUE, 0, buffer_size,
                                (void*)ch->grid4_gpu, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    buffer_size = max_ele_no*sizeof(unsigned int);
    b->rng_x_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err = clEnqueueWriteBuffer(cmd_queue, b->rng_x_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->rng_x, 0, NULL, NULL);
    b->rng_c_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, b->rng_c_mem, CL_TRUE, 0, buffer_size,
                                (void*)e->rng_c, 0, NULL, NULL);
    assert(err == CL_SUCCESS);

    clFinish(cmd_queue);
}

void ocl_set_bin_grid_args(const ocl_ctx * ocl, int scan_nblocks) {
    cl_int err = 0;
    
    // the grid can shrink too, so the counts refresh on every refit
    err = clSetKernelArg(ocl->kernel[K_BIN_BLOCK_SCAN], 1, sizeof(int), &g_nbins);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCAN_BLOCK_SUMS], 1, sizeof(int), &scan_nblocks);
    err |= clSetKernelArg(ocl->kernel[K_BIN_ADD_BLOCK_OFFSETS], 1, sizeof(int), &g_nbins);
    err |= clSetKernelArg(ocl->kernel[K_BIN_ADD_BLOCK_OFFSETS], 4, sizeof(int), &scan_nblocks);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 6, sizeof(int), &g_nbx);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 7, sizeof(int), &g_nby);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 8, sizeof(int), &g_nbz);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 10, sizeof(float), &g_bin_ox);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 11, sizeof(float), &g_bin_oy);
    err |= clSetKernelArg(ocl->kernel[K_BIN_COUNT], 12, sizeof(float), &g_bin_oz);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 6, sizeof(int), &g_nbx);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 7, sizeof(int), &g_nby);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 8, sizeof(int), &g_nbz);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 11, sizeof(float), &g_bin_ox);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 12, sizeof(float), &g_bin_oy);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 13, sizeof(float), &g_bin_oz);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 24, sizeof(int), &g_nbx);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 25, sizeof(int), &g_nby);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 26, sizeof(int), &g_nbz);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 27, sizeof(float), &g_bin_ox);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 28, sizeof(float), &g_bin_oy);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 29, sizeof(float), &g_bin_oz);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 30, sizeof(float), &g_sub_z);
    assert(err == CL_SUCCESS);
}

void ocl_set_args(const ocl_ctx * ocl, const ocl_bufs * b, int cell_no, int ele_no, float dt, int scan_nblocks) {
    cl_int err = 0;
    const cl_kernel * kernel = ocl->kernel;
    size_t scan_wg = ocl->scan_wg;

    #pragma mark Kernel Arguments
	{
        int max_ele = MAX_ELE;
	    // Now setup the arguments to kernel "movement"
    	err  = clSetKernelArg(kernel[K_MOVEMENT],  0, sizeof(cl_mem), &b->id_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  1, sizeof(cl_mem), &b->x_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  2, sizeof(cl_mem), &b->y_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  3, sizeof(cl_mem), &b->z_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  4, sizeof(cl_mem), &b->ele_type_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  5, sizeof(int), &cell_no);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  6, sizeof(int), &ele_no);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  7, sizeof(cl_mem), &b->ele_per_cell_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  8, sizeof(cl_mem), &b->xc_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT],  9, sizeof(cl_mem), &b->yc_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 10, sizeof(cl_mem), &b->zc_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 11, sizeof(float), &dt);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 12, sizeof(cl_mem), &b->xF_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 13, sizeof(cl_mem), &b->yF_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 14, sizeof(cl_mem), &b->zF_mem);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 15, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 16, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 17, sizeof(float), &g_lz);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 18, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[K_MOVEMENT], 19, sizeof(cl_mem), &b->cell_type_mem);
        err |= clSetKernelArg(kernel[K_MOVEMENT], 20, sizeof(cl_mem), &b->flag_gener_mem);
    	assert(err == CL_SUCCESS);

	}
	
	{
        int max_ele = MAX_ELE;
    	// Now setup the arguments to kernel "growth"
    	err  = clSetKernelArg(kernel[K_GROWTH],  0, sizeof(cl_mem), &b->id_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  1, sizeof(cl_mem), &b->x_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  2, sizeof(cl_mem), &b->y_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  3, sizeof(cl_mem), &b->z_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  4, sizeof(cl_mem), &b->ele_type_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  5, sizeof(int), &ele_no);
    	err |= clSetKernelArg(kernel[K_GROWTH],  6, sizeof(cl_mem), &b->flag_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  7, sizeof(cl_mem), &b->ele_per_cell_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH],  8, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[K_GROWTH],  9, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[K_GROWTH], 10, sizeof(float), &g_lz);
    	err |= clSetKernelArg(kernel[K_GROWTH], 11, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[K_GROWTH], 12, sizeof(cl_mem), &b->cell_type_mem);
    	err |= clSetKernelArg(kernel[K_GROWTH], 13, sizeof(cl_mem), &b->flag_death_mem);
    	assert(err == CL_SUCCESS);
	}
	
    {
        int max_ele = MAX_ELE;
        // Now setup the arguments to kernel "cellcenter"
        err  = clSetKernelArg(kernel[K_CELLCENTER],  0, sizeof(cl_mem), &b->id_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  1, sizeof(cl_mem), &b->x_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  2, sizeof(cl_mem), &b->y_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  3, sizeof(cl_mem), &b->z_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  4, sizeof(cl_mem), &b->ele_per_cell_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  5, sizeof(cl_mem), &b->xc_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  6, sizeof(cl_mem), &b->yc_mem);
        err |= clSetKernelArg(kernel[K_CELLCENTER],  7, sizeof(cl_mem), &b->zc_mem);
    	err |= clSetKernelArg(kernel[K_CELLCENTER],  8, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[K_CELLCENTER],  9, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[K_CELLCENTER], 10, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[K_CELLCENTER], 11, sizeof(float), &g_lz);
        assert(err == CL_SUCCESS);
    }

    {
        int nx0 = g_nx, ny0 = g_ny, nz0 = g_nz, tmpii = 0;
        float dx0 = g_dx, dy0 = g_dy, dz0 = g_dz;
        float dt_chem = 1.0*dt;
        // Now setup the arguments to kernel "chem_rk_gpu"
        err  = clSetKernelArg(kernel[K_CHEM_RK],  0, sizeof(cl_mem), &b->chem1_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  1, sizeof(cl_mem), &b->grid_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  2, sizeof(int), &nx0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  3, sizeof(int), &ny0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  4, sizeof(int), &nz0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  5, sizeof(float), &dx0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  6, sizeof(float), &dy0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  7, sizeof(float), &dz0);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  8, sizeof(float), &dt_chem);
        err |= clSetKernelArg(kernel[K_CHEM_RK],  9, sizeof(cl_mem), &b->chem11_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 10, sizeof(cl_mem), &b->grid1_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 11, sizeof(cl_mem), &b->chem2_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 12, sizeof(cl_mem), &b->chem22_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 13, sizeof(cl_mem), &b->grid2_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 14, sizeof(cl_mem), &b->grid3_gpu_mem);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 15, sizeof(int), &tmpii);
        err |= clSetKernelArg(kernel[K_CHEM_RK], 16, sizeof(cl_mem), &b->grid4_gpu_mem);
    	err |= clSetKernelArg(kernel[K_CHEM_RK], 17, sizeof(cl_mem), &b->vx_mem);
    	err |= clSetKernelArg(kernel[K_CHEM_RK], 18, sizeof(cl_mem), &b->vy_mem);
    	err |= clSetKernelArg(kernel[K_CHEM_RK], 19, sizeof(cl_mem), &b->vz_mem);
        assert(err == CL_SUCCESS);
    }


    {
        int max_ele = MAX_ELE;
        // Now setup the arguments to kernel "ovol_fun"
        err  = clSetKernelArg(kernel[K_OVOL],  0, sizeof(cl_mem), &b->id_mem);
        assert(err == CL_SUCCESS);
        err |= clSetKernelArg(kernel[K_OVOL],  1, sizeof(cl_mem), &b->x_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  2, sizeof(cl_mem), &b->y_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  3, sizeof(cl_mem), &b->z_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  4, sizeof(cl_mem), &b->ele_per_cell_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  5, sizeof(cl_mem), &b->chem1_gpu_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  6, sizeof(cl_mem), &b->chem2_gpu_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  7, sizeof(cl_mem), &b->o1_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  8, sizeof(cl_mem), &b->o2_mem);
        err |= clSetKernelArg(kernel[K_OVOL],  9, sizeof(int), &max_ele);
        err |= clSetKernelArg(kernel[K_OVOL], 10, sizeof(cl_mem), &b->cell_type_mem);
        err |= clSetKernelArg(kernel[K_OVOL], 11, sizeof(int), &g_nx);
        err |= clSetKernelArg(kernel[K_OVOL], 12, sizeof(int), &g_ny);
        err |= clSetKernelArg(kernel[K_OVOL], 13, sizeof(int), &g_nz);
        err |= clSetKernelArg(kernel[K_OVOL], 14, sizeof(float), &g_chem_lx);
        err |= clSetKernelArg(kernel[K_OVOL], 15, sizeof(float), &g_chem_ly);
        err |= clSetKernelArg(kernel[K_OVOL], 16, sizeof(float), &g_chem_lz);
        assert(err == CL_SUCCESS);
    }
    
    {
        const float bin_size_f = BIN_SIZE;

        // kernel[K_BIN_COUNT] bin_count
        err = clSetKernelArg(kernel[K_BIN_COUNT], 0, sizeof(cl_mem), &b->xc_mem);
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 1, sizeof(cl_mem), &b->yc_mem);
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 2, sizeof(cl_mem), &b->zc_mem);
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 3, sizeof(cl_mem), &b->ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 5, sizeof(float), &bin_size_f);
       
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 9, sizeof(cl_mem), &b->bin_count_mem);
        
        err |= clSetKernelArg(kernel[K_BIN_COUNT], 13, sizeof(cl_mem), &b->bin_overflow_mem);

        // kernel[K_CELL_BOUNDS] cell_bounds, launched only when the grid needs a refit
        err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 0, sizeof(cl_mem), &b->xc_mem);
        err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 1, sizeof(cl_mem), &b->yc_mem);
        err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 2, sizeof(cl_mem), &b->zc_mem);
        err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 3, sizeof(cl_mem), &b->ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[K_CELL_BOUNDS], 5, sizeof(cl_mem), &b->bin_bounds_mem);

        // K_BIN_BLOCK_SCAN, K_BIN_SCAN_BLOCK_SUMS, K_BIN_ADD_BLOCK_OFFSETS; the three levels of the bin scan
        const size_t scan_scratch = 2 * scan_wg * sizeof(int);
        err |= clSetKernelArg(kernel[K_BIN_BLOCK_SCAN], 0, sizeof(cl_mem), &b->bin_count_mem);

        err |= clSetKernelArg(kernel[K_BIN_BLOCK_SCAN], 2, sizeof(cl_mem), &b->bin_offset_mem);
        err |= clSetKernelArg(kernel[K_BIN_BLOCK_SCAN], 3, sizeof(cl_mem), &b->block_sums_mem);
        err |= clSetKernelArg(kernel[K_BIN_BLOCK_SCAN], 4, scan_scratch, NULL);

        err |= clSetKernelArg(kernel[K_BIN_SCAN_BLOCK_SUMS], 0, sizeof(cl_mem), &b->block_sums_mem);

        err |= clSetKernelArg(kernel[K_BIN_SCAN_BLOCK_SUMS], 2, scan_scratch, NULL);

        err |= clSetKernelArg(kernel[K_BIN_ADD_BLOCK_OFFSETS], 0, sizeof(cl_mem), &b->bin_count_mem);

        err |= clSetKernelArg(kernel[K_BIN_ADD_BLOCK_OFFSETS], 2, sizeof(cl_mem), &b->bin_offset_mem);
        err |= clSetKernelArg(kernel[K_BIN_ADD_BLOCK_OFFSETS], 3, sizeof(cl_mem), &b->block_sums_mem);

        // kernel[K_BIN_SCATTER] bin_scatter
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 0, sizeof(cl_mem), &b->xc_mem);
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 1, sizeof(cl_mem), &b->yc_mem);
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 2, sizeof(cl_mem), &b->zc_mem);
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 3, sizeof(cl_mem), &b->ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 5, sizeof(float), &bin_size_f);
        
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 9, sizeof(cl_mem), &b->bin_count_mem);
        err |= clSetKernelArg(kernel[K_BIN_SCATTER], 10, sizeof(cl_mem), &b->cell_list_mem);
        

        assert(err == CL_SUCCESS);

        // Bin related arguments
        err = clSetKernelArg(kernel[K_MOVEMENT], 21, sizeof(cl_mem), &b->cell_list_mem);
        err |= clSetKernelArg(kernel[K_MOVEMENT], 22, sizeof(cl_mem), &b->bin_offset_mem);
        err |= clSetKernelArg(kernel[K_MOVEMENT], 23, sizeof(float), &bin_size_f);
        
        assert(err == CL_SUCCESS);

        ocl_set_bin_grid_args(ocl, scan_nblocks);
    }
}

void ocl_regrow_bins(const ocl_ctx * ocl, ocl_bufs * b, int scan_nblocks) {
    cl_int err = 0;
    cl_context context = ocl->context;

    clReleaseMemObject(b->bin_count_mem);
    clReleaseMemObject(b->bin_offset_mem);
    clReleaseMemObject(b->block_sums_mem);
    b->bin_count_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    g_nbins * sizeof(int), NULL, &err);
    b->bin_offset_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    (g_nbins + 1) * sizeof(int), NULL, &err);
    b->block_sums_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    (scan_nblocks + 1) * sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);

    err = clSetKernelArg(ocl->kernel[K_BIN_COUNT], 9, sizeof(cl_mem), &b->bin_count_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_BLOCK_SCAN], 0, sizeof(cl_mem), &b->bin_count_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_BLOCK_SCAN], 2, sizeof(cl_mem), &b->bin_offset_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_BLOCK_SCAN], 3, sizeof(cl_mem), &b->block_sums_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCAN_BLOCK_SUMS], 0, sizeof(cl_mem), &b->block_sums_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_ADD_BLOCK_OFFSETS], 0, sizeof(cl_mem), &b->bin_count_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_ADD_BLOCK_OFFSETS], 2, sizeof(cl_mem), &b->bin_offset_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_ADD_BLOCK_OFFSETS], 3, sizeof(cl_mem), &b->block_sums_mem);
    err |= clSetKernelArg(ocl->kernel[K_BIN_SCATTER], 9, sizeof(cl_mem), &b->bin_count_mem);
    err |= clSetKernelArg(ocl->kernel[K_MOVEMENT], 22, sizeof(cl_mem), &b->bin_offset_mem);
    assert(err == CL_SUCCESS);
}

void ocl_free_buffers(ocl_bufs * b) {
    clReleaseMemObject(b->x_mem);
    clReleaseMemObject(b->y_mem);
    clReleaseMemObject(b->z_mem);
    clReleaseMemObject(b->xc_mem);
    clReleaseMemObject(b->yc_mem);
    clReleaseMemObject(b->zc_mem);
    clReleaseMemObject(b->id_mem);
    clReleaseMemObject(b->cell_type_mem);
    clReleaseMemObject(b->ele_type_mem);
    clReleaseMemObject(b->xF_mem);
    clReleaseMemObject(b->yF_mem);
    clReleaseMemObject(b->zF_mem);
    clReleaseMemObject(b->ele_per_cell_mem);
    clReleaseMemObject(b->flag_mem);
    clReleaseMemObject(b->flag_death_mem);
    clReleaseMemObject(b->flag_gener_mem);
    clReleaseMemObject(b->chem1_gpu_mem);
    clReleaseMemObject(b->chem2_gpu_mem);
    clReleaseMemObject(b->chem11_gpu_mem);
    clReleaseMemObject(b->chem22_gpu_mem);
    clReleaseMemObject(b->grid_gpu_mem);
    clReleaseMemObject(b->grid1_gpu_mem);
    clReleaseMemObject(b->grid2_gpu_mem);
    clReleaseMemObject(b->grid3_gpu_mem);
    clReleaseMemObject(b->grid4_gpu_mem);
    clReleaseMemObject(b->rng_x_mem);
    clReleaseMemObject(b->rng_c_mem);
    clReleaseMemObject(b->o1_mem);
    clReleaseMemObject(b->o2_mem);
    clReleaseMemObject(b->vx_mem);
    clReleaseMemObject(b->vy_mem);
    clReleaseMemObject(b->vz_mem);
    clReleaseMemObject(b->xo_mem);
    clReleaseMemObject(b->yo_mem);
    clReleaseMemObject(b->zo_mem);
    clReleaseMemObject(b->bin_count_mem);
    clReleaseMemObject(b->bin_offset_mem);
    clReleaseMemObject(b->cell_list_mem);
    clReleaseMemObject(b->bin_bounds_mem);
    clReleaseMemObject(b->bin_overflow_mem);
    clReleaseMemObject(b->block_sums_mem);
}

void ocl_release(ocl_ctx * ocl) {

    for (int i = 0; i < K_COUNT; ++i) {
        if (ocl->kernel[i]) clReleaseKernel(ocl->kernel[i]);
    }

    for (int i = 0; i < P_COUNT; ++i) {
        if (ocl->program[i]) clReleaseProgram(ocl->program[i]);
    }

    clReleaseCommandQueue(ocl->queue);
    clReleaseContext(ocl->context);
}
