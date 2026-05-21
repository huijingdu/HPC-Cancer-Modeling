#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <math.h>

#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(p) _mkdir(p)
#else
    #define MKDIR(p) mkdir((p), S_IRWXU)
#endif

#define CL_TARGET_OPENCL_VERSION 300
#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #include <CL/opencl.h>
#endif

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

#define BIN_SIZE 12.0f
#define NBX 4
#define NBY 4
#define NBZ 7
#define NBINS (NBX * NBY * NBZ)

#define BC 0
#define RATIO 1
#define NX 40
#define NY NX
#define NZ 80
#define NEQ  NX*NY*NZ          /* problem dimension */

#define LX 40.0
#define LY LX
#define LZ 80.0
#define DX LX/NX
#define DY DX
#define DZ DX

#define GROWTH_FR 1000
#define DIVISION_FR 1000
#define OUTPUT_FR 10000

#define MAX_ITERATIONS 400001
#define MAX_CHEM_ITER  1000
#define MAX_CELL 2000
#define MAX_ELE 20
#define MAX_DEATH 60000

#define C1_MEAN 180.0 // slow-dividing doubles in 180 days
#define C1_STD 40.0 // standard deviation
#define C2_MEAN 4.0 // fast-dividing doubles in 4 days
#define C2_STD 1.0 // standard deviation
#define MAX_GENER 2

static const int nx = NX;
static const int ny = NY;
static const int nz = NZ;
static const float lx = LX;
static const float ly = LY;
static const float lz = LZ;

static const float R_Diff = 0.01; // probability that a slow-dividing cell gives birth to a fast-divising daughter

float ***chem1, ***chem2, ***chem_diff;
int chem_kinsol(int);

int ***grid, ***grid1, ***grid2, ***grid3, ***grid4;
void  initial_chem_paras(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell);
void  cell_in_block(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell, float * xo, float * yo, float * zo, float * vx, float * vy, float * vz);
void  cell_lineage(int id, float x, float y, float z, int ctype0, int * ctype1, int * ctype2, float * chem1_gpu, float * chem2_gpu, float * OVOL1, float * OVOL2, int flag_gener);
int count0, count1, count2, count3, count4;
float *OVOL1, *OVOL2;

// Random Number Generators
float RNG()
{
    float random;
    random=(1.0*(rand()%RAND_MAX))/(1.0*RAND_MAX);
    return (random);
}


// Reader for initial cells' location
// From file Input/IC
void InitialReader(const char InName[], float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int * cell_no, int * ele_no, int * ele_per_cell, int * cclock, int * cycle)
{
    FILE *parIn;
    int i, j;
    float px, py, pz, pxc, pyc, pzc;
    int pid, pcell_type, tmp, ptime, pcycle, pele_per_cell;
    int flag_return;
    int tmp_cell_no;

    parIn = fopen(InName , "r");

    if (parIn == NULL)
    {
       exit(1);
    }

    flag_return = fscanf(parIn, "%d", &tmp_cell_no);
    (*ele_no) = 0;
    (*cell_no) = 0;
    int actual_cell_no = 0;
    for(i = 0; i < tmp_cell_no; i++)
    {
        flag_return = fscanf(parIn, "%d %d  %d  %f  %f  %f  %d %d\n",
                &pid, &pele_per_cell, &pcell_type, &pxc, &pyc, &pzc, &ptime, &pcycle);
        if(pele_per_cell <= 0) continue;

        ele_per_cell[actual_cell_no] = pele_per_cell;
        cell_type[actual_cell_no] = pcell_type;
        cycle[actual_cell_no] = (int)(1000.0f*(C1_MEAN + RNG()*C1_STD));
        cclock[actual_cell_no] = (int)(1.0*cycle[actual_cell_no]*RNG());

      for(j = actual_cell_no*MAX_ELE+0; j < actual_cell_no*MAX_ELE+pele_per_cell; j++)
      {
        flag_return = fscanf(parIn, "%f %f %f", &px, &py, &pz);
        x[j] = px;
        y[j] = py;
        z[j] = pz;
        id[j] = actual_cell_no+1;
        ele_type[j] = 0;
        if(cell_type[actual_cell_no] == 1 && (j-actual_cell_no*MAX_ELE) <= 2)
            ele_type[j] = 2;
        (*ele_no)++;
      }
        actual_cell_no++;
    }
    (*cell_no) = actual_cell_no;
    fclose(parIn);
}

// Output PQR files for visual in VMD
void output(float * x, float * y, float * z, float * xc, float * yc, float * zc, int * id, int * cell_type, int * ele_type, int cell_no, int ele_no, int * ele_per_cell, int step, float * chem1_gpu, float * chem2_gpu, int * cclock, int * cycle, int * flag_gener, float * vx, float * vy, float * vz)
{
    int i, j, k, check, index;
    char filename[50], outdir[50];
    FILE *fp = NULL;

    int e;
    struct stat st;

    sprintf(outdir, "%s", "PQR");

    e = stat(outdir, &st);

    if(stat(outdir, &st) == -1)
    {
        if(errno == ENOENT)
        {
        check = MKDIR(outdir);
        if(check != 0)
        {
        (void) printf("WARNING in output(), directory "
                     "%s doesn't exist and can't be created\n",outdir);
        }
        else
        {
            printf("created the dirctory %s\n", outdir);
        }
        }
    }

    sprintf(filename, "%s/data.%d.pqr", outdir, step);

    fp = fopen(filename, "w");
    if(fp == NULL)
    {
        printf("Failed to open file for writing\n");
        exit(-1);
    }

    int id_skip = 0;
    int flag_skip = 0;
    for(i=0;i<cell_no;i++)
    {
        if(ele_per_cell[i] == 0) continue;
        flag_skip = 0;
        for(j=0;j<ele_per_cell[i];j++)
        {
            k = i*MAX_ELE+j;

            if(z[k] > LZ+2.0)
                flag_skip = 1;;
        }
        if(flag_skip == 1) continue;

        id_skip = cell_type[i];
        for(j=0;j<ele_per_cell[i];j++)
        {
            k = i*MAX_ELE+j;

            fprintf(fp, "ATOM %6d C  THR   %5d      % .3f % .3f % .3f % .4f % .4f\n",
                    k+1, id[k]+id_skip*1000, x[k], y[k], z[k], 0.1*ele_type[k], 1.5);
        }
    }
    fclose(fp);
}

#pragma mark -
#pragma mark Utilities
char * load_program_source(const char *filename)
{ 
	
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
	
	return source; 
} 

#pragma mark -
#pragma mark Main OpenCL Routine
int runCL(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, float dt, 
          int max_ele_no, 
          int ele_no, int cell_no, int * ele_per_cell,
          int * cclock, int * cycle)
{
    int flag_print = 0;
    cl_program program[10], program1;
    cl_kernel kernel[12], kernel1;
	
    cl_command_queue cmd_queue;
    cl_context   context;
 
    cl_int err = 0;
    size_t returned_size = 0;
    size_t buffer_size;
	
    // Allocate memory for the results
    float * xF = (float *)malloc(max_ele_no*sizeof(float));
    float * yF = (float *)malloc(max_ele_no*sizeof(float));
    float * zF = (float *)malloc(max_ele_no*sizeof(float));
    float * xc = (float *)malloc(MAX_CELL*sizeof(float));
    float * yc = (float *)malloc(MAX_CELL*sizeof(float));
    float * zc = (float *)malloc(MAX_CELL*sizeof(float));
    int * flag = (int *)malloc(MAX_CELL*sizeof(int));
    float * rand_no = (float *)malloc(max_ele_no*sizeof(float));
    float * chem1_gpu = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * chem2_gpu = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * chem11_gpu = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * chem22_gpu = (float *)malloc(NX*NY*NZ*sizeof(float));
    int * grid_gpu = (int *)malloc(NX*NY*NZ*sizeof(int));
    int * grid1_gpu = (int *)malloc(NX*NY*NZ*sizeof(int));
    int * grid2_gpu = (int *)malloc(NX*NY*NZ*sizeof(int));
    int * grid3_gpu = (int *)malloc(NX*NY*NZ*sizeof(int));
    int * grid4_gpu = (int *)malloc(NX*NY*NZ*sizeof(int));
    int * flag_death = (int *)malloc(MAX_CELL*sizeof(int));
    int * flag_gener = (int *)malloc(MAX_CELL*sizeof(int));
    float * OVOL1 = (float *)malloc(MAX_CELL*sizeof(float));
    float * OVOL2 = (float *)malloc(MAX_CELL*sizeof(float));
    float * vx = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * vy = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * vz = (float *)malloc(NX*NY*NZ*sizeof(float));
    float * xo = (float *)malloc(max_ele_no*sizeof(float));
    float * yo = (float *)malloc(max_ele_no*sizeof(float));
    float * zo = (float *)malloc(max_ele_no*sizeof(float));

    // Bin realated declarations
    int * bin_id_host = (int *)malloc(max_ele_no * sizeof(int));
    int * bin_count_host = (int *)malloc(NBINS * sizeof(int));
    int * bin_offset_host = (int *)malloc((NBINS + 1) * sizeof(int));
    int * bin_pos_host = (int *)malloc(NBINS * sizeof(int));
    int * sorted_ids_host = (int *)malloc(max_ele_no * sizeof(int));


    srand(time(NULL));

    for(int i = 0; i < max_ele_no; i++)
    {
        xF[i] = yF[i] = zF[i] = 0.0f;
        xo[i] = yo[i] = zo[i] = 0.0f;
    }
    for(int i = 0; i < MAX_CELL; i++)
    {
        xc[i] = yc[i] = zc[i] = 0.0;
        flag[i] = flag_death[i] = flag_gener[i] = 0;
        OVOL1[i] = 0.9;
        OVOL2[i] = 0.9;
    }

    unsigned int * rng_x = (unsigned int *)malloc(max_ele_no*sizeof(unsigned int));
    unsigned int * rng_c = (unsigned int *)malloc(max_ele_no*sizeof(unsigned int));
    for(int i = 0; i < max_ele_no; i++)
    {
        rng_x[i] = rand();
        rng_c[i] = rand();
    }

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

    // Bin related declarations (buffers created below, after context exists)
    cl_mem bin_id_mem, bin_count_mem, bin_offset_mem, bin_pos_mem, sorted_ids_mem;

    // Get platform and device information
    cl_platform_id *platform_id = NULL;
    cl_device_id cpu = NULL, device = NULL;
    cl_device_id* devices_new;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_platform_id cpPlatform = NULL; // OpenCL platform
    cl_device_id device_id = NULL; // OpenCL device

    cl_uint nplat = 0;
    err = clGetPlatformIDs(0, NULL, &nplat);
    assert(err == CL_SUCCESS);
    printf("# of Platforms = %u\n", nplat);

    cl_platform_id *plats = malloc(sizeof(cl_platform_id) * nplat);
    clGetPlatformIDs(nplat, plats, NULL);

    const cl_device_type wanted = CL_DEVICE_TYPE_GPU; // The type of requested device

    int found = 0;
    for (cl_uint i = 0; i < nplat && !found; ++i) {
        char pname[128] = {0};
        clGetPlatformInfo(plats[i], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        printf("Platform %u: %s\n", i, pname);

        if (clGetDeviceIDs(plats[i], wanted, 1, &device_id, NULL) == CL_SUCCESS) {
            cpPlatform = plats[i];
            found = 1;
        }
    }
    free(plats);
    assert(found);

    printf("err = %d\n", err);
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
        printf("Connecting to %s %s supporting ", vendor_name, device_name);
        printf("%s...\n", buf);
    }
	
#pragma mark Context and Command Queue
	{
    	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    	assert(err == CL_SUCCESS);
    	cmd_queue = clCreateCommandQueueWithProperties(context, device_id, 0, NULL);
	}

    // Bin buffer allocation (requires context)
    bin_id_mem     = clCreateBuffer(context, CL_MEM_READ_WRITE, max_ele_no * sizeof(int), NULL, &err);
    bin_count_mem  = clCreateBuffer(context, CL_MEM_READ_WRITE, NBINS * sizeof(int), NULL, &err);
    bin_offset_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (NBINS + 1) * sizeof(int), NULL, &err);
    bin_pos_mem    = clCreateBuffer(context, CL_MEM_READ_WRITE, NBINS * sizeof(int), NULL, &err);
    sorted_ids_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, max_ele_no * sizeof(int), NULL, &err);
    assert(err == CL_SUCCESS);
	
#pragma mark Program and Kernel Creation
	{
        // create program for cell movement
    	const char * filename = "movement.cl";
    	char *program_source = load_program_source(filename);
    	program[0] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
											   NULL, &err);
    	assert(err == CL_SUCCESS);
    	err = clBuildProgram(program[0], 0, NULL, NULL, NULL, NULL);
    	assert(err == CL_SUCCESS);

        char build[2048];
        clGetProgramBuildInfo(program[0], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

    	kernel[0] = clCreateKernel(program[0], "movement", &err);

        size_t workgroup_size;
        err = clGetKernelWorkGroupInfo(kernel[0], device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &workgroup_size, NULL);
	}
	
	{
        // create program for cell growth
    	const char * filename = "growth.cl";
    	char *program_source = load_program_source(filename);

        program[1] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
											   NULL, &err);
        assert(err == CL_SUCCESS);

        err = clBuildProgram(program[1], 0, NULL, NULL, NULL, NULL);

        char build[2048];
        clGetProgramBuildInfo(program[1], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

    	kernel[1] = clCreateKernel(program[1], "growth", &err);

        size_t workgroup_size;
        err = clGetKernelWorkGroupInfo(kernel[1], device, CL_KERNEL_WORK_GROUP_SIZE,
                                sizeof(size_t), &workgroup_size, NULL);
	}

    {
        // create program to update cell center position
        const char * filename = "cellcenter.cl";
        char *program_source = load_program_source(filename);

        program[4] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
                                               NULL, &err);
        assert(err == CL_SUCCESS);

        err = clBuildProgram(program[4], 0, NULL, NULL, NULL, NULL);

        char build[2048];
        clGetProgramBuildInfo(program[4], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

        kernel[4] = clCreateKernel(program[4], "cellcenter", &err);

        size_t workgroup_size;
        err = clGetKernelWorkGroupInfo(kernel[4], device, CL_KERNEL_WORK_GROUP_SIZE,
                                sizeof(size_t), &workgroup_size, NULL);
    }

    {
        // create program to evolve chemical field
        const char * filename = "chem_rk_gpu.cl";
        char *program_source = load_program_source(filename);

        program[5] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
                                               NULL, &err);
        assert(err == CL_SUCCESS);

        err = clBuildProgram(program[5], 0, NULL, NULL, NULL, NULL);

        char build[2048];
        clGetProgramBuildInfo(program[5], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

        kernel[5] = clCreateKernel(program[5], "chem_rk_gpu", &err);

        size_t workgroup_size;
        err = clGetKernelWorkGroupInfo(kernel[5], device, CL_KERNEL_WORK_GROUP_SIZE,
                                sizeof(size_t), &workgroup_size, NULL);
    }

    {
        // create program to evolve intracellular gene regulatory network
        const char * filename = "ovol_fun.cl";
        char *program_source = load_program_source(filename);

        program[7] = clCreateProgramWithSource(context, 1, (const char**)&program_source,
                                               NULL, &err);
        assert(err == CL_SUCCESS);

        err = clBuildProgram(program[7], 0, NULL, NULL, NULL, NULL);

        char build[2048];
        clGetProgramBuildInfo(program[7], device, CL_PROGRAM_BUILD_LOG, 2048, build, NULL);

        kernel[7] = clCreateKernel(program[7], "ovol_fun", &err);

        size_t workgroup_size;
        err = clGetKernelWorkGroupInfo(kernel[7], device, CL_KERNEL_WORK_GROUP_SIZE,
                                sizeof(size_t), &workgroup_size, NULL);
    }
    
    {
        // create program for binning logic
        const char * filename = "binning.cl";
        char *program_source = load_program_source(filename);
        
        program[8] = clCreateProgramWithSource(context, 1, (const char**)&program_source, NULL, &err);
        assert(err == CL_SUCCESS);
        err = clBuildProgram(program[8], 0, NULL, NULL, NULL, NULL);

        char build[2048] = {0};
        clGetProgramBuildInfo(program[8], device_id, CL_PROGRAM_BUILD_LOG, sizeof(build), build, NULL);

        kernel[8] = clCreateKernel(program[8], "bin_assign", &err);
        assert(err == CL_SUCCESS);
        kernel[9] = clCreateKernel(program[8], "bin_histogram", &err);
        assert(err == CL_SUCCESS);
        kernel[10] = clCreateKernel(program[8], "bin_prefix_sum", &err);
        assert(err == CL_SUCCESS);
        kernel[11] = clCreateKernel(program[8], "bin_scatter", &err);
        assert(err == CL_SUCCESS);
    }


#pragma mark Memory Allocation
	{
        buffer_size = sizeof(float) * max_ele_no;
		
    	x_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err = clEnqueueWriteBuffer(cmd_queue, x_mem, CL_TRUE, 0, buffer_size,
								   (void*)x, 0, NULL, NULL);
    	y_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, y_mem, CL_TRUE, 0, buffer_size,
									(void*)y, 0, NULL, NULL);
    	z_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, z_mem, CL_TRUE, 0, buffer_size,
									(void*)z, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);
             		xF_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err = clEnqueueWriteBuffer(cmd_queue, xF_mem, CL_TRUE, 0, buffer_size,
								   (void*)xF, 0, NULL, NULL);
    	yF_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, yF_mem, CL_TRUE, 0, buffer_size,
									(void*)yF, 0, NULL, NULL);
    	zF_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, zF_mem, CL_TRUE, 0, buffer_size,
									(void*)zF, 0, NULL, NULL);
        xo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, xo_mem, CL_TRUE, 0, buffer_size,
                                   (void*)xo, 0, NULL, NULL);
        yo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, yo_mem, CL_TRUE, 0, buffer_size,
                                   (void*)yo, 0, NULL, NULL);
        zo_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, zo_mem, CL_TRUE, 0, buffer_size,
                                   (void*)zo, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);

    	buffer_size = sizeof(int) * max_ele_no;
    	id_mem	= clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err = clEnqueueWriteBuffer(cmd_queue, id_mem, CL_TRUE, 0, buffer_size,
									(void*)id, 0, NULL, NULL);
    	ele_type_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, buffer_size,
									(void*)ele_type, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);
		
    	buffer_size = sizeof(int) * MAX_CELL;
        cell_type_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err = clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, buffer_size,
                                    (void*)cell_type, 0, NULL, NULL);
    	ele_per_cell_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, buffer_size,
									(void*)ele_per_cell, 0, NULL, NULL);
    	flag_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, flag_mem, CL_TRUE, 0, buffer_size,
									(void*)flag, 0, NULL, NULL);
    	flag_death_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, flag_death_mem, CL_TRUE, 0, buffer_size,
									(void*)flag_death, 0, NULL, NULL);
        flag_gener_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, flag_gener_mem, CL_TRUE, 0, buffer_size,
                                    (void*)flag_gener, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);

    	buffer_size = sizeof(float) * MAX_CELL;
    	xc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err = clEnqueueWriteBuffer(cmd_queue, xc_mem, CL_TRUE, 0, buffer_size,
								   (void*)xc, 0, NULL, NULL);
    	yc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, yc_mem, CL_TRUE, 0, buffer_size,
								   (void*)yc, 0, NULL, NULL);
    	zc_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
    	err |= clEnqueueWriteBuffer(cmd_queue, zc_mem, CL_TRUE, 0, buffer_size,
								   (void*)zc, 0, NULL, NULL);
        o1_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, o1_mem, CL_TRUE, 0, buffer_size,
                                   (void*)OVOL1, 0, NULL, NULL);
        o2_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, o2_mem, CL_TRUE, 0, buffer_size,
                                   (void*)OVOL2, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);

        buffer_size = sizeof(float) * NX*NY*NZ;
        chem1_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err = clEnqueueWriteBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)chem1_gpu, 0, NULL, NULL);
        chem2_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)chem2_gpu, 0, NULL, NULL);
        chem11_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err = clEnqueueWriteBuffer(cmd_queue, chem11_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)chem11_gpu, 0, NULL, NULL);
        chem22_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, chem22_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)chem22_gpu, 0, NULL, NULL);
        vx_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, vx_mem, CL_TRUE, 0, buffer_size,
                                   (void*)vx, 0, NULL, NULL);
        vy_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, vy_mem, CL_TRUE, 0, buffer_size,
                                   (void*)vy, 0, NULL, NULL);
        vz_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, vz_mem, CL_TRUE, 0, buffer_size,
                                   (void*)vz, 0, NULL, NULL);
    	assert(err == CL_SUCCESS);

        buffer_size = sizeof(int) * NX*NY*NZ;
        grid_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, grid_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)grid_gpu, 0, NULL, NULL);
        grid1_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, grid1_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)grid1_gpu, 0, NULL, NULL);
        grid2_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, grid2_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)grid2_gpu, 0, NULL, NULL);
        grid3_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, grid3_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)grid3_gpu, 0, NULL, NULL);
        grid4_gpu_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, grid4_gpu_mem, CL_TRUE, 0, buffer_size,
                                   (void*)grid4_gpu, 0, NULL, NULL);
        assert(err == CL_SUCCESS);

        buffer_size = max_ele_no*sizeof(unsigned int);
        rng_x_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err = clEnqueueWriteBuffer(cmd_queue, rng_x_mem, CL_TRUE, 0, buffer_size,
                                   (void*)rng_x, 0, NULL, NULL);
        rng_c_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, buffer_size, NULL, NULL);
        err |= clEnqueueWriteBuffer(cmd_queue, rng_c_mem, CL_TRUE, 0, buffer_size,
                                   (void*)rng_c, 0, NULL, NULL);
        assert(err == CL_SUCCESS);

    	clFinish(cmd_queue);
	}
	
#pragma mark Kernel Arguments
	{
        int max_ele = MAX_ELE;
	    // Now setup the arguments to kernel "movement"
    	err  = clSetKernelArg(kernel[0],  0, sizeof(cl_mem), &id_mem);
    	err |= clSetKernelArg(kernel[0],  1, sizeof(cl_mem), &x_mem);
    	err |= clSetKernelArg(kernel[0],  2, sizeof(cl_mem), &y_mem);
    	err |= clSetKernelArg(kernel[0],  3, sizeof(cl_mem), &z_mem);
    	err |= clSetKernelArg(kernel[0],  4, sizeof(cl_mem), &ele_type_mem);
    	err |= clSetKernelArg(kernel[0],  5, sizeof(int), &cell_no);
    	err |= clSetKernelArg(kernel[0],  6, sizeof(int), &ele_no);
    	err |= clSetKernelArg(kernel[0],  7, sizeof(cl_mem), &ele_per_cell_mem);
    	err |= clSetKernelArg(kernel[0],  8, sizeof(cl_mem), &xc_mem);
    	err |= clSetKernelArg(kernel[0],  9, sizeof(cl_mem), &yc_mem);
    	err |= clSetKernelArg(kernel[0], 10, sizeof(cl_mem), &zc_mem);
    	err |= clSetKernelArg(kernel[0], 11, sizeof(float), &dt);
    	err |= clSetKernelArg(kernel[0], 12, sizeof(cl_mem), &xF_mem);
    	err |= clSetKernelArg(kernel[0], 13, sizeof(cl_mem), &yF_mem);
    	err |= clSetKernelArg(kernel[0], 14, sizeof(cl_mem), &zF_mem);
    	err |= clSetKernelArg(kernel[0], 15, sizeof(float), &lx);
    	err |= clSetKernelArg(kernel[0], 16, sizeof(float), &ly);
    	err |= clSetKernelArg(kernel[0], 17, sizeof(float), &lz);
    	err |= clSetKernelArg(kernel[0], 18, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[0], 19, sizeof(cl_mem), &cell_type_mem);
        err |= clSetKernelArg(kernel[0], 20, sizeof(cl_mem), &flag_gener_mem);
    	assert(err == CL_SUCCESS);

	}
	
	{
        int max_ele = MAX_ELE;
    	// Now setup the arguments to kernel "growth"
    	err  = clSetKernelArg(kernel[1],  0, sizeof(cl_mem), &id_mem);
    	err |= clSetKernelArg(kernel[1],  1, sizeof(cl_mem), &x_mem);
    	err |= clSetKernelArg(kernel[1],  2, sizeof(cl_mem), &y_mem);
    	err |= clSetKernelArg(kernel[1],  3, sizeof(cl_mem), &z_mem);
    	err |= clSetKernelArg(kernel[1],  4, sizeof(cl_mem), &ele_type_mem);
    	err |= clSetKernelArg(kernel[1],  5, sizeof(int), &ele_no);
    	err |= clSetKernelArg(kernel[1],  6, sizeof(cl_mem), &flag_mem);
    	err |= clSetKernelArg(kernel[1],  7, sizeof(cl_mem), &ele_per_cell_mem);
    	err |= clSetKernelArg(kernel[1],  8, sizeof(float), &lx);
    	err |= clSetKernelArg(kernel[1],  9, sizeof(float), &ly);
    	err |= clSetKernelArg(kernel[1], 10, sizeof(float), &lz);
    	err |= clSetKernelArg(kernel[1], 11, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[1], 12, sizeof(cl_mem), &cell_type_mem);
    	err |= clSetKernelArg(kernel[1], 13, sizeof(cl_mem), &flag_death_mem);
    	assert(err == CL_SUCCESS);
	}
	
    {
        int max_ele = MAX_ELE;
        // Now setup the arguments to kernel "cellcenter"
        err  = clSetKernelArg(kernel[4],  0, sizeof(cl_mem), &id_mem);
        err |= clSetKernelArg(kernel[4],  1, sizeof(cl_mem), &x_mem);
        err |= clSetKernelArg(kernel[4],  2, sizeof(cl_mem), &y_mem);
        err |= clSetKernelArg(kernel[4],  3, sizeof(cl_mem), &z_mem);
        err |= clSetKernelArg(kernel[4],  4, sizeof(cl_mem), &ele_per_cell_mem);
        err |= clSetKernelArg(kernel[4],  5, sizeof(cl_mem), &xc_mem);
        err |= clSetKernelArg(kernel[4],  6, sizeof(cl_mem), &yc_mem);
        err |= clSetKernelArg(kernel[4],  7, sizeof(cl_mem), &zc_mem);
    	err |= clSetKernelArg(kernel[4],  8, sizeof(int), &max_ele);
    	err |= clSetKernelArg(kernel[4],  9, sizeof(float), &lx);
    	err |= clSetKernelArg(kernel[4], 10, sizeof(float), &ly);
    	err |= clSetKernelArg(kernel[4], 11, sizeof(float), &lz);
        assert(err == CL_SUCCESS);
    }

    {
        int nx0 = NX, ny0 = NY, nz0 = NZ, tmpii = 0;
        float dx0 = DX, dy0 = DY, dz0 = DZ;
        float dt_chem = 1.0*dt;
        // Now setup the arguments to kernel "chem_rk_gpu"
        err  = clSetKernelArg(kernel[5],  0, sizeof(cl_mem), &chem1_gpu_mem);
        err |= clSetKernelArg(kernel[5],  1, sizeof(cl_mem), &grid_gpu_mem);
        err |= clSetKernelArg(kernel[5],  2, sizeof(int), &nx0);
        err |= clSetKernelArg(kernel[5],  3, sizeof(int), &ny0);
        err |= clSetKernelArg(kernel[5],  4, sizeof(int), &nz0);
        err |= clSetKernelArg(kernel[5],  5, sizeof(float), &dx0);
        err |= clSetKernelArg(kernel[5],  6, sizeof(float), &dy0);
        err |= clSetKernelArg(kernel[5],  7, sizeof(float), &dz0);
        err |= clSetKernelArg(kernel[5],  8, sizeof(float), &dt_chem);
        err |= clSetKernelArg(kernel[5],  9, sizeof(cl_mem), &chem11_gpu_mem);
        err |= clSetKernelArg(kernel[5], 10, sizeof(cl_mem), &grid1_gpu_mem);
        err |= clSetKernelArg(kernel[5], 11, sizeof(cl_mem), &chem2_gpu_mem);
        err |= clSetKernelArg(kernel[5], 12, sizeof(cl_mem), &chem22_gpu_mem);
        err |= clSetKernelArg(kernel[5], 13, sizeof(cl_mem), &grid2_gpu_mem);
        err |= clSetKernelArg(kernel[5], 14, sizeof(cl_mem), &grid3_gpu_mem);
        err |= clSetKernelArg(kernel[5], 15, sizeof(int), &tmpii);
        err |= clSetKernelArg(kernel[5], 16, sizeof(cl_mem), &grid4_gpu_mem);
    	err |= clSetKernelArg(kernel[5], 17, sizeof(cl_mem), &vx_mem);
    	err |= clSetKernelArg(kernel[5], 18, sizeof(cl_mem), &vy_mem);
    	err |= clSetKernelArg(kernel[5], 19, sizeof(cl_mem), &vz_mem);
        assert(err == CL_SUCCESS);
    }


    {
        int max_ele = MAX_ELE;
        // Now setup the arguments to kernel "ovol_fun"
        err  = clSetKernelArg(kernel[7],  0, sizeof(cl_mem), &id_mem);
        err |= clSetKernelArg(kernel[7],  1, sizeof(cl_mem), &x_mem);
        err |= clSetKernelArg(kernel[7],  2, sizeof(cl_mem), &y_mem);
        err |= clSetKernelArg(kernel[7],  3, sizeof(cl_mem), &z_mem);
        err |= clSetKernelArg(kernel[7],  4, sizeof(cl_mem), &ele_per_cell_mem);
        err |= clSetKernelArg(kernel[7],  5, sizeof(cl_mem), &chem1_gpu_mem);
        err |= clSetKernelArg(kernel[7],  6, sizeof(cl_mem), &chem2_gpu_mem);
        err |= clSetKernelArg(kernel[7],  7, sizeof(cl_mem), &o1_mem);
        err |= clSetKernelArg(kernel[7],  8, sizeof(cl_mem), &o2_mem);
        err |= clSetKernelArg(kernel[7],  9, sizeof(int), &max_ele);
        err |= clSetKernelArg(kernel[7], 10, sizeof(cl_mem), &cell_type_mem);
        assert(err == CL_SUCCESS);
    }
    
    {
        const float bin_size_f = BIN_SIZE;
        const int nbx_i = NBX, nby_i = NBY, nbz_i = NBZ, nbins_i = NBINS;
        const int max_ele_i = MAX_ELE;

        // kernel[8] bin_assign
        err = clSetKernelArg(kernel[8], 0, sizeof(cl_mem), &x_mem);
        err |= clSetKernelArg(kernel[8], 1, sizeof(cl_mem), &y_mem);
        err |= clSetKernelArg(kernel[8], 2, sizeof(cl_mem), &z_mem);
        err |= clSetKernelArg(kernel[8], 3, sizeof(cl_mem), &ele_per_cell_mem);
        err |= clSetKernelArg(kernel[8], 4, sizeof(int), &max_ele_i);
        
        // arg 5 (cell_no) set per step
        err |= clSetKernelArg(kernel[8], 6, sizeof(float), &bin_size_f);
        err |= clSetKernelArg(kernel[8], 7, sizeof(int), &nbx_i);
        err |= clSetKernelArg(kernel[8], 8, sizeof(int), &nby_i); 
        err |= clSetKernelArg(kernel[8], 9, sizeof(int), &nbz_i);
        err |= clSetKernelArg(kernel[8], 10, sizeof(int), &nbins_i);
        err |= clSetKernelArg(kernel[8], 11, sizeof(cl_mem), &bin_id_mem);  

        // kernel[9] bin_histogram
        err |= clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &bin_id_mem);
        err |= clSetKernelArg(kernel[9], 1, sizeof(int), &nbins_i);
        err |= clSetKernelArg(kernel[9], 2, sizeof(cl_mem), &bin_count_mem);

        // kernel[10] bin_prefix_sum
        err |= clSetKernelArg(kernel[10], 0, sizeof(cl_mem), &bin_count_mem);
        err |= clSetKernelArg(kernel[10], 1, sizeof(int), &nbins_i);
        err |= clSetKernelArg(kernel[10], 2, sizeof(cl_mem), &bin_offset_mem);

        // kernel[11] bin_scatter
        err |= clSetKernelArg(kernel[11], 0, sizeof(cl_mem), &bin_id_mem);
        err |= clSetKernelArg(kernel[11], 1, sizeof(cl_mem), &bin_offset_mem);
        err |= clSetKernelArg(kernel[11], 2, sizeof(int), &nbins_i);
        err |= clSetKernelArg(kernel[11], 3, sizeof(cl_mem), &bin_pos_mem);
        err |= clSetKernelArg(kernel[11], 4, sizeof(cl_mem), &sorted_ids_mem);
        
        assert(err == CL_SUCCESS);

        // Bin related arguments
        err = clSetKernelArg(kernel[0], 21, sizeof(int), &nbx_i);
        err |= clSetKernelArg(kernel[0], 22, sizeof(int), &nby_i);
        err |= clSetKernelArg(kernel[0], 23, sizeof(int), &nbz_i);
        err |= clSetKernelArg(kernel[0], 24, sizeof(float), &bin_size_f);
        err |= clSetKernelArg(kernel[0], 25, sizeof(cl_mem), &bin_offset_mem);
        err |= clSetKernelArg(kernel[0], 26, sizeof(cl_mem), &sorted_ids_mem);
        assert(err == CL_SUCCESS);
    }

#pragma imark Execution and Read

#if 0
    // update the initial chemical field
    for(int i = 0; i < NX; i++)    
        for(int j = 0; j < NY; j++)    
            for(int k = 0; k < NZ; k++) 
    {
        int indx = i + j*NX + k*NX*NY;
        chem1_gpu[indx] = chem1[i][j][k];
        chem2_gpu[indx] = chem2[i][j][k];
        grid_gpu[indx] = grid[i][j][k];
        grid1_gpu[indx] = grid1[i][j][k];
        grid2_gpu[indx] = grid2[i][j][k];
        grid3_gpu[indx] = grid3[i][j][k];
        grid4_gpu[indx] = grid4[i][j][k];
    }   
    err = clEnqueueWriteBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float),
                        (void*)chem1_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float),
                        (void*)chem2_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(int),
                        (void*)grid_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid1_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(int),
                        (void*)grid1_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid2_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(int),
                        (void*)grid2_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid3_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(int),
                        (void*)grid3_gpu, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(cmd_queue, grid4_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(int),
                        (void*)grid4_gpu, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
#endif

    {
    // update cell centers
    size_t global_work_size = cell_no;
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[4], 1, NULL,
                                 &global_work_size, NULL, 0, NULL, NULL);
    assert(err == CL_SUCCESS);
    clFinish(cmd_queue);

    // index to note how many empty spots in the cell array due to death
    int no_empty = 0;

    // start of time iterations
    int iterations = MAX_ITERATIONS;
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
            for(int i = 0; i < NX; i++)    
                for(int j = 0; j < NY; j++)    
                    for(int k = 0; k < NZ; k++) 
            {
                int indx = i + j*NX + k*NX*NY;
                grid_gpu[indx] = grid[i][j][k];
                grid1_gpu[indx] = grid1[i][j][k];
                grid2_gpu[indx] = grid2[i][j][k];
                grid3_gpu[indx] = grid3[i][j][k];
                grid4_gpu[indx] = grid4[i][j][k];
            }   

            buffer_size = sizeof(int) * NX*NY*NZ;
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
            buffer_size = sizeof(float) * NX*NY*NZ;
            err = clEnqueueWriteBuffer(cmd_queue, vx_mem, CL_TRUE, 0, buffer_size,
                                (void*)vx, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, vy_mem, CL_TRUE, 0, buffer_size,
                                (void*)vy, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, vz_mem, CL_TRUE, 0, buffer_size,
                                (void*)vz, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            /*
            // update chemical field
            size_t global_work_size = NX*NY*NZ;
            for(int iter_chem = 0; iter_chem < MAX_CHEM_ITER; iter_chem++)
            {
                err = clSetKernelArg(kernel[5], 15, sizeof(int), &(iter_chem));
                err |= clEnqueueNDRangeKernel(cmd_queue, kernel[5], 1, NULL,
                                         &global_work_size, NULL, 0, NULL, NULL);
                assert(err == CL_SUCCESS);

                err = clEnqueueCopyBuffer(cmd_queue, chem11_gpu_mem, chem1_gpu_mem, 0, 0, NX*NY*NZ*sizeof(float),
                                  0, 0, NULL);
                err |= clEnqueueCopyBuffer(cmd_queue, chem22_gpu_mem, chem2_gpu_mem, 0, 0, NX*NY*NZ*sizeof(float),
                                  0, 0, NULL);
                assert(err == CL_SUCCESS);
            }
            clFinish(cmd_queue);
            */

            /*
            // update intracellular gene network
            global_work_size = cell_no;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel[7], 1, NULL,
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
        size_t bin_global = cell_no * MAX_ELE;
        size_t one = 1;
        int zero_int = 0;
        
        err = clSetKernelArg(kernel[8], 5, sizeof(int), &cell_no); // bin_assign cell_no
        
        // zero histogram and write cursor
        err |= clEnqueueFillBuffer(cmd_queue, bin_count_mem, &zero_int, sizeof(int),
                                    0, NBINS * sizeof(int), 0, NULL, NULL);
        err |= clEnqueueFillBuffer(cmd_queue, bin_pos_mem, &zero_int, sizeof(int),
                                    0, NBINS * sizeof(int), 0, NULL, NULL);

        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[8], 1, NULL, &bin_global, NULL, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[9], 1, NULL, &bin_global, NULL, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[10], 1, NULL, &one, NULL, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[11], 1, NULL, &bin_global, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);

        // cell movement
        err = clSetKernelArg(kernel[0],  5, sizeof(int), &cell_no);
        err |= clSetKernelArg(kernel[0],  6, sizeof(int), &ele_no);
        assert(err == CL_SUCCESS);
        // global_work_size = max_ele_no;
        global_work_size = cell_no*MAX_ELE;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[0], 1, NULL, 
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


#if 0
        // start of cell growth
        if((iter-1)%GROWTH_FR == 0)
        {
            for(int i = 0; i < cell_no; i++)
            {
                flag[i] = 0;
            }
	        buffer_size = sizeof(int) * MAX_CELL;
  	        err = clEnqueueWriteBuffer(cmd_queue, flag_mem, CL_TRUE, 0, buffer_size,
	    						(void*)flag, 0, NULL, NULL);
	    	err |= clEnqueueWriteBuffer(cmd_queue, flag_death_mem, CL_TRUE, 0, buffer_size,
	    						(void*)flag_death, 0, NULL, NULL);
    		err |= clSetKernelArg(kernel[1],  5, sizeof(int), &ele_no);
    		assert(err == CL_SUCCESS);
            global_work_size = cell_no;
	    	err = clEnqueueNDRangeKernel(cmd_queue, kernel[1], 1, NULL, 
	    								 &global_work_size, NULL, 0, NULL, NULL);
	    	assert(err == CL_SUCCESS);
    		clFinish(cmd_queue);
            
    		err = clEnqueueReadBuffer(cmd_queue, flag_mem, CL_TRUE, 0, cell_no*sizeof(int), 
    				          flag, 0, NULL, NULL);
    		assert(err == CL_SUCCESS);
            for(int i = 0; i < cell_no; i++)
            {
                if(flag[i] == 1) ele_no++;
            }
        } // end of cell growth
#endif

#if 0
        // start of cell division
        if((iter-1)%DIVISION_FR == 0 && cell_no < MAX_CELL)
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
            err |= clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, cell_no*sizeof(int),
                              ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float),
                              chem1_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float),
                              chem2_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0, MAX_CELL*sizeof(float),
                              xc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0, MAX_CELL*sizeof(float),
                              yc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0, MAX_CELL*sizeof(float),
                              zc, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            int tmp_cell_no = cell_no;
            float tmpx0, tmpy0, tmpz0;
            int tmpn0, tmptype0;
            float tmpx[MAX_ELE], tmpy[MAX_ELE], tmpz[MAX_ELE];
            int tmptype[MAX_ELE];
            int id_for_new_cell;
            for(int i = 0; i < cell_no; i++)
            {
                cclock[i] += DIVISION_FR;
                if(cell_type[i]==2)
                    printf("cell[%d]  cell_type %d  element_per_cell %d  cclock %d  cycle %d\n",
                            i, cell_type[i], ele_per_cell[i], cclock[i], cycle[i]);
                if(cell_type[i] >= 3) continue; // cell no division
                if(ele_per_cell[i] < 16) continue;
                if(cclock[i] < cycle[i]) continue; // cell not mature

                printf("read to divide\n");

                if(no_empty > 0)
                {
                    for(int j = 0; j < cell_no; j++)
                    {
                        if(ele_per_cell[j] == 0)
                        {
                            id_for_new_cell = j;
                        }
                    }
                }
                else
                {
                    id_for_new_cell = tmp_cell_no;
                }

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
                    exit(-1);
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
                    cell_type[i] = 1;
                    cell_type[id_for_new_cell] = 1;
                    if(RNG() < R_Diff) cell_type[id_for_new_cell] = 2;
                    ctype1 = cell_type[i];
                    ctype2 = cell_type[id_for_new_cell];
                }
                else // mother is fast-dividing cell
                {
                    cell_type[i] = 2;
                    cell_type[id_for_new_cell] = 2;
                    ctype1 = cell_type[i];
                    ctype2 = cell_type[id_for_new_cell];
                }

                flag_death[i] = flag_death[id_for_new_cell] = 0;

                cclock[i] = cclock[id_for_new_cell] = 0;
                if(cell_type[i] == 1)
                    cycle[i] = (int)(1000.0f*(C1_MEAN + RNG()*C1_STD));
                else if(cell_type[i] == 2)
                    cycle[i] = (int)(1000.0f*(C2_MEAN + RNG()*C2_STD));
                else
                {
                    printf("ERROR::cell division, wrong cell type = %d\n", cell_type[i]);
                    exit(0);
                }
                if(cell_type[id_for_new_cell] == 1)
                    cycle[id_for_new_cell] = (int)(1000.0f*(C1_MEAN + RNG()*C1_STD));
                else if(cell_type[id_for_new_cell] == 2)
                    cycle[id_for_new_cell] = (int)(1000.0f*(C2_MEAN + RNG()*C2_STD));
                else
                {
                    printf("ERROR::cell division, wrong cell type = %d\n", cell_type[i]);
                    exit(0);
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
                        int k = i*MAX_ELE + ele_per_cell[i];
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
                        int k = id_for_new_cell*MAX_ELE + ele_per_cell[id_for_new_cell];
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

                if(no_empty > 0)
                {
                    no_empty--;
                }
                else
                {
                    tmp_cell_no++;
                }
                if(tmp_cell_no >= MAX_CELL)
                {
                    exit(0);
                } 
            }
            cell_no = tmp_cell_no;

            err = clEnqueueWriteBuffer(cmd_queue, id_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                            (void*)id, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)x, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)y, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float),
                            (void*)z, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, cell_type_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)cell_type, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, max_ele_no*sizeof(int),
                            (void*)ele_type, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueWriteBuffer(cmd_queue, flag_gener_mem, CL_TRUE, 0, cell_no*sizeof(int),
                            (void*)flag_gener, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
        } // end of cell division
#endif

#if 0
        // start of cell death
        int FR_death = 500;
        if((iter-1)%FR_death == 0)
        {
            // update OVOL1 once since death runs more frequently
            global_work_size = cell_no;
            err = clEnqueueNDRangeKernel(cmd_queue, kernel[7], 1, NULL,
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

        // update cell center
        size_t global_work_size = cell_no;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[4], 1, NULL,
                                     &global_work_size, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);
        clFinish(cmd_queue);

        // start of output
        if(iter%OUTPUT_FR == 0)
        {
            err = clEnqueueReadBuffer(cmd_queue, x_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  x, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, y_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  y, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, z_mem, CL_TRUE, 0, max_ele_no*sizeof(float), 
								  z, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, xc_mem, CL_TRUE, 0, cell_no*sizeof(float),
                                  xc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, yc_mem, CL_TRUE, 0, cell_no*sizeof(float),
                                  yc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, zc_mem, CL_TRUE, 0, cell_no*sizeof(float),
                                  zc, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, id_mem, CL_TRUE, 0, max_ele_no*sizeof(int), 
								  id, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, ele_type_mem, CL_TRUE, 0, max_ele_no*sizeof(int), 
								  ele_type, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, ele_per_cell_mem, CL_TRUE, 0, MAX_CELL*sizeof(int), 
								  ele_per_cell, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem1_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float), 
								  chem1_gpu, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, chem2_gpu_mem, CL_TRUE, 0, NX*NY*NZ*sizeof(float), 
								  chem2_gpu, 0, NULL, NULL);
            assert(err == CL_SUCCESS);
            output(x, y, z, xc, yc, zc, id, cell_type, ele_type, cell_no, ele_no, ele_per_cell, iter, chem1_gpu, chem2_gpu, cclock, cycle, flag_gener, vx, vy, vz);
            for(int i = 0; i < cell_no; i++)
                for(int j = 0; j < ele_per_cell[i]; j++)
            {
                int k = i*MAX_ELE + j;
                
                if(isnan(x[k]) || isnan(y[k]) || isnan(z[k]))
                {
                    exit(-1);
                }
            }
        } // end of output

        // count the number of each cell type
        if(iter%1000 == 0)
        {
            count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0;
            for(int i = 0; i < cell_no; i++)
            {
                if(ele_per_cell[i] == 0) continue;
                count0++;
                if(cell_type[i] == 1 && ele_per_cell[i] > 0) count1++;
                if(cell_type[i] == 2 && ele_per_cell[i] > 0) count2++;
                if(cell_type[i] == 3 && ele_per_cell[i] > 0) count3++;
                if(cell_type[i] == 4 && ele_per_cell[i] > 0) count4++;
            }
            printf("%6d  %4d %4d %4d %4d\n", iter, count1, count2, count3, count4); fflush(stdout);
            if(count1 == 0)
            {
                exit(0);
            }
            if(count1 >= 300 || count2 >= 300 || count3 >= 300 || count4 >= 300)
            {
                exit(0);
            }
        }
    }
	}
	
#pragma mark Teardown
	{
    	clReleaseMemObject(x_mem);
    	clReleaseMemObject(y_mem);
    	clReleaseMemObject(z_mem);
    	clReleaseMemObject(xc_mem);
    	clReleaseMemObject(yc_mem);
    	clReleaseMemObject(zc_mem);
    	clReleaseMemObject(id_mem);
    	clReleaseMemObject(cell_type_mem);
    	clReleaseMemObject(ele_type_mem);
    	clReleaseMemObject(xF_mem);
    	clReleaseMemObject(yF_mem);
    	clReleaseMemObject(zF_mem);
    	clReleaseMemObject(ele_per_cell_mem);
    	clReleaseMemObject(flag_mem);
    	clReleaseMemObject(chem1_gpu_mem);
    	clReleaseMemObject(chem2_gpu_mem);
    	clReleaseMemObject(chem11_gpu_mem);
    	clReleaseMemObject(chem22_gpu_mem);
    	clReleaseMemObject(grid_gpu_mem);
    	clReleaseMemObject(grid1_gpu_mem);
    	clReleaseMemObject(grid2_gpu_mem);
    	clReleaseMemObject(grid3_gpu_mem);
        clReleaseMemObject(grid4_gpu_mem);
		
    	clReleaseCommandQueue(cmd_queue);
    	clReleaseContext(context);
	}
	return CL_SUCCESS;
}

// map cell to each PDE grid
void  cell_in_block(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell, float * xo, float * yo, float * zo, float * vx, float * vy, float * vz)
{
    int i, j, k, index;
    int ni, nj, nk;

    for(i = 0; i < NX; i++)
        for(j = 0; j < NY; j++)
            for(k = 0; k < NZ; k++)
    {
        grid[i][j][k] = 0;
        grid1[i][j][k] = 0;
        grid2[i][j][k] = 0;
        grid3[i][j][k] = 0;
        grid4[i][j][k] = 0;
        index = i + j*NX + k*NX*NY;
        vx[index] = 0.0;
        vy[index] = 0.0;
        vz[index] = 0.0;
    }

    for(i = 0; i < cell_no; i++)
    {
        for(j = 0; j < ele_per_cell[i]; j++)
        {
            k = i*MAX_ELE + j;
            ni = (int)(1.0*x[k]/(1.0*DX)); if(ni < 0) ni = 0; if(ni >= NX) ni = NX-1;
            nj = (int)(1.0*y[k]/(1.0*DY)); if(nj < 0) nj = 0; if(nj >= NY) nj = NY-1;
            nk = (int)(1.0*z[k]/(1.0*DZ)); if(nk < 0) nk = 0; if(nk >= NZ) nk = NZ-1;

            index = ni + nj*NX + nk*NX*NY;
            vx[index] += x[k] - xo[k];
            vy[index] += y[k] - yo[k];
            vz[index] += z[k] - zo[k];

            grid[ni][nj][nk]++;
            if(cell_type[i] == 1)
            {
                grid1[ni][nj][nk]++;
            }
            else if(cell_type[i] == 2)
            {
                grid2[ni][nj][nk]++;
            }
            else if(cell_type[i] == 3)
            {
                grid3[ni][nj][nk]++;
            }
            else if(cell_type[i] == 4)
            {
                grid4[ni][nj][nk]++;
            }
            else
            {
                printf("ERROR::cell_in_block(), wrong cell type = %d\n", cell_type[i]);
                exit(0);
            }
            xo[k] = x[k];
            yo[k] = y[k];
            zo[k] = z[k];
        }
    } 

    for(i = 0; i < NX; i++)
        for(j = 0; j < NY; j++)
            for(k = 0; k < NZ; k++)
    {
        index = i + j*NX + k*NX*NY;
        if(grid[i][j][k] == 0)
        {
            vx[index] = 0.0;
            vy[index] = 0.0;
            vz[index] = 0.0;
        }
        else
        {
            vx[index] /= 1.0*grid[i][j][k];
            vy[index] /= 1.0*grid[i][j][k];
            vz[index] /= 1.0*grid[i][j][k];
        }
    }
}

// initial condition of the chemical fields
void  initial_chem_paras(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell)
{
    int i, j, k;
    int ni, nj, nk;
    int i0, i1, j0, j1, k0, k1;
    float cc = 0.05;

    for(i = 0; i < NX; i++)
        for(j = 0; j < NY; j++)
            for(k = 0; k < NZ; k++)
    {
        grid[i][j][k] = 0;
        chem1[i][j][k] = 0.0;
        chem2[i][j][k] = 0.0;
    }

    return;
}

// determine the fate of each cell after division
void  cell_lineage(int id, float x, float y, float z, int ctype0, int * ctype1, int * ctype2, float * chem1_gpu, float * chem2_gpu, float * OVOL1, float * OVOL2, int flag_gener)
{
    int i;
    int ni, nj, nk;
    float cc, pp1, pp2;

    float alpha = 10000.0;

    ni = (int)(x/(1.0*DX)); if(ni < 0) ni = 0; if(ni >= NX) ni = NX-1;
    nj = (int)(y/(1.0*DY)); if(nj < 0) nj = 0; if(nj >= NY) nj = NY-1;
    nk = (int)(z/(1.0*DZ)); if(nk < 0) nk = 0; if(nk >= NZ) nk = NZ-1;

    i = ni + nj*NX + nk*NX*NY;
    cc = chem1_gpu[i]/1.0;
    if(ctype0 == 1)
    {
        float q1 = 0.15, q2 = 0.1;
        float tmpp = RNG();
        if(tmpp <= q1)
        {
            *ctype1 = 1;
            *ctype2 = 1;
        }
        else if(tmpp <= (q1+q2))
        {
            *ctype1 = 2;
            *ctype2 = 2;
        }
        else
        {
            *ctype1 = 1;
            *ctype2 = 2;
        }

    }
    else if(ctype0 == 2)
    {
        float p1 = 0.1 + 0.2f/(1.0f + 1000.0f*OVOL1[id]);
        float p2 = 0.5 + 0.2f/(1.0f + 200.0f*OVOL1[id]);
        float tmpp = RNG();
        if(tmpp < p1)
        {
            *ctype1 = 2;
            *ctype2 = 2;
        }
        else if(tmpp < (p1+p2))
        {
            *ctype1 = *ctype2 = 3;
        }
        else
        {
            *ctype1 = 2;
            *ctype2 = 3;
        }

    }
}


int main (int argc, const char * argv[]) {

    int i, j, k;
   
    char InName[] = "Input/IC";
    int max_ele_no = MAX_CELL*MAX_ELE; //DEFAULT_MAX_ELE_NO;
    int ele_no = 0;
    int cell_no = 0; 
    float dt = 0.1;

    int n = 100;
	
    float * x = (float *)malloc(max_ele_no*sizeof(float));
    float * y = (float *)malloc(max_ele_no*sizeof(float));
    float * z = (float *)malloc(max_ele_no*sizeof(float));
    int * id = (int *)malloc(max_ele_no*sizeof(int));
    int * ele_type = (int *)malloc(max_ele_no*sizeof(int));
    int * cell_type = (int *)malloc(MAX_CELL*sizeof(int));
    int * ele_per_cell = (int *)malloc(MAX_CELL*sizeof(int));
    int * cclock = (int *)malloc(MAX_CELL*sizeof(int));
    int * cycle = (int *)malloc(MAX_CELL*sizeof(int));

    chem1 = (float ***)malloc(NX*sizeof(float**));
    for(i = 0; i < NX; i++)
    {
        chem1[i] = (float **)malloc(NY*sizeof(float*));
        for(j = 0; j < NY; j++)
            chem1[i][j] = (float *)malloc(NZ*sizeof(float));
    }
    chem2 = (float ***)malloc(NX*sizeof(float**));
    for(i = 0; i < NX; i++)
    {
        chem2[i] = (float **)malloc(NY*sizeof(float*));
        for(j = 0; j < NY; j++)
            chem2[i][j] = (float *)malloc(NZ*sizeof(float));
    }
    chem_diff = (float ***)malloc(NX*sizeof(float**));
    for(i = 0; i < NX; i++)
    {
        chem_diff[i] = (float **)malloc(NY*sizeof(float*));
        for(j = 0; j < NY; j++)
            chem_diff[i][j] = (float *)malloc(NZ*sizeof(float));
    }
    grid = (int ***)malloc(NX*sizeof(int**));
    for(i = 0; i < NX; i++)
    {
        grid[i] = (int **)malloc(NY*sizeof(int*));
        for(j = 0; j < NY; j++)
            grid[i][j] = (int *)malloc(NZ*sizeof(int));
    }
    grid1 = (int ***)malloc(NX*sizeof(int**));
    for(i = 0; i < NX; i++)
    {
        grid1[i] = (int **)malloc(NY*sizeof(int*));
        for(j = 0; j < NY; j++)
            grid1[i][j] = (int *)malloc(NZ*sizeof(int));
    }
    grid2 = (int ***)malloc(NX*sizeof(int**));
    for(i = 0; i < NX; i++)
    {
        grid2[i] = (int **)malloc(NY*sizeof(int*));
        for(j = 0; j < NY; j++)
            grid2[i][j] = (int *)malloc(NZ*sizeof(int));
    }
    grid3 = (int ***)malloc(NX*sizeof(int**));
    for(i = 0; i < NX; i++)
    {
        grid3[i] = (int **)malloc(NY*sizeof(int*));
        for(j = 0; j < NY; j++)
            grid3[i][j] = (int *)malloc(NZ*sizeof(int));
    }
    grid4 = (int ***)malloc(NX*sizeof(int**));
    for(i = 0; i < NX; i++)
    {
        grid4[i] = (int **)malloc(NY*sizeof(int*));
        for(j = 0; j < NY; j++)
            grid4[i][j] = (int *)malloc(NZ*sizeof(int));
    }

	for(i=0;i<max_ele_no;i++)
    {
        x[i] = y[i] = z[i] = 0.0f;
        id[i] = ele_type[i] = 0;
    }
    for(i = 0; i < MAX_CELL; i++)
    {
        ele_per_cell[i] = 0;
        cell_type[i] = 0;
        cclock[i] = 0;
        cycle[i] = 0;
    }
    
    InitialReader(InName, x, y, z, id, cell_type, ele_type, &cell_no, &ele_no, ele_per_cell, cclock, cycle);
    initial_chem_paras(x, y, z, id, cell_type, ele_type, cell_no, ele_per_cell);

    // Do the OpenCL calculation
    runCL(x, y, z, id, cell_type, ele_type, dt, 
          max_ele_no, 
          ele_no, cell_no, ele_per_cell,
          cclock, cycle);

    // Free up memory
    free(x);
    free(y);
    free(z);
    free(id);
    free(cell_type);
    free(ele_type);
    free(ele_per_cell);

    return 0;
}
