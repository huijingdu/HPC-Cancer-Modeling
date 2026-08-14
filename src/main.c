#include <stdio.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

// an assert that tears down the whole job: a failure on one rank must not leave
// the other ranks spinning in a collective
#undef assert
#define assert(x) \
 do { if (!(x)) { \
 fprintf(stderr, "[MPI] Assertion failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); \
 MPI_Abort(MPI_COMM_WORLD, 1); \
 } } while (0)

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
#define g_dx_TARGET 1.0f
#define PAD 5.0f
#define CELL_GROWTH 2.0f
#define MAX_ELE 20

// room to expand into, as a fraction of the cell span on each side
#define DOMAIN_PAD_FRAC 0.5f

// slack added around the cells when the bin grid is refitted, so growth is rare
#define BIN_FIT_MARGIN (2.0f * BIN_SIZE)

// the chem grid is spaced at 1.0, so it tracks the cells only and never the
// padded bin box, whose cube would run to gigabytes
#define CHEM_MARGIN 8.0f

// how many times a rebuild may refit the grid before giving up
#define BIN_FIT_TRIES 4

#define BC 0
#define RATIO 1

#define GROWTH_FR 1000
#define DIVISION_FR 1000
#define OUTPUT_FR 100

#define MAX_ITERATIONS 1001
#define MAX_CHEM_ITER  1000
#define MAX_DEATH 60000

#define C1_MEAN 180.0 // slow-dividing doubles in 180 days
#define C1_STD 40.0 // standard deviation
#define C2_MEAN 4.0 // fast-dividing doubles in 4 days
#define C2_STD 1.0 // standard deviation
#define MAX_GENER 2

// ghost band width in X, equal to the movement cell-center cull radius, so a rank
// sees every partner that could act on a cell it owns
#define GHOST_DEPTH_F 20.0f

// ghosts carry only the partner fields movement.cl reads
#define GHOST_FLOATS_PER_CELL (3 * MAX_ELE + 3) // x,y,z blocks + xc,yc,zc
#define GHOST_INTS_PER_CELL 2 // ele_per_cell, cell_type

// migrants carry enough to fully reconstruct a cell on the receiving rank
#define MIG_FLOATS_PER_CELL (3 * MAX_ELE + 4) // x,y,z blocks + xc,yc,zc + OVOL1
#define MIG_INTS_PER_CELL (2 * MAX_ELE + 5) // id[], ele_type[], epc, ctype, flag_gener, cclock, cycle

// staging for one direction of an exchange. send_idx lists the local slots
// going out; recv holds what arrives from the opposite neighbour
typedef struct {
    int * send_idx;
    float * send_f;
    int * send_i;
    float * recv_f;
    int * recv_i;
} xfer_buf;

static void xfer_alloc(xfer_buf * h, int max_cells, int floats_per_cell, int ints_per_cell)
{
    h->send_idx = (int *)malloc(max_cells * sizeof(int));
    h->send_f = (float *)malloc((size_t)max_cells * floats_per_cell * sizeof(float));
    h->send_i = (int *)malloc((size_t)max_cells * ints_per_cell * sizeof(int));
    h->recv_f = (float *)malloc((size_t)max_cells * floats_per_cell * sizeof(float));
    h->recv_i = (int *)malloc((size_t)max_cells * ints_per_cell * sizeof(int));
    assert(h->send_idx && h->send_f && h->send_i && h->recv_f && h->recv_i);
}

static void xfer_free(xfer_buf * h)
{
    free(h->send_idx);
    free(h->send_f);
    free(h->send_i);
    free(h->recv_f);
    free(h->recv_i);
}

// copy one local cell into the outgoing halo record at slot s
static void halo_pack(xfer_buf * h, int s, int ci, float * x, float * y, float * z,
                      float * xc, float * yc, float * zc, int * ele_per_cell, int * cell_type)
{
    float * fp = h->send_f + (size_t)s * GHOST_FLOATS_PER_CELL;
    int * ip = h->send_i + (size_t)s * GHOST_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) fp[j] = x[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[MAX_ELE + j] = y[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[2 * MAX_ELE + j] = z[ci * MAX_ELE + j];
    fp[3 * MAX_ELE + 0] = xc[ci];
    fp[3 * MAX_ELE + 1] = yc[ci];
    fp[3 * MAX_ELE + 2] = zc[ci];

    ip[0] = ele_per_cell[ci];
    ip[1] = cell_type[ci];
}

// place an arrived halo record into local cell slot ci
static void halo_unpack(xfer_buf * h, int r, int ci, float * x, float * y, float * z,
                        float * xc, float * yc, float * zc, int * ele_per_cell, int * cell_type)
{
    float * fp = h->recv_f + (size_t)r * GHOST_FLOATS_PER_CELL;
    int * ip = h->recv_i + (size_t)r * GHOST_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) x[ci * MAX_ELE + j] = fp[j];
    for (int j = 0; j < MAX_ELE; j++) y[ci * MAX_ELE + j] = fp[MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) z[ci * MAX_ELE + j] = fp[2 * MAX_ELE + j];
    xc[ci] = fp[3 * MAX_ELE + 0];
    yc[ci] = fp[3 * MAX_ELE + 1];
    zc[ci] = fp[3 * MAX_ELE + 2];

    ele_per_cell[ci] = ip[0];
    cell_type[ci] = ip[1];
}

// host cell arrays grouped so the migration helpers stay readable
typedef struct {
    float * x;
    float * y;
    float * z;
    float * xc;
    float * yc;
    float * zc;
    float * OVOL1;
    int * id;
    int * ele_type;
    int * ele_per_cell;
    int * cell_type;
    int * flag_gener;
    int * cclock;
    int * cycle;
} cell_arrays;

// a migrant carries everything needed to rebuild the cell on the receiving rank
static void mig_pack(xfer_buf * h, int s, int ci, cell_arrays * a)
{
    float * fp = h->send_f + (size_t)s * MIG_FLOATS_PER_CELL;
    int * ip = h->send_i + (size_t)s * MIG_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) fp[j] = a->x[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[MAX_ELE + j] = a->y[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[2 * MAX_ELE + j] = a->z[ci * MAX_ELE + j];
    fp[3 * MAX_ELE + 0] = a->xc[ci];
    fp[3 * MAX_ELE + 1] = a->yc[ci];
    fp[3 * MAX_ELE + 2] = a->zc[ci];
    fp[3 * MAX_ELE + 3] = a->OVOL1[ci];

    for (int j = 0; j < MAX_ELE; j++) ip[j] = a->id[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) ip[MAX_ELE + j] = a->ele_type[ci * MAX_ELE + j];
    ip[2 * MAX_ELE + 0] = a->ele_per_cell[ci];
    ip[2 * MAX_ELE + 1] = a->cell_type[ci];
    ip[2 * MAX_ELE + 2] = a->flag_gener[ci];
    ip[2 * MAX_ELE + 3] = a->cclock[ci];
    ip[2 * MAX_ELE + 4] = a->cycle[ci];
}

// id[] is rewritten to the new slot, not copied: cellcenter.cl needs (id[k] - 1) == slot
static void mig_unpack(xfer_buf * h, int r, int slot, cell_arrays * a)
{
    float * fp = h->recv_f + (size_t)r * MIG_FLOATS_PER_CELL;
    int * ip = h->recv_i + (size_t)r * MIG_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) a->x[slot * MAX_ELE + j] = fp[j];
    for (int j = 0; j < MAX_ELE; j++) a->y[slot * MAX_ELE + j] = fp[MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) a->z[slot * MAX_ELE + j] = fp[2 * MAX_ELE + j];
    a->xc[slot] = fp[3 * MAX_ELE + 0];
    a->yc[slot] = fp[3 * MAX_ELE + 1];
    a->zc[slot] = fp[3 * MAX_ELE + 2];
    a->OVOL1[slot] = fp[3 * MAX_ELE + 3];

    for (int j = 0; j < MAX_ELE; j++) a->id[slot * MAX_ELE + j] = slot + 1;
    for (int j = 0; j < MAX_ELE; j++) a->ele_type[slot * MAX_ELE + j] = ip[MAX_ELE + j];
    a->ele_per_cell[slot] = ip[2 * MAX_ELE + 0];
    a->cell_type[slot] = ip[2 * MAX_ELE + 1];
    a->flag_gener[slot] = ip[2 * MAX_ELE + 2];
    a->cclock[slot] = ip[2 * MAX_ELE + 3];
    a->cycle[slot] = ip[2 * MAX_ELE + 4];
}

static int g_nx, g_ny, g_nz;
static int g_nbx, g_nby, g_nbz, g_nbins;
static int g_max_cell;
static float g_lx, g_ly, g_lz;
static float g_dx, g_dy, g_dz;

// grids index from 0, so the cloud is shifted at load and shifted back on output
static float g_shift_x, g_shift_y, g_shift_z;

// occupied x range after the shift, excluding padding
static float g_cell_lo_x, g_cell_hi_x;

// lower corner of bin (0,0,0). the bin grid is only (L/BIN_SIZE)^3 ints, so it
// is refitted around the cells whenever they drift out of it
static float g_bin_ox, g_bin_oy, g_bin_oz;

// chem grid corner and extent, kept independent of the bin box
static float g_chem_ox, g_chem_oy, g_chem_oz;
static float g_chem_lx, g_chem_ly, g_chem_lz;

// refits the bin grid around [lo, hi] plus a margin. returns the new bin count.
static int fit_bin_grid(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z)
{
    g_bin_ox = lo_x - BIN_FIT_MARGIN;
    g_bin_oy = lo_y - BIN_FIT_MARGIN;
    g_bin_oz = lo_z - BIN_FIT_MARGIN;

    g_nbx = (int)ceilf((hi_x + BIN_FIT_MARGIN - g_bin_ox) / BIN_SIZE);
    g_nby = (int)ceilf((hi_y + BIN_FIT_MARGIN - g_bin_oy) / BIN_SIZE);
    g_nbz = (int)ceilf((hi_z + BIN_FIT_MARGIN - g_bin_oz) / BIN_SIZE);

    if (g_nbx < 1) g_nbx = 1;
    if (g_nby < 1) g_nby = 1;
    if (g_nbz < 1) g_nbz = 1;

    g_nbins = g_nbx * g_nby * g_nbz;
    return g_nbins;
}

// MPI X-split decomposition state
static float g_x_lo, g_x_hi; // this rank's X band, [x_lo, x_hi)
static int g_x_neighbour_left; // rank at smaller x, or MPI_PROC_NULL
static int g_x_neighbour_right; // rank at larger x, or MPI_PROC_NULL
static int g_rank, g_nprocs;

// equal-width X bands over the occupied range; splitting the padded domain would
// hand the end ranks mostly empty bands
static void assign_x_band(void)
{
    float occupied = g_cell_hi_x - g_cell_lo_x;
    float band = occupied / (float)g_nprocs;

    // halos only reach the immediate neighbours and migration only hands a cell one
    // band over, so a band narrower than the cull radius would silently lose forces
    // and drop cells that cross two bands in a step
    if (g_nprocs > 1 && band < GHOST_DEPTH_F)
    {
        if (g_rank == 0)
            fprintf(stderr, "[mpi] %d ranks gives an X band of %.2f, below the %.2f halo depth. "
                            "use at most %d ranks for an occupied width of %.2f\n",
                    g_nprocs, band, (float)GHOST_DEPTH_F,
                    (int)(occupied / GHOST_DEPTH_F), occupied);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    g_x_lo = g_cell_lo_x + band * (float)g_rank;
    g_x_hi = g_cell_lo_x + band * (float)(g_rank + 1);

    // the end bands stay open so anything drifting into the padding is still owned
    if (g_rank == 0) g_x_lo = 0.0f;
    if (g_rank == g_nprocs - 1) g_x_hi = g_lx;
}

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

void sniff_ic(const char InName[], int *cell_no_out, float *mx_out, float *my_out, float *mz_out,
              float *nx_out, float *ny_out, float *nz_out) {
    FILE *fp = fopen(InName, "r");
    if (!fp) {
        printf("sniff_ic: cannot open %s\n", InName);
        exit(1);
    }
    
    int temp_cell_no = 0;
    if (fscanf(fp, "%d", &temp_cell_no) != 1) {
        printf("sniff_ic: bad header in %s\n", InName);
        exit(1);
    }

    int actual = 0;
    int seen = 0;
    float mx = 0.0f, my = 0.0f, mz = 0.0f;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float px, py, pz, pxc, pyc, pzc;
    int pid, ptype, pele, ptime, pcyc;

    for (int i = 0; i < temp_cell_no; i++) {
        fscanf(fp, "%d %d %d %f %f %f %d %d\n",
                &pid, &pele, &ptype, &pxc, &pyc, &pzc, &ptime, &pcyc);
        if (pele <= 0) continue;
        for (int j = 0; j < pele; j++) {
            fscanf(fp, "%f %f %f", &px, &py, &pz);
            // seed from the first point; seeding at 0 is what hid the negative side
            if (!seen) {
                mx = nx = px;
                my = ny = py;
                mz = nz = pz;
                seen = 1;
                continue;
            }
            if (px > mx) mx = px;
            if (py > my) my = py;
            if (pz > mz) mz = pz;
            if (px < nx) nx = px;
            if (py < ny) ny = py;
            if (pz < nz) nz = pz;
        }
        actual++;
    }
    fclose(fp);

    if (actual == 0) {
        printf("sniff_ic: %s contains no non-empty cells\n", InName);
        exit(1);
    }

    *cell_no_out = actual;
    *mx_out = mx;
    *my_out = my;
    *mz_out = mz;
    *nx_out = nx;
    *ny_out = ny;
    *nz_out = nz;
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
// Every rank formats and writes its own cells
static double g_output_io = 0.0;
static double g_output_comm = 0.0;

void output_mpi(float * x, float * y, float * z, float * xc, float * yc, float * zc, int * id, int * cell_type, int * ele_type, int cell_no, int ele_no, int * ele_per_cell, int step, float * chem1_gpu, float * chem2_gpu, int * cclock, int * cycle, int * flag_gener, float * vx, float * vy, float * vz)
{
    // the cells this rank writes
    int * out = (int *)malloc((cell_no > 0 ? cell_no : 1) * sizeof(int));
    assert(out);
    int my_cells = 0, my_ele = 0;

    for (int i = 0; i < cell_no; i++)
    {
        if (ele_per_cell[i] == 0) continue;

        int skip = 0;
        for (int j = 0; j < ele_per_cell[i]; j++)
            if (z[i * MAX_ELE + j] > g_lz + 2.0) skip = 1;
        if (skip) continue;

        out[my_cells++] = i;
        my_ele += ele_per_cell[i];
    }

    double t0 = MPI_Wtime();
    long long mine[2] = { my_cells, my_ele };
    long long base[2] = { 0, 0 };
    MPI_Exscan(mine, base, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) base[0] = base[1] = 0;
    g_output_comm += MPI_Wtime() - t0;

    int gcell = (int)base[0];
    int gatom = (int)base[1];

    // format into a local buffer, about 96 bytes an ATOM line
    t0 = MPI_Wtime();
    size_t cap = (size_t)my_ele * 96 + 1024;
    char * buf = (char *)malloc(cap);
    assert(buf);
    size_t n = 0;

    for (int s = 0; s < my_cells; s++)
    {
        int i = out[s];
        for (int j = 0; j < ele_per_cell[i]; j++)
        {
            int k = i * MAX_ELE + j;
            if (n + 256 > cap)
            {
                cap *= 2;
                buf = (char *)realloc(buf, cap);
                assert(buf);
            }
            // written in the input frame, not the shifted one
            n += (size_t)snprintf(buf + n, cap - n,
                    "ATOM %6d C  THR   %5d      % .3f % .3f % .3f % .4f % .4f\n",
                    gatom + 1, (gcell + 1) + cell_type[i] * 1000,
                    x[k] - g_shift_x, y[k] - g_shift_y, z[k] - g_shift_z,
                    0.1 * ele_type[k], 1.5);
            gatom++;
        }
        gcell++;
    }
    free(out);
    g_output_io += MPI_Wtime() - t0;

    // byte offset of this rank's slice, and the exact size of the finished file
    t0 = MPI_Wtime();
    MPI_Offset local = (MPI_Offset)n;
    MPI_Offset offset = 0;
    MPI_Offset total = 0;
    MPI_Exscan(&local, &offset, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) offset = 0;
    MPI_Allreduce(&local, &total, 1, MPI_OFFSET, MPI_SUM, MPI_COMM_WORLD);

    // the directory has to exist before the collective open
    if (g_rank == 0)
    {
        struct stat st;
        if (stat("PQR", &st) == -1 && errno == ENOENT)
            MKDIR("PQR");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    char filename[64];
    sprintf(filename, "PQR/data.%d.pqr", step);

    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                      MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    {
        fprintf(stderr, "[rank %d] MPI_File_open(%s) failed\n", g_rank, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_File_set_size(fh, total); // trim a stale longer file to this run's size
    g_output_comm += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_File_write_at_all(fh, offset, buf, (int)n, MPI_CHAR, MPI_STATUS_IGNORE);
    g_output_io += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    MPI_File_close(&fh);
    g_output_comm += MPI_Wtime() - t0;

    free(buf);
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
          int * cclock, int * cycle,
          double *out_setup, double *out_loop, double *out_comm,
          double *out_io)
{
    double t_func_start = MPI_Wtime();

    // t_comm accumulates the wall time spent inside MPI calls in the loop
    double t_comm = 0.0;
    double t_comm0 = 0.0;

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
    float * xc = (float *)malloc(g_max_cell*sizeof(float));
    float * yc = (float *)malloc(g_max_cell*sizeof(float));
    float * zc = (float *)malloc(g_max_cell*sizeof(float));
    int * flag = (int *)malloc(g_max_cell*sizeof(int));
    float * rand_no = (float *)malloc(max_ele_no*sizeof(float));
    float * chem1_gpu = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * chem2_gpu = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * chem11_gpu = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * chem22_gpu = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    int * grid_gpu = (int *)malloc(g_nx*g_ny*g_nz*sizeof(int));
    int * grid1_gpu = (int *)malloc(g_nx*g_ny*g_nz*sizeof(int));
    int * grid2_gpu = (int *)malloc(g_nx*g_ny*g_nz*sizeof(int));
    int * grid3_gpu = (int *)malloc(g_nx*g_ny*g_nz*sizeof(int));
    int * grid4_gpu = (int *)malloc(g_nx*g_ny*g_nz*sizeof(int));
    int * flag_death = (int *)malloc(g_max_cell*sizeof(int));
    int * flag_gener = (int *)malloc(g_max_cell*sizeof(int));
    float * OVOL1 = (float *)malloc(g_max_cell*sizeof(float));
    float * OVOL2 = (float *)malloc(g_max_cell*sizeof(float));
    float * vx = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * vy = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * vz = (float *)malloc(g_nx*g_ny*g_nz*sizeof(float));
    float * xo = (float *)malloc(max_ele_no*sizeof(float));
    float * yo = (float *)malloc(max_ele_no*sizeof(float));
    float * zo = (float *)malloc(max_ele_no*sizeof(float));

    // Bin realated declarations
    int * bin_count_host = (int *)malloc(g_nbins * sizeof(int));
    int * bin_offset_host = (int *)malloc((g_nbins + 1) * sizeof(int));
    int * cell_list_host = (int *)malloc(g_max_cell * sizeof(int));


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
    cl_mem bin_count_mem, bin_offset_mem, cell_list_mem;
    cl_mem bin_bounds_mem, bin_overflow_mem;

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
            cpPlatform = plats[i];
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

    // Bin buffer allocation (requires context)
    bin_count_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, g_nbins * sizeof(int), NULL, &err);
    bin_offset_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, (g_nbins + 1) * sizeof(int), NULL, &err);
    cell_list_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, g_max_cell * sizeof(int), NULL, &err);
    // occupied extent and out-of-grid tally, read back once per rebuild
    bin_bounds_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, 6 * sizeof(int), NULL, &err);
    bin_overflow_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &err);
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

        kernel[8] = clCreateKernel(program[8], "bin_count", &err);
        assert(err == CL_SUCCESS);
        kernel[9] = clCreateKernel(program[8], "bin_prefix_sum", &err);
        assert(err == CL_SUCCESS);
        kernel[10] = clCreateKernel(program[8], "bin_scatter", &err);
        assert(err == CL_SUCCESS);
        kernel[11] = clCreateKernel(program[8], "cell_bounds", &err);
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
		
    	buffer_size = sizeof(int) * g_max_cell;
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

    	buffer_size = sizeof(float) * g_max_cell;
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

        buffer_size = sizeof(float) * g_nx*g_ny*g_nz;
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

        buffer_size = sizeof(int) * g_nx*g_ny*g_nz;
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
    	err |= clSetKernelArg(kernel[0], 15, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[0], 16, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[0], 17, sizeof(float), &g_lz);
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
    	err |= clSetKernelArg(kernel[1],  8, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[1],  9, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[1], 10, sizeof(float), &g_lz);
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
    	err |= clSetKernelArg(kernel[4],  9, sizeof(float), &g_lx);
    	err |= clSetKernelArg(kernel[4], 10, sizeof(float), &g_ly);
    	err |= clSetKernelArg(kernel[4], 11, sizeof(float), &g_lz);
        assert(err == CL_SUCCESS);
    }

    {
        int nx0 = g_nx, ny0 = g_ny, nz0 = g_nz, tmpii = 0;
        float dx0 = g_dx, dy0 = g_dy, dz0 = g_dz;
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
        assert(err == CL_SUCCESS);
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
        err |= clSetKernelArg(kernel[7], 11, sizeof(int), &g_nx);
        err |= clSetKernelArg(kernel[7], 12, sizeof(int), &g_ny);
        err |= clSetKernelArg(kernel[7], 13, sizeof(int), &g_nz);
        err |= clSetKernelArg(kernel[7], 14, sizeof(float), &g_chem_lx);
        err |= clSetKernelArg(kernel[7], 15, sizeof(float), &g_chem_ly);
        err |= clSetKernelArg(kernel[7], 16, sizeof(float), &g_chem_lz);
        assert(err == CL_SUCCESS);
    }
    
    {
        const float bin_size_f = BIN_SIZE;
        const int nbx_i = g_nbx, nby_i = g_nby, nbz_i = g_nbz, nbins_i = g_nbins;

        // kernel[8] bin_count
        err = clSetKernelArg(kernel[8], 0, sizeof(cl_mem), &xc_mem);
        err |= clSetKernelArg(kernel[8], 1, sizeof(cl_mem), &yc_mem);
        err |= clSetKernelArg(kernel[8], 2, sizeof(cl_mem), &zc_mem);
        err |= clSetKernelArg(kernel[8], 3, sizeof(cl_mem), &ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[8], 5, sizeof(float), &bin_size_f);
        err |= clSetKernelArg(kernel[8], 6, sizeof(int), &nbx_i);
        err |= clSetKernelArg(kernel[8], 7, sizeof(int), &nby_i);
        err |= clSetKernelArg(kernel[8], 8, sizeof(int), &nbz_i);
        err |= clSetKernelArg(kernel[8], 9, sizeof(cl_mem), &bin_count_mem);
        err |= clSetKernelArg(kernel[8], 10, sizeof(float), &g_bin_ox);
        err |= clSetKernelArg(kernel[8], 11, sizeof(float), &g_bin_oy);
        err |= clSetKernelArg(kernel[8], 12, sizeof(float), &g_bin_oz);
        err |= clSetKernelArg(kernel[8], 13, sizeof(cl_mem), &bin_overflow_mem);

        // kernel[11] cell_bounds, launched only when the grid needs a refit
        err |= clSetKernelArg(kernel[11], 0, sizeof(cl_mem), &xc_mem);
        err |= clSetKernelArg(kernel[11], 1, sizeof(cl_mem), &yc_mem);
        err |= clSetKernelArg(kernel[11], 2, sizeof(cl_mem), &zc_mem);
        err |= clSetKernelArg(kernel[11], 3, sizeof(cl_mem), &ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[11], 5, sizeof(cl_mem), &bin_bounds_mem);

        // kernel[9] bin_prefix_sum
        err |= clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &bin_count_mem);
        err |= clSetKernelArg(kernel[9], 1, sizeof(int), &nbins_i);
        err |= clSetKernelArg(kernel[9], 2, sizeof(cl_mem), &bin_offset_mem);

        // kernel[10] bin_scatter
        err |= clSetKernelArg(kernel[10], 0, sizeof(cl_mem), &xc_mem);
        err |= clSetKernelArg(kernel[10], 1, sizeof(cl_mem), &yc_mem);
        err |= clSetKernelArg(kernel[10], 2, sizeof(cl_mem), &zc_mem);
        err |= clSetKernelArg(kernel[10], 3, sizeof(cl_mem), &ele_per_cell_mem);
        // arg 4 (cell_no) set per step
        err |= clSetKernelArg(kernel[10], 5, sizeof(float), &bin_size_f);
        err |= clSetKernelArg(kernel[10], 6, sizeof(int), &nbx_i);
        err |= clSetKernelArg(kernel[10], 7, sizeof(int), &nby_i);
        err |= clSetKernelArg(kernel[10], 8, sizeof(int), &nbz_i);
        err |= clSetKernelArg(kernel[10], 9, sizeof(cl_mem), &bin_count_mem);
        err |= clSetKernelArg(kernel[10], 10, sizeof(cl_mem), &cell_list_mem);
        err |= clSetKernelArg(kernel[10], 11, sizeof(float), &g_bin_ox);
        err |= clSetKernelArg(kernel[10], 12, sizeof(float), &g_bin_oy);
        err |= clSetKernelArg(kernel[10], 13, sizeof(float), &g_bin_oz);

        assert(err == CL_SUCCESS);

        // Bin related arguments
        err = clSetKernelArg(kernel[0], 21, sizeof(cl_mem), &cell_list_mem);
        err |= clSetKernelArg(kernel[0], 22, sizeof(cl_mem), &bin_offset_mem);
        err |= clSetKernelArg(kernel[0], 23, sizeof(float), &bin_size_f);
        err |= clSetKernelArg(kernel[0], 24, sizeof(int), &nbx_i);
        err |= clSetKernelArg(kernel[0], 25, sizeof(int), &nby_i);
        err |= clSetKernelArg(kernel[0], 26, sizeof(int), &nbz_i);
        err |= clSetKernelArg(kernel[0], 27, sizeof(float), &g_bin_ox);
        err |= clSetKernelArg(kernel[0], 28, sizeof(float), &g_bin_oy);
        err |= clSetKernelArg(kernel[0], 29, sizeof(float), &g_bin_oz);
        assert(err == CL_SUCCESS);
    }

#pragma imark Execution and Read

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
    err = clEnqueueNDRangeKernel(cmd_queue, kernel[4], 1, NULL,
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

    cell_arrays arrays;
    arrays.x = x; arrays.y = y; arrays.z = z;
    arrays.xc = xc; arrays.yc = yc; arrays.zc = zc;
    arrays.OVOL1 = OVOL1;
    arrays.id = id; arrays.ele_type = ele_type;
    arrays.ele_per_cell = ele_per_cell; arrays.cell_type = cell_type;
    arrays.flag_gener = flag_gener; arrays.cclock = cclock; arrays.cycle = cycle;

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

    // start of time iterations
    int iterations = MAX_ITERATIONS;
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
                err = clSetKernelArg(kernel[5], 15, sizeof(int), &(iter_chem));
                err |= clEnqueueNDRangeKernel(cmd_queue, kernel[5], 1, NULL,
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
                halo_pack(&halo_left, s, halo_left.send_idx[s], x, y, z, xc, yc, zc,
                          ele_per_cell, cell_type);
            for (int s = 0; s < n_send_right; s++)
                halo_pack(&halo_right, s, halo_right.send_idx[s], x, y, z, xc, yc, zc,
                          ele_per_cell, cell_type);

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
                halo_unpack(&halo_left, r, local_cell_no + r, x, y, z, xc, yc, zc,
                            ele_per_cell, cell_type);
            for (int r = 0; r < n_recv_right; r++)
                halo_unpack(&halo_right, r, local_cell_no + n_recv_left + r, x, y, z, xc, yc, zc,
                            ele_per_cell, cell_type);

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
        size_t one = 1;
        int zero_int = 0;

        err = clSetKernelArg(kernel[8], 4, sizeof(int), &combined_cell_no); // bin_count cell_no
        err |= clSetKernelArg(kernel[10], 4, sizeof(int), &combined_cell_no); // bin_scatter cell_no
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
            err |= clEnqueueFillBuffer(cmd_queue, bin_count_mem, &zero_int, sizeof(int),
                                       0, g_nbins * sizeof(int), 0, NULL, NULL);
            err |= clEnqueueNDRangeKernel(cmd_queue, kernel[8], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
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
            err |= clSetKernelArg(kernel[11], 4, sizeof(int), &combined_cell_no);
            err |= clEnqueueNDRangeKernel(cmd_queue, kernel[11], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
            err |= clEnqueueReadBuffer(cmd_queue, bin_bounds_mem, CL_TRUE, 0,
                                       sizeof(b), b, 0, NULL, NULL);
            assert(err == CL_SUCCESS);

            int old_nbins = g_nbins;
            fit_bin_grid((float)b[0], (float)b[1] + 1.0f, (float)b[2], (float)b[3] + 1.0f,
                         (float)b[4], (float)b[5] + 1.0f);

            if (g_nbins > old_nbins)
            {
                clReleaseMemObject(bin_count_mem);
                clReleaseMemObject(bin_offset_mem);
                bin_count_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               g_nbins * sizeof(int), NULL, &err);
                bin_offset_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                (g_nbins + 1) * sizeof(int), NULL, &err);
                assert(err == CL_SUCCESS);

                free(bin_count_host);
                free(bin_offset_host);
                bin_count_host = (int *)malloc(g_nbins * sizeof(int));
                bin_offset_host = (int *)malloc((g_nbins + 1) * sizeof(int));
                assert(bin_count_host && bin_offset_host);

                err = clSetKernelArg(kernel[8], 9, sizeof(cl_mem), &bin_count_mem);
                err |= clSetKernelArg(kernel[9], 0, sizeof(cl_mem), &bin_count_mem);
                err |= clSetKernelArg(kernel[9], 2, sizeof(cl_mem), &bin_offset_mem);
                err |= clSetKernelArg(kernel[10], 9, sizeof(cl_mem), &bin_count_mem);
                err |= clSetKernelArg(kernel[0], 22, sizeof(cl_mem), &bin_offset_mem);
                assert(err == CL_SUCCESS);
            }

            err = clSetKernelArg(kernel[9], 1, sizeof(int), &g_nbins);
            err |= clSetKernelArg(kernel[8], 6, sizeof(int), &g_nbx);
            err |= clSetKernelArg(kernel[8], 7, sizeof(int), &g_nby);
            err |= clSetKernelArg(kernel[8], 8, sizeof(int), &g_nbz);
            err |= clSetKernelArg(kernel[8], 10, sizeof(float), &g_bin_ox);
            err |= clSetKernelArg(kernel[8], 11, sizeof(float), &g_bin_oy);
            err |= clSetKernelArg(kernel[8], 12, sizeof(float), &g_bin_oz);
            err |= clSetKernelArg(kernel[10], 6, sizeof(int), &g_nbx);
            err |= clSetKernelArg(kernel[10], 7, sizeof(int), &g_nby);
            err |= clSetKernelArg(kernel[10], 8, sizeof(int), &g_nbz);
            err |= clSetKernelArg(kernel[10], 11, sizeof(float), &g_bin_ox);
            err |= clSetKernelArg(kernel[10], 12, sizeof(float), &g_bin_oy);
            err |= clSetKernelArg(kernel[10], 13, sizeof(float), &g_bin_oz);
            err |= clSetKernelArg(kernel[0], 24, sizeof(int), &g_nbx);
            err |= clSetKernelArg(kernel[0], 25, sizeof(int), &g_nby);
            err |= clSetKernelArg(kernel[0], 26, sizeof(int), &g_nbz);
            err |= clSetKernelArg(kernel[0], 27, sizeof(float), &g_bin_ox);
            err |= clSetKernelArg(kernel[0], 28, sizeof(float), &g_bin_oy);
            err |= clSetKernelArg(kernel[0], 29, sizeof(float), &g_bin_oz);
            assert(err == CL_SUCCESS);
        }

        err = clEnqueueNDRangeKernel(cmd_queue, kernel[9], 1, NULL, &one, NULL, 0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(cmd_queue, kernel[10], 1, NULL, &gws_cells, NULL, 0, NULL, NULL);
        assert(err == CL_SUCCESS);

        // cell movement. ghosts are integrated too and their results discarded,
        // which keeps the dispatch one contiguous range
        int combined_ele_no = combined_cell_no * MAX_ELE;
        err = clSetKernelArg(kernel[0],  5, sizeof(int), &combined_cell_no);
        err |= clSetKernelArg(kernel[0],  6, sizeof(int), &combined_ele_no);
        assert(err == CL_SUCCESS);
        // global_work_size = max_ele_no;
        global_work_size = combined_cell_no * MAX_ELE;
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
	        buffer_size = sizeof(int) * g_max_cell;
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
                    ctype2 = (RNG() < R_Diff)? 2 : 1;
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

        // update cell center. local only: ghost slots have the sender's id[], which
        // cellcenter.cl would reject by poisoning ele_per_cell with -1
        size_t global_work_size = local_cell_no;
        err = clEnqueueNDRangeKernel(cmd_queue, kernel[4], 1, NULL,
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
                mig_pack(&mig_left, s, mig_left.send_idx[s], &arrays);
            for (int s = 0; s < n_mig_right; s++)
                mig_pack(&mig_right, s, mig_right.send_idx[s], &arrays);

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

                mig_unpack(src, rr, slot, &arrays);

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
        // chem field halo. off while chem_rk_gpu, growth, division and death are all
        // #if 0, so chem1/chem2 stay zero and this would only shuffle zeros. turning
        // it on first needs the 3D chem field partitioned in X, one slab per rank;
        // right now every rank holds the whole field.
        {
            int nyz = g_ny * g_nz;
            float * send_lo = (float *)malloc(nyz * sizeof(float));
            float * send_hi = (float *)malloc(nyz * sizeof(float));
            float * recv_lo = (float *)malloc(nyz * sizeof(float));
            float * recv_hi = (float *)malloc(nyz * sizeof(float));

            int i_lo = (int)((g_x_lo - g_chem_ox) / g_dx);
            int i_hi = (int)((g_x_hi - g_chem_ox) / g_dx) - 1;

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
            count0 = 0, count1 = 0, count2 = 0, count3 = 0, count4 = 0;
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

    *out_setup      = t_loop0 - t_func_start;
    *out_loop       = t_loop_end - t_loop0;
    *out_comm       = t_comm + g_output_comm;
    *out_io         = g_output_io;

    xfer_free(&halo_left);
    xfer_free(&halo_right);
    xfer_free(&mig_left);
    xfer_free(&mig_right);
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

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
    {
        grid[i][j][k] = 0;
        grid1[i][j][k] = 0;
        grid2[i][j][k] = 0;
        grid3[i][j][k] = 0;
        grid4[i][j][k] = 0;
        index = i + j*g_nx + k*g_nx*g_ny;
        vx[index] = 0.0;
        vy[index] = 0.0;
        vz[index] = 0.0;
    }

    for(i = 0; i < cell_no; i++)
    {
        for(j = 0; j < ele_per_cell[i]; j++)
        {
            k = i*MAX_ELE + j;
            ni = (int)((x[k] - g_chem_ox)/(1.0*g_dx)); if(ni < 0) ni = 0; if(ni >= g_nx) ni = g_nx-1;
            nj = (int)((y[k] - g_chem_oy)/(1.0*g_dy)); if(nj < 0) nj = 0; if(nj >= g_ny) nj = g_ny-1;
            nk = (int)((z[k] - g_chem_oz)/(1.0*g_dz)); if(nk < 0) nk = 0; if(nk >= g_nz) nk = g_nz-1;

            index = ni + nj*g_nx + nk*g_nx*g_ny;
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

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
    {
        index = i + j*g_nx + k*g_nx*g_ny;
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

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
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

    ni = (int)((x - g_chem_ox)/(1.0*g_dx)); if(ni < 0) ni = 0; if(ni >= g_nx) ni = g_nx-1;
    nj = (int)((y - g_chem_oy)/(1.0*g_dy)); if(nj < 0) nj = 0; if(nj >= g_ny) nj = g_ny-1;
    nk = (int)((z - g_chem_oz)/(1.0*g_dz)); if(nk < 0) nk = 0; if(nk >= g_nz) nk = g_nz-1;

    i = ni + nj*g_nx + nk*g_nx*g_ny;
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

// reports elements sitting outside the bin grid. the grid refits itself during the
// run, so this is a note that the initial padding was tight, not a failure
void verifyDomain(float * x, float * y, float * z, int max_ele_no) {
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
    assign_x_band();

    if (g_rank == 0) printf("[mpi] %d rank(s), X band %.2f over occupied [%.2f, %.2f) of lx %.2f\n",
                            g_nprocs, (g_cell_hi_x - g_cell_lo_x) / (float)g_nprocs,
                            g_cell_lo_x, g_cell_hi_x, g_lx);
    if (g_rank == 0) printf("[grid] bins %dx%dx%d = %d, chem %dx%dx%d = %d\n",
                            g_nbx, g_nby, g_nbz, g_nbins, g_nx, g_ny, g_nz, g_nx*g_ny*g_nz);
    printf("[rank %d] x band [%.2f, %.2f) left=%d right=%d\n",
           g_rank, g_x_lo, g_x_hi, g_x_neighbour_left, g_x_neighbour_right);

    g_max_cell = (int)(sniff_cell_no * CELL_GROWTH);

    int max_ele_no = g_max_cell * MAX_ELE;
    int ele_no = 0;
    int cell_no = 0;
    float dt = 0.1;

    int n = 100;
	
    float * x = (float *)malloc(max_ele_no*sizeof(float));
    float * y = (float *)malloc(max_ele_no*sizeof(float));
    float * z = (float *)malloc(max_ele_no*sizeof(float));
    int * id = (int *)malloc(max_ele_no*sizeof(int));
    int * ele_type = (int *)malloc(max_ele_no*sizeof(int));
    int * cell_type = (int *)malloc(g_max_cell*sizeof(int));
    int * ele_per_cell = (int *)malloc(g_max_cell*sizeof(int));
    int * cclock = (int *)malloc(g_max_cell*sizeof(int));
    int * cycle = (int *)malloc(g_max_cell*sizeof(int));

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
        x[i] = y[i] = z[i] = 0.0f;
        id[i] = ele_type[i] = 0;
    }
    for(i = 0; i < g_max_cell; i++)
    {
        ele_per_cell[i] = 0;
        cell_type[i] = 0;
        cclock[i] = 0;
        cycle[i] = 0;
    }
    
    InitialReader(InName, x, y, z, id, cell_type, ele_type, &cell_no, &ele_no, ele_per_cell, cclock, cycle);

    // shift into the padded domain before anything bins or partitions on position
    for (int ci = 0; ci < cell_no; ci++)
    {
        for (j = 0; j < ele_per_cell[ci]; j++)
        {
            k = ci * MAX_ELE + j;
            x[k] += g_shift_x;
            y[k] += g_shift_y;
            z[k] += g_shift_z;
        }
    }

    // every rank reads the whole IC, then drops the cells outside its own X band.
    // the two end bands are open ended, so nothing drifting into the padding is lost.
    // arrays are compacted in place so element blocks stay MAX_ELE strided, and
    // id[] is renumbered because cellcenter.cl matches on (id[k] - 1) == cell slot.
    {
        int kept_cell = 0;
        int kept_ele = 0;
        for (int ci = 0; ci < cell_no; ci++)
        {
            int n = ele_per_cell[ci];
            if (n == 0) continue;

            float sum_x = 0.0f;
            for (j = 0; j < n; j++) sum_x += x[ci * MAX_ELE + j];
            float mean_x = sum_x / (float)n;

            int keep = 1;
            if (g_rank > 0 && mean_x < g_x_lo) keep = 0;
            if (g_rank < g_nprocs - 1 && mean_x >= g_x_hi) keep = 0;
            if (!keep) continue;

            for (j = 0; j < n; j++)
            {
                int src = ci * MAX_ELE + j;
                int dst = kept_cell * MAX_ELE + j;
                x[dst] = x[src];
                y[dst] = y[src];
                z[dst] = z[src];
                ele_type[dst] = ele_type[src];
                id[dst] = kept_cell + 1;
            }
            ele_per_cell[kept_cell] = n;
            cell_type[kept_cell] = cell_type[ci];
            cclock[kept_cell] = cclock[ci];
            cycle[kept_cell] = cycle[ci];

            kept_cell++;
            kept_ele += n;
        }

        // clear the vacated tail so no stale cell can be read back as live
        for (int ci = kept_cell; ci < cell_no; ci++)
        {
            ele_per_cell[ci] = 0;
            cell_type[ci] = 0;
            cclock[ci] = 0;
            cycle[ci] = 0;
        }
        for (int e = kept_cell * MAX_ELE; e < cell_no * MAX_ELE; e++)
        {
            x[e] = y[e] = z[e] = 0.0f;
            id[e] = ele_type[e] = 0;
        }

        cell_no = kept_cell;
        ele_no = kept_ele;
    }

    printf("[rank %d] after compaction: cell_no=%d ele_no=%d\n", g_rank, cell_no, ele_no);

    initial_chem_paras(x, y, z, id, cell_type, ele_type, cell_no, ele_per_cell);

    // Do the OpenCL calculation
    double t_setup = 0.0, t_loop = 0.0, t_comm = 0.0, t_io = 0.0;
    runCL(x, y, z, id, cell_type, ele_type, dt,
          max_ele_no,
          ele_no, cell_no, ele_per_cell,
          cclock, cycle,
          &t_setup, &t_loop, &t_comm, &t_io);

    // compute is the loop with the measured comm and io taken out
    printf("[TIMING] IC=%s rank=%d cells=%d setup=%.2f loop=%.2f comm=%.2f io=%.2f "
           "compute=%.2f wall=%.2f s\n",
           InName, g_rank, cell_no, t_setup, t_loop, t_comm, t_io,
           t_loop - t_comm - t_io, t_setup + t_loop);
    fflush(stdout);


    // Verify if the cells remained within the domain
    verifyDomain(x, y, z, max_ele_no);

    // Free up memory
    free(x);
    free(y);
    free(z);
    free(id);
    free(cell_type);
    free(ele_type);
    free(ele_per_cell);

    MPI_Finalize();

    return 0;
}
