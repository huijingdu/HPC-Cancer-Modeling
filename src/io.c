#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/stat.h>
#include <mpi.h>

#include "mpi_assert.h"
#include "sim_globals.h"
#include "io.h"

#ifdef _WIN32
    #include <direct.h>
    #define MKDIR(p) _mkdir(p)
#else
    #define MKDIR(p) mkdir((p), S_IRWXU)
#endif

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
