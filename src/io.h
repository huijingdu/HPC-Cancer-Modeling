#ifndef IO_H
#define IO_H

// scan the IC for simluation setup
void sniff_ic(const char InName[], int *cell_no_out, float *mx_out, float *my_out, float *mz_out,
              float *nx_out, float *ny_out, float *nz_out);

// reader for initial cells' location
// from file Input/IC
void InitialReader(const char InName[], float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int * cell_no, int * ele_no, int * ele_per_cell, int * cclock, int * cycle);

// output PQR files for visual in VMD
// every rank formats and writes its own cells
void output_mpi(float * x, float * y, float * z, float * xc, float * yc, float * zc, int * id, int * cell_type, int * ele_type, int cell_no, int ele_no, int * ele_per_cell, int step, float * chem1_gpu, float * chem2_gpu, int * cclock, int * cycle, int * flag_gener, float * vx, float * vy, float * vz);

#endif // IO_H