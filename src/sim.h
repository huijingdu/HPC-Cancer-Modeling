#ifndef SIM_H
#define SIM_H

#include "sim_types.h"

int runCL(ele_arrays * e, cell_arrays * c, chem_grid * ch, float dt,
          int max_ele_no, int ele_no, int cell_no,
          double *out_setup, double *out_loop, double *out_comm, double *out_io);
          
#endif // SIM_H