#ifndef BIOLOGY_H
#define BIOLOGY_H

// map cell to each PDE grid
void  cell_in_block(float * x, float * y, float * z,
                    int * id, int * cell_type, int * ele_type,
                    int cell_no, int * ele_per_cell,
                    float * xo, float * yo, float * zo,
                    float * vx, float * vy, float * vz);

// initial condition of the chemical fields
void  initial_chem_paras(float * x, float * y, float * z,
                        int * id, int * cell_type, int * ele_type,
                        int cell_no, int * ele_per_cell);

// determine the fate of each cell after division
void  cell_lineage(int id, float x, float y, float z,
                    int ctype0, int * ctype1, int * ctype2,
                    float * chem1_gpu, float * chem2_gpu,
                    float * OVOL1, float * OVOL2, int flag_gener);

#endif // BIOLOGY_H
