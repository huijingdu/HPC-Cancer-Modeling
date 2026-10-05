#ifndef SIM_TYPES_H
#define SIM_TYPES_H

#include <stdlib.h>

#include "mpi_assert.h"

// struct to manage element related arrays
typedef struct {
    float * x, * y, * z; // element position
    float * xF, * yF, * zF; // element force
    float * xo, * yo, * zo; // previous element position
    int * id, * ele_type;
    unsigned int * rng_x, * rng_c; // random number stream
} ele_arrays;

// helper method for allocating element related arrays
static inline void ele_alloc(ele_arrays * e, int max_ele_no) {
    // element position
    e->x = (float *)malloc(max_ele_no * sizeof(float));
    e->y = (float *)malloc(max_ele_no * sizeof(float));
    e->z = (float *)malloc(max_ele_no * sizeof(float));

    // element force
    e->xF = (float *)malloc(max_ele_no * sizeof(float));
    e->yF = (float *)malloc(max_ele_no * sizeof(float));
    e->zF = (float *)malloc(max_ele_no * sizeof(float));

    // previous element position
    e->xo = (float *)malloc(max_ele_no * sizeof(float));
    e->yo = (float *)malloc(max_ele_no * sizeof(float));
    e->zo = (float *)malloc(max_ele_no * sizeof(float));

    // element id and type
    e->id = (int *)malloc(max_ele_no * sizeof(int));
    e->ele_type = (int *)malloc(max_ele_no * sizeof(int));

    // rng streams
    e->rng_x = (unsigned int *)malloc(max_ele_no * sizeof(unsigned int));
    e->rng_c = (unsigned int *)malloc(max_ele_no * sizeof(unsigned int));

    assert(e->x && e->y && e->z &&
           e->xF && e->yF && e->zF &&
           e->xo && e->yo && e->zo &&
           e->id && e->ele_type &&
           e->rng_x && e->rng_c);
}

// helper method for tearing down element related arrays
static inline void ele_free(ele_arrays * e) {
    free(e->x); free(e->y); free(e->z);
    free(e->xF); free(e->yF); free(e->zF);
    free(e->xo); free(e->yo); free(e->zo);
    free(e->id); free(e->ele_type);
    free(e->rng_x); free(e->rng_c);
    free(e);
}

// struct to manage cell related arrays
typedef struct {
    float * xc, * yc, * zc; // cell center coordinates
    float * OVOL1, * OVOL2; // daughter cell type probabilities during division
    int * cell_type, * ele_per_cell; // cell type and number of elements per cell
    int * cclock, * cycle; // ticks since cell's last cycle and cycle length threshold
    int * flag, * flag_gener, * flag_death; // flags for cell status, generation marking and death marking
} cell_arrays;

// helper method for allocating cell related arrays
static inline void cell_alloc(cell_arrays * c, int max_cell) {
    // cell center coordinates
    c->xc = (float *)malloc(max_cell * sizeof(float));
    c->yc = (float *)malloc(max_cell * sizeof(float));
    c->zc = (float *)malloc(max_cell * sizeof(float));

    // daughter cell type probabilities during division
    c->OVOL1 = (float *)malloc(max_cell * sizeof(float));
    c->OVOL2 = (float *)malloc(max_cell * sizeof(float));

    // cell type and number of elements per cell
    c->cell_type = (int *)malloc(max_cell * sizeof(int));
    c->ele_per_cell = (int *)malloc(max_cell * sizeof(int));

    // ticks since cell's last cycle and cycle length threshold
    c->cclock = (int *)malloc(max_cell * sizeof(int));
    c->cycle = (int *)malloc(max_cell * sizeof(int));

    // flags for cell status, generation marking and death marking
    c->flag = (int *)malloc(max_cell * sizeof(int));
    c->flag_gener = (int *)malloc(max_cell * sizeof(int));
    c->flag_death = (int *)malloc(max_cell * sizeof(int));

    assert(c->xc && c->yc && c->zc &&
           c->OVOL1 && c->OVOL2 &&
           c->cell_type && c->ele_per_cell &&
           c->cclock && c->cycle &&
           c->flag && c->flag_gener && c->flag_death);
}

// helper method for tearing down cell related arrays
static inline void cell_free(cell_arrays * c) {
    free(c->xc); free(c->yc); free(c->zc);
    free(c->OVOL1); free(c->OVOL2);
    free(c->cell_type); free(c->ele_per_cell);
    free(c->cclock); free(c->cycle);
    free(c->flag); free(c->flag_gener); free(c->flag_death);
    free(c);
}

// struct to manage chem related arrays
typedef struct {
    float * vx, * vy, * vz; // chem cell accumulated displacement
    float * chem1_gpu, * chem2_gpu; // concentration of a chemical over chem grid
    float * chem11_gpu, * chem22_gpu; // diffusion buffer for next step
    int * grid_gpu, * grid1_gpu, * grid2_gpu, * grid3_gpu, * grid4_gpu; // total, and type 1-4 cell counts
} chem_grid;

// helper method for allocating chem related arrays
static inline void chem_alloc(chem_grid * ch, int nx, int ny, int nz) {
    // chem cell accumulated displacement
    ch->vx = (float *)malloc(nx * ny * nz * sizeof(float));
    ch->vy = (float *)malloc(nx * ny * nz * sizeof(float));
    ch->vz = (float *)malloc(nx * ny * nz * sizeof(float));

    // concentration of a chemical over chem grid
    ch->chem1_gpu = (float *)malloc(nx * ny * nz * sizeof(float));
    ch->chem2_gpu = (float *)malloc(nx * ny * nz * sizeof(float));

    // diffusion buffer for next step
    ch->chem11_gpu = (float *)malloc(nx * ny * nz * sizeof(float));
    ch->chem22_gpu = (float *)malloc(nx * ny * nz * sizeof(float));

    // total, and type 1-4 cell counts
    ch->grid_gpu = (int *)malloc(nx * ny * nz * sizeof(int));
    ch->grid1_gpu = (int *)malloc(nx * ny * nz * sizeof(int));
    ch->grid2_gpu = (int *)malloc(nx * ny * nz * sizeof(int));
    ch->grid3_gpu = (int *)malloc(nx * ny * nz * sizeof(int));
    ch->grid4_gpu = (int *)malloc(nx * ny * nz * sizeof(int));

    assert(ch->vx && ch->vy && ch->vz &&
           ch->chem1_gpu && ch->chem2_gpu &&
           ch->chem11_gpu && ch->chem22_gpu &&
           ch->grid_gpu && ch->grid1_gpu && ch->grid2_gpu &&
           ch->grid3_gpu && ch->grid4_gpu);
}

// helper method for tearing down chem related arrays
static inline void chem_free(chem_grid * ch) {
    free(ch->vx); free(ch->vy); free(ch->vz);
    free(ch->chem1_gpu); free(ch->chem2_gpu);
    free(ch->chem11_gpu); free(ch->chem22_gpu);
    free(ch->grid_gpu); free(ch->grid1_gpu); free(ch->grid2_gpu); free(ch->grid3_gpu); free(ch->grid4_gpu);
    free(ch);
}


// wall-clock breakdown reported by runCL
typedef struct {
    double setup; // time before the main loop starts
    double loop;  // time inside the main loop
    double comm;  // MPI time (loop + output)
    double io;    // file output time
} sim_timing;


// staging for one direction of an exchange. send_idx lists the local slots
// going out; recv holds what arrives from the opposite neighbour
typedef struct {
    int * send_idx;
    float * send_f;
    int * send_i;
    float * recv_f;
    int * recv_i;
} xfer_buf;


#endif // SIM_TYPES_H