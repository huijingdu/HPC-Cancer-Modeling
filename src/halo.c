#include <stdlib.h>

#include "mpi_assert.h"
#include "sim_globals.h"
#include "sim_types.h"
#include "halo.h"

void xfer_alloc(xfer_buf * h, int max_cells, int floats_per_cell, int ints_per_cell)
{
    h->send_idx = (int *)malloc(max_cells * sizeof(int));
    h->send_f = (float *)malloc((size_t)max_cells * floats_per_cell * sizeof(float));
    h->send_i = (int *)malloc((size_t)max_cells * ints_per_cell * sizeof(int));
    h->recv_f = (float *)malloc((size_t)max_cells * floats_per_cell * sizeof(float));
    h->recv_i = (int *)malloc((size_t)max_cells * ints_per_cell * sizeof(int));
    assert(h->send_idx && h->send_f && h->send_i && h->recv_f && h->recv_i);
}

void xfer_free(xfer_buf * h)
{
    free(h->send_idx);
    free(h->send_f);
    free(h->send_i);
    free(h->recv_f);
    free(h->recv_i);
}

void halo_pack(xfer_buf * h, int s, int ci, const cell_arrays * c, const ele_arrays * e)
{
    float * fp = h->send_f + (size_t)s * GHOST_FLOATS_PER_CELL;
    int * ip = h->send_i + (size_t)s * GHOST_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) fp[j] = e->x[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[MAX_ELE + j] = e->y[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[2 * MAX_ELE + j] = e->z[ci * MAX_ELE + j];
    fp[3 * MAX_ELE + 0] = c->xc[ci];
    fp[3 * MAX_ELE + 1] = c->yc[ci];
    fp[3 * MAX_ELE + 2] = c->zc[ci];

    ip[0] = c->ele_per_cell[ci];
    ip[1] = c->cell_type[ci];
}

void halo_unpack(const xfer_buf * h, int r, int ci, cell_arrays * c, ele_arrays * e)
{
    float * fp = h->recv_f + (size_t)r * GHOST_FLOATS_PER_CELL;
    int * ip = h->recv_i + (size_t)r * GHOST_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) e->x[ci * MAX_ELE + j] = fp[j];
    for (int j = 0; j < MAX_ELE; j++) e->y[ci * MAX_ELE + j] = fp[MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) e->z[ci * MAX_ELE + j] = fp[2 * MAX_ELE + j];
    c->xc[ci] = fp[3 * MAX_ELE + 0];
    c->yc[ci] = fp[3 * MAX_ELE + 1];
    c->zc[ci] = fp[3 * MAX_ELE + 2];

    c->ele_per_cell[ci] = ip[0];
    c->cell_type[ci] = ip[1];
}

void mig_pack(xfer_buf * h, int s, int ci, const cell_arrays * c, const ele_arrays * e)
{
    float * fp = h->send_f + (size_t)s * MIG_FLOATS_PER_CELL;
    int * ip = h->send_i + (size_t)s * MIG_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) fp[j] = e->x[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[MAX_ELE + j] = e->y[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) fp[2 * MAX_ELE + j] = e->z[ci * MAX_ELE + j];
    fp[3 * MAX_ELE + 0] = c->xc[ci];
    fp[3 * MAX_ELE + 1] = c->yc[ci];
    fp[3 * MAX_ELE + 2] = c->zc[ci];
    fp[3 * MAX_ELE + 3] = c->OVOL1[ci];

    for (int j = 0; j < MAX_ELE; j++) ip[j] = e->id[ci * MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) ip[MAX_ELE + j] = e->ele_type[ci * MAX_ELE + j];
    ip[2 * MAX_ELE + 0] = c->ele_per_cell[ci];
    ip[2 * MAX_ELE + 1] = c->cell_type[ci];
    ip[2 * MAX_ELE + 2] = c->flag_gener[ci];
    ip[2 * MAX_ELE + 3] = c->cclock[ci];
    ip[2 * MAX_ELE + 4] = c->cycle[ci];
}

void mig_unpack(const xfer_buf * h, int r, int slot, cell_arrays * c, ele_arrays * e)
{
    float * fp = h->recv_f + (size_t)r * MIG_FLOATS_PER_CELL;
    int * ip = h->recv_i + (size_t)r * MIG_INTS_PER_CELL;

    for (int j = 0; j < MAX_ELE; j++) e->x[slot * MAX_ELE + j] = fp[j];
    for (int j = 0; j < MAX_ELE; j++) e->y[slot * MAX_ELE + j] = fp[MAX_ELE + j];
    for (int j = 0; j < MAX_ELE; j++) e->z[slot * MAX_ELE + j] = fp[2 * MAX_ELE + j];
    c->xc[slot] = fp[3 * MAX_ELE + 0];
    c->yc[slot] = fp[3 * MAX_ELE + 1];
    c->zc[slot] = fp[3 * MAX_ELE + 2];
    c->OVOL1[slot] = fp[3 * MAX_ELE + 3];

    for (int j = 0; j < MAX_ELE; j++) e->id[slot * MAX_ELE + j] = slot + 1;
    for (int j = 0; j < MAX_ELE; j++) e->ele_type[slot * MAX_ELE + j] = ip[MAX_ELE + j];
    c->ele_per_cell[slot] = ip[2 * MAX_ELE + 0];
    c->cell_type[slot] = ip[2 * MAX_ELE + 1];
    c->flag_gener[slot] = ip[2 * MAX_ELE + 2];
    c->cclock[slot] = ip[2 * MAX_ELE + 3];
    c->cycle[slot] = ip[2 * MAX_ELE + 4];
}