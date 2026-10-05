#ifndef HALO_H
#define HALO_H

#include "sim_globals.h"
#include "sim_types.h"

// ghosts carry only the partner fields movement.cl reads
#define GHOST_FLOATS_PER_CELL (3 * MAX_ELE + 3) // x,y,z blocks + xc,yc,zc
#define GHOST_INTS_PER_CELL 2 // ele_per_cell, cell_type

// migrants carry enough to fully reconstruct a cell on the receiving rank
#define MIG_FLOATS_PER_CELL (3 * MAX_ELE + 4) // x,y,z blocks + xc,yc,zc + OVOL1
#define MIG_INTS_PER_CELL (2 * MAX_ELE + 5) // id[], ele_type[], epc, ctype, flag_gener, cclock, cycle

// size the send and recv staging for up to max_cells records
void xfer_alloc(xfer_buf * h, int max_cells, int floats_per_cell, int ints_per_cell);

// free the arrays xfer_alloc made
// the xfer_buf itself is not freed
void xfer_free(xfer_buf * h);

// copy one local cell into the outgoing halo record at slot s
void halo_pack(xfer_buf * h, int s, int ci, const cell_arrays * c, const ele_arrays * e);

// place an arrived halo record into local cell slot ci
void halo_unpack(const xfer_buf * h, int r, int ci, cell_arrays * c, ele_arrays * e);

// a migrant carries everything needed to rebuild the cell on the receiving rank
void mig_pack(xfer_buf * h, int s, int ci, const cell_arrays * c, const ele_arrays * e);

// id[] is rewritten to the new slot, not copied: cellcenter.cl needs (id[k] - 1) == slot
void mig_unpack(const xfer_buf * h, int r, int slot, cell_arrays * c, ele_arrays * e);

#endif // HALO_H