// Helper method for determining the bin of a cell given its center.
// Returns -1 when the center is outside the grid so the host can regrow it.
int bin_of_center(float x, float y, float z, float bin_size,
                  float ox, float oy, float oz, int nbx, int nby, int nbz) {
    int bx = (int)floor((x - ox) / bin_size);
    int by = (int)floor((y - oy) / bin_size);
    int bz = (int)floor((z - oz) / bin_size);
    if (bx < 0 || bx >= nbx) return -1;
    if (by < 0 || by >= nby) return -1;
    if (bz < 0 || bz >= nbz) return -1;
    return bx + by * nbx + bz * nbx * nby;
}

// Kernel 1: counts the number of cells per bin, tallying any that miss the grid
__kernel void bin_count(
    __global const float * xc,
    __global const float * yc,
    __global const float * zc,
    __global const int * ele_per_cell,
    __const int cell_no,
    __const float bin_size,
    __const int nbx,
    __const int nby,
    __const int nbz,
    __global int * bin_count,
    __const float ox,
    __const float oy,
    __const float oz,
    __global int * overflow) {
        int gId = get_global_id(0);
        if (gId >= cell_no || ele_per_cell[gId] == 0) return;
        int b = bin_of_center(xc[gId], yc[gId], zc[gId], bin_size, ox, oy, oz, nbx, nby, nbz);
        if (b < 0) {
            atomic_inc(&overflow[0]);
            return;
        }
        atomic_inc(&bin_count[b]);
}

// Kernel 4: occupied extent of the cell centers, at unit resolution
__kernel void cell_bounds(
    __global const float * xc,
    __global const float * yc,
    __global const float * zc,
    __global const int * ele_per_cell,
    __const int cell_no,
    __global int * bounds) {
        int gId = get_global_id(0);
        if (gId >= cell_no || ele_per_cell[gId] == 0) return;
        atomic_min(&bounds[0], (int)floor(xc[gId]));
        atomic_max(&bounds[1], (int)floor(xc[gId]));
        atomic_min(&bounds[2], (int)floor(yc[gId]));
        atomic_max(&bounds[3], (int)floor(yc[gId]));
        atomic_min(&bounds[4], (int)floor(zc[gId]));
        atomic_max(&bounds[5], (int)floor(zc[gId]));
}

// Kernel 2: constructs a prefix sum array from the bin_count array
__kernel void bin_prefix_sum(
    __global int * bin_count,
    __const int nbins,
    __global int * bin_offset) {
        if (get_global_id(0) != 0) return;
        int acc = 0;
        for (int i = 0; i < nbins; ++i) {
            int cnt = bin_count[i];
            bin_offset[i] = acc;
            bin_count[i] = acc; // reset to bin start so scatter can use it as a cursor
            acc += cnt;
        }
        bin_offset[nbins] = acc;
}

// Kernel 3: distributes cell ids by batching them by bins
__kernel void bin_scatter(
    __global const float * xc,
    __global const float * yc,
    __global const float * zc,
    __global const int * ele_per_cell,
    __const int cell_no,
    __const float bin_size,
    __const int nbx,
    __const int nby,
    __const int nbz,
    __global int * bin_count,
    __global int * cell_list,
    __const float ox,
    __const float oy,
    __const float oz) {
        int gId = get_global_id(0);
        if (gId >= cell_no || ele_per_cell[gId] == 0) return;
        int b = bin_of_center(xc[gId], yc[gId], zc[gId], bin_size, ox, oy, oz, nbx, nby, nbz);
        if (b < 0) return; // the host regrows before this runs, so never taken
        int slot = atomic_inc(&bin_count[b]);
        cell_list[slot] = gId;
}
