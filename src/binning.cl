// Helper method for determining the bin of a cell given its center
int bin_of_center(float x, float y, float z, float bin_size, int nbx, int nby, int nbz) {
    int bx = (int)floor(x / bin_size);
    int by = (int)floor(y / bin_size);
    int bz = (int)floor(z / bin_size);
    if (bx < 0) bx = 0; if (bx >= nbx) bx = nbx - 1;
    if (by < 0) by = 0; if (by >= nby) by = nby - 1;
    if (bz < 0) bz = 0; if (bz >= nbz) bz = nbz - 1;
    return bx + by * nbx + bz * nbx * nby;
}

// Kernel 1: counts the number of cells per bin
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
    __global int * bin_count) {
        int gId = get_global_id(0);
        if (gId >= cell_no || ele_per_cell[gId] == 0) return;
        atomic_inc(&bin_count[bin_of_center(xc[gId], yc[gId], zc[gId], bin_size, nbx, nby, nbz)]);
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
    __global int * cell_list) {
        int gId = get_global_id(0);
        if (gId >= cell_no || ele_per_cell[gId] == 0) return;
        int b = bin_of_center(xc[gId], yc[gId], zc[gId], bin_size, nbx, nby, nbz);
        int slot = atomic_inc(&bin_count[b]);
        cell_list[slot] = gId;
}
