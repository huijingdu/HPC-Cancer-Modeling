// Helper method for determining the bin of a cell given its position
int bin_of_pos(float x, float y, float z, float bin_size, int nbx, int nby, int nbz) {
    int bx = (int)floor(x / bin_size);
    int by = (int)floor(y / bin_size);
    int bz = (int)floor(z / bin_size);
    if (bx < 0) bx = 0;
    if (by < 0) by = 0;
    if (bz < 0) bz = 0;
    if (bx >= nbx) bx = nbx - 1;
    if (by >= nby) by = nby - 1;
    if (bz >= nbz) bz = nbz - 1;
    return bx + by * nbx + bz * nbx * nby;
}

// Kernel 1: assigns cells to bins
__kernel void bin_assign(
    __global const float * x,
    __global const float * y, 
    __global const float * z,
    __global const int * ele_per_cell,
    __const int max_ele,
    __const int cell_no,
    __const float bin_size,
    __const int nbx,
    __const int nby,
    __const int nbz,
    __const int nbins,
    __global int * bin_id) {
        int gId = get_global_id(0);
        int cellId = gId / max_ele;
        int eleId = gId - cellId * max_ele;

        if (cellId >= cell_no || eleId >= ele_per_cell[cellId]) {
            bin_id[gId] = nbins;
            return;
        }
        
        bin_id[gId] = bin_of_pos(x[gId], y[gId], z[gId], bin_size, nbx, nby, nbz); 
}


// Kernel 2: keeps track of the number of assigned elements per bin.
__kernel void bin_histogram(
    __global const int * bin_id,
    __const int nbins,
    __global int * bin_count) {
    int gId = get_global_id(0);
    int b = bin_id[gId];
    if (b >= nbins) return;
    atomic_inc(&bin_count[b]);
}

// Kernel 3: constructs a prefix sum array from the bin_count array
__kernel void bin_prefix_sum(
    __global const int * bin_count,
    __const int nbins,
    __global int * bin_offset) {
    if (get_global_id(0) != 0) return;
    int acc = 0;
    for (int i = 0; i < nbins; ++i) {
        bin_offset[i] = acc;
        acc += bin_count[i];
    }
    bin_offset[nbins] = acc;
}

// Kernel 4: distributes element ids by batching them by bins
__kernel void bin_scatter(
    __global const int * bin_id,
    __global const int * bin_offset,
    __const int nbins, 
    __global int * bin_pos,
    __global int * sorted_ids) {
        int gId = get_global_id(0);
        int b = bin_id[gId];
        if (b >= nbins) return;
        
        int slot = atomic_inc(&bin_pos[b]);
        sorted_ids[bin_offset[b] + slot] = gId;
}
