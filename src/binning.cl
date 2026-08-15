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

// Exclusive scan across one work group. Returns this item's prefix and leaves
// the inclusive scan in scratch[0..lsz-1], so the group total is scratch[lsz-1]
int wg_exclusive_scan(int v, __local int * scratch, int lid, int lsz) {
    __local int * cur = scratch;
    __local int * nxt = scratch + lsz;

    cur[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int off = 1; off < lsz; off <<= 1) {
        int t = cur[lid];
        if (lid >= off) t += cur[lid - off];
        barrier(CLK_LOCAL_MEM_FENCE); // reads of cur have landed
        nxt[lid] = t;
        barrier(CLK_LOCAL_MEM_FENCE); // writes of nxt have landed
        __local int * tmp = cur; cur = nxt; nxt = tmp;
    }

    // the sweep ends in either half, so settle it in the first one
    int inclusive = cur[lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    scratch[lid] = inclusive;
    barrier(CLK_LOCAL_MEM_FENCE);

    return inclusive - v;
}

// Kernel 2a: exclusive scan of bin_count within each work group
// reads bin_count but never writes it
__kernel void bin_block_scan(
    __global const int * bin_count,
    __const int nbins,
    __global int * bin_offset,
    __global int * block_sums,
    __local int * scratch) {
        int i = get_global_id(0);
        int lid = get_local_id(0);
        int lsz = get_local_size(0);

        // the last group runs past nbins, and only [0, nbins) is zeroed, so the
        // tail contributes zero rather than reading stale counts
        int v = (i < nbins) ? bin_count[i] : 0;

        int excl = wg_exclusive_scan(v, scratch, lid, lsz);

        if (i < nbins) bin_offset[i] = excl;
        if (lid == lsz - 1) block_sums[get_group_id(0)] = scratch[lsz - 1];
}

// Kernel 2b: exclusive scan of the group totals by a single work group
__kernel void bin_scan_block_sums(
    __global int * block_sums,
    __const int nblocks,
    __local int * scratch) {
        int lid = get_local_id(0);
        int lsz = get_local_size(0);

        int chunk = (nblocks + lsz - 1) / lsz;
        int lo = lid * chunk;
        int hi = lo + chunk;
        if (hi > nblocks) hi = nblocks; // trailing items get an empty slice

        int sum = 0;
        for (int i = lo; i < hi; ++i) sum += block_sums[i];

        int base = wg_exclusive_scan(sum, scratch, lid, lsz);
        int total = scratch[lsz - 1];

        // each item rewrites only the slice it read, so the passes never race
        int acc = base;
        for (int i = lo; i < hi; ++i) {
            int cnt = block_sums[i];
            block_sums[i] = acc;
            acc += cnt;
        }

        // the grand total rides in the spare slot for 2c to publish
        if (lid == 0) block_sums[nblocks] = total;
}

// Kernel 2c: shifts each group's local scan by that group's offset, mirroring
// the result into bin_count as bin_scatter's write cursor
__kernel void bin_add_block_offsets(
    __global int * bin_count,
    __const int nbins,
    __global int * bin_offset,
    __global const int * block_sums,
    __const int nblocks) {
        int i = get_global_id(0);
        // movement.cl reads bin_offset[nbin + 1], so the last bin needs this
        if (i == 0) bin_offset[nbins] = block_sums[nblocks];
        if (i >= nbins) return;

        int start = bin_offset[i] + block_sums[get_group_id(0)];
        bin_offset[i] = start;
        bin_count[i] = start; // bin start, so scatter can use it as a cursor
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
