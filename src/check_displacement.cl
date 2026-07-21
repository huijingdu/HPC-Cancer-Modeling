// Lazy cell-list rebuild check
// Flags need_rebuild if a cell moved more than skin/2 since the last rebuild
// Flag must be zeroed first
__kernel void check_displacement(
    __global const float* xc,
    __global const float* yc,
    __global const float* zc,
    __global const float* xc_ref,
    __global const float* yc_ref,
    __global const float* zc_ref,
    __global const int* ele_per_cell,
    __global int* need_rebuild, // single int, pre-zeroed
    const int cell_no,
    const float half_skin) // skin / 2.0
{
    int c = get_global_id(0);
    if (c >= cell_no) return;
    if (ele_per_cell[c] == 0) return;

    float dx = xc[c] - xc_ref[c];
    float dy = yc[c] - yc_ref[c];
    float dz = zc[c] - zc_ref[c];
    float disp2 = dx*dx + dy*dy + dz*dz;

    if (disp2 > half_skin * half_skin) {
        need_rebuild[0] = 1;
    }
}
