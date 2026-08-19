//the intracellular potential, the morse potential in this case
float intra_potential(float r, float x, float y, float z, float * x_F1, float * y_F1, float * z_F1) 
{
    if(r <= 0.0f)  return 0.0;

    float r2 = 1.0f/(r*r);
    float rm = 1.3f/r; // 1.8
    float rm6 = rm*rm*rm*rm*rm*rm;
    float m = 1.5f*(rm6*rm6*r2 - rm6*r2);
    m -= .01*r;

    if(m > 10.0) return 10.0f;
    else return 1.0f*m; 
}

float inter_potential_01(float r, float x, float y, float z, float * x_F1, float * y_F1, float * z_F1)
{
    if(r <= 0.0f || r >= 12.0f) return 0.0;

    float r2 = 1.0f/(r*r);
    float rm = 3.85f/r; // 4.5
    float rm6 = rm*rm*rm*rm*rm*rm;
    float m = 0.3f*(rm6*rm6*r2 - rm6*r2); // 0.1

    if(m>0)
        m=1.0f*m; // repulsion
    else
        m=2.5f*m; // adhesion

    if(m > 10.0) return 10.0f; // 10.0
    else
        return m;
}

float inter_potential(float r, float x, float y, float z, float * x_F1, float * y_F1, float * z_F1)
{
    if(r <= 0.0f || r >= 10.5f) return 0.0;

    float r2 = 1.0f/(r*r);
    float rm = 4.0f/r; // 4.5
    float rm6 = rm*rm*rm*rm*rm*rm;
    float m = 0.3f*(rm6*rm6*r2 - rm6*r2); // 0.1

    if(m>0)
        m=2.0f*m; // repulsion
    else
        m=0.5f*m; // adhesion

    if(m > 10.0) return 10.0f; // 10.0
    else
        return m;
}

float inter_potential_same(float r, float x, float y, float z, float * x_F1, float * y_F1, float * z_F1)
{
    if(r <= 0.0f || r >= 10.5f) return 0.0;

    float r2 = 1.0f/(r*r);
    float rm = 4.0f/r; // 4.5
    float rm6 = rm*rm*rm*rm*rm*rm;
    float m = 0.3f*(rm6*rm6*r2 - rm6*r2); // 0.1

    if(m>0)
        m=1.0f*m; // repulsion
    else
    {
        if(r < 5.0f)
            m=2.0f*m; // adhesion - short distance
        else
            m=2.0f*m; // adhesion - long distance
    }
    if(m > 10.0) return 10.0f; // 10.0
    else
        return m;
}

void external_potential(float x, float y, float z, 
                        float * x_F1, float * y_F1, float * z_F1,
                        float lx, float ly, float lz)
{
    float r = z + 1.0f;
    float m;
    {
        float rm = 5.0f/r;
        m = -3.0*rm;//-3.0f*z;
        if(m*m > 1000.0)
            m = 100.0f;
    }
    if(r < 0) m = 10.0f;
    *z_F1 += m;
    return;
}

// 3d distance function
float distance3(float x1, float y1, float z1, float x2, float y2, float z2, float lx, float ly, float lz)
{
    float dx = fabs(x1 - x2);
    // if(dx > lx/2.0) dx = lx - dx;
    float dy = fabs(y1 - y2);
    // if(dy > ly/2.0) dy = ly - dy;
    float dz = fabs(z1 - z2);

    return sqrt(dx*dx + dy*dy + dz*dz) + 0.0001;
}

float distance3_new(float dx, float dy, float dz)
{
    return sqrt(dx*dx + dy*dy + dz*dz) + 0.0001;
}

//Kernel for cell movement using subcellular element model
//The code returns the force for element[gId] of cell[cellId].
__kernel void movement(
__global int * id, 
__global float * x, 
__global float * y, 
__global float * z, 
__global int * type, 
__const int cell_no,
__const int ele_no,
__global int * ele_per_cell, 
__global float * xc,
__global float * yc,
__global float * zc,
__const float dt, 
__global float * x_F, 
__global float * y_F, 
__global float * z_F,
__const float lx, 
__const float ly, 
__const float lz,
__const int max_ele,
__global int * cell_type,
__global int * flag_gener,
__global const int * cell_list,
__global const int * bin_offset,
__const float inter_dist,
__const int bins_x,
__const int bins_y,
__const int bins_z,
__const float bin_ox,
__const float bin_oy,
__const float bin_oz,
__const float sub_z)
{
    int gId = get_global_id(0); //get the global ID of this work unit.
    int cellId = gId/max_ele;
    int eleId = gId - max_ele * cellId;
    if(cellId >= cell_no || eleId >= ele_per_cell[cellId]) return;
    if(ele_per_cell[cellId] == 0) return;

    int i, j, k;
    float x_F1 = 0.0f, y_F1 = 0.0f, z_F1 = 0.0f;
    float r, V;
    float x_dist, y_dist, z_dist;

    // intra-cell interactions (same cell, no bins, no cutoff)
    for (j = 0; j < ele_per_cell[cellId]; ++j) {
        i = cellId * max_ele + j;
        if (i == gId) continue;
        x_dist = x[gId] - x[i];
        y_dist = y[gId] - y[i];
        z_dist = z[gId] - z[i];
        r = distance3_new(x_dist, y_dist, z_dist);
        V = intra_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
        x_F1 += V * x_dist;
        y_F1 += V * y_dist;
        z_F1 += V * z_dist;
    }

    // inter-cell interactions, sweeping the 27-bin neighborhood of cell centers
    int my_bx = (int)floor((xc[cellId] - bin_ox) / inter_dist);
    int my_by = (int)floor((yc[cellId] - bin_oy) / inter_dist);
    int my_bz = (int)floor((zc[cellId] - bin_oz) / inter_dist);
    // the host regrows the grid to fit every center, so these only guard indexing
    if (my_bx < 0) my_bx = 0;
    if (my_by < 0) my_by = 0;
    if (my_bz < 0) my_bz = 0;
    if (my_bx >= bins_x) my_bx = bins_x - 1;
    if (my_by >= bins_y) my_by = bins_y - 1;
    if (my_bz >= bins_z) my_bz = bins_z - 1;

    for (int dz = -1; dz <= 1; ++dz) {
        int nz = my_bz + dz;
        if (nz < 0 || nz >= bins_z) continue;

        for (int dy = -1; dy <= 1; ++dy) {
            int ny = my_by + dy;
            if (ny < 0 || ny >= bins_y) continue;

            for (int dx = -1; dx <= 1; ++dx) {
                int nx = my_bx + dx;
                if (nx < 0 || nx >= bins_x) continue;

                int nbin = nx + ny * bins_x + nz * bins_x * bins_y;
                int start = bin_offset[nbin];
                int end = bin_offset[nbin + 1];

                for (int p = start; p < end; ++p) {
                    k = cell_list[p];
                    if (k == cellId) continue; // same-cell handled by the intra loop

                    // cell-cell cull: reject the whole 20x20 element block cheaply
                    float dist = 20.0f;
                    x_dist = fabs(xc[cellId] - xc[k]);
                    if (x_dist > dist) continue;
                    y_dist = fabs(yc[cellId] - yc[k]);
                    if (y_dist > dist) continue;
                    z_dist = fabs(zc[cellId] - zc[k]);
                    if (distance3_new(x_dist, y_dist, z_dist) > dist) continue;

                    for (j = 0; j < ele_per_cell[k]; ++j) {
                        i = k * max_ele + j;
                        x_dist = x[gId] - x[i];
                        y_dist = y[gId] - y[i];
                        z_dist = z[gId] - z[i];
                        r = distance3_new(x_dist, y_dist, z_dist);
                        if (r >= 12.0f) continue; // inter potentials are 0 past 12

                        if (cell_type[cellId] == cell_type[k]) {
                            V = inter_potential_same(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        } else if ((cell_type[cellId] + cell_type[k]) == 3) {
                            V = inter_potential_01(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        } else {
                            V = inter_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        }
                        x_F1 += V * x_dist;
                        y_F1 += V * y_dist;
                        z_F1 += V * z_dist;
                    }
                }
            }
        }
    }

    // external force, applied once per element, to the
    // substrate-anchored elements of slow-dividing cells
    if (cell_type[cellId] == 1 && type[gId] == 2) {
        external_potential(x[gId], y[gId], z[gId] - sub_z - 1.0f, &x_F1, &y_F1, &z_F1, lx, ly, lz);
    }

    float ratio = 0.5f;
    x_F[gId] = ratio * dt * x_F1 + x[gId];
    y_F[gId] = ratio * dt * y_F1 + y[gId];
    z_F[gId] = ratio * dt * z_F1 + z[gId];
    
    // no element may fall through the same plane external_potential pulls toward
    if (z_F[gId] < sub_z) z_F[gId] = sub_z + 0.05f * eleId;
    
    r = distance3(x[gId], y[gId], z[gId], x_F[gId], y_F[gId], z_F[gId], lx, ly, lz);
    if (r > 4.0f) {
        x_F[gId] = (x_F[gId] - x[gId]) / r * (3.5f + eleId / 20.0f) + x[gId];
        y_F[gId] = (y_F[gId] - y[gId]) / r * (3.5f + eleId / 20.0f) + y[gId];
        z_F[gId] = (z_F[gId] - z[gId]) / r * (3.5f + eleId / 20.0f) + z[gId];
    }
}

