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
__const int nbx,
__const int nby, 
__const int nbz,
__const float bin_size,
__global const int * bin_offset,
__global const int * sorted_ids)
{
    int gId = get_global_id(0); //get the global ID of this work unit.
    int cellId = gId/max_ele;
    int eleId = gId - max_ele * cellId;
    if(cellId >= cell_no || eleId >= ele_per_cell[cellId]) return;
    if(ele_per_cell[cellId] == 0) return;

    float x_F1 = 0.0f, y_F1 = 0.0f, z_F1 = 0.0f;
    float r, V;
    float x_dist, y_dist, z_dist;
    
    int my_bx = (int)floor(x[gId] / bin_size);
    int my_by = (int)floor(y[gId] / bin_size);
    int my_bz = (int)floor(z[gId] / bin_size);
    if (my_bx < 0) my_bx = 0;
    if (my_by < 0) my_by = 0;
    if (my_bz < 0) my_bz = 0;
    if (my_bx >= nbx) my_bx = nbx - 1;
    if (my_by >= nby) my_by = nby - 1;
    if (my_bz >= nbz) my_bz = nbz - 1; 
    
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = my_bz + dz;
        if (nz < 0 || nz >= nbz) continue;
        
        for (int dy = -1; dy <= 1; ++dy) {
            int ny_raw = my_by + dy;
            int ny = (ny_raw + nby) % nby;
            float y_shift = 0.0f;
            if (ny_raw < 0) {
                y_shift = -ly;
            } else if (ny_raw >= nby) {
                y_shift = ly;
            }
           
            for (int dx = -1; dx <= 1; ++dx) { 
                int nx_raw = my_bx + dx;
                int nx = (nx_raw + nbx) % nbx;
                float x_shift = 0.0f;
                if (nx_raw < 0) {
                    x_shift = -lx;
                } else if (nx_raw >= nbx) {
                    x_shift = lx;
                }
                
                int nbin = nx + ny * nbx + nz * nbx * nby;
                int start = bin_offset[nbin];
                int end = bin_offset[nbin + 1];
                
                for (int p = start; p < end; ++p) {
                    int i = sorted_ids[p];
                    if (i == gId) continue;
                    
                    x_dist = x[gId] - (x[i] + x_shift);
                    y_dist = y[gId] - (y[i] + y_shift);
                    z_dist = z[gId] - z[i];
                    r = distance3_new(x_dist, y_dist, z_dist);
                    

                    if (id[gId] == id[i]) {
                        V = intra_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                    } else {
                        int k = i / max_ele;
                        if (cell_type[cellId] == cell_type[k]) {
                            V = inter_potential_same(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        } else if ((cell_type[cellId] + cell_type[k]) == 3) {
                            V = inter_potential_01(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        } else {
                            V = inter_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
                        }
                    }
                    x_F1 += V * x_dist;
                    y_F1 += V * y_dist;
                    z_F1 += V * z_dist;
                }
            }
        }
    }

    float ratio = 0.5f;
    x_F[gId] = ratio * dt * x_F1 + x[gId];
    y_F[gId] = ratio * dt * y_F1 + y[gId];
    z_F[gId] = ratio * dt * z_F1 + z[gId];
    
    if (z_F[gId] < -1.0f) z_F[gId] = -1.0f + 0.05f * eleId;
    
    r = distance3(x[gId], y[gId], z[gId], x_F[gId], y_F[gId], z_F[gId], lx, ly, lz);
    if (r > 4.0f) {
        x_F[gId] = (x_F[gId] - x[gId]) / r * (3.5f + eleId / 20.0f) + x[gId];
        y_F[gId] = (y_F[gId] - y[gId]) / r * (3.5f + eleId / 20.0f) + y[gId];
        z_F[gId] = (z_F[gId] - z[gId]) / r * (3.5f + eleId / 20.0f) + z[gId];
    }
}

