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

/*
Kernel for cell movement using subcellular element model
The code returns the force for element[gId] of cell[cellId].
*/

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
__global int * flag_gener)
{
    int gId = get_global_id(0); //get the global ID of this work unit.
    int cellId = gId/max_ele;
    int eleId = gId - max_ele*cellId;
    if(cellId >= cell_no || eleId >= ele_per_cell[cellId]) return;
    if(ele_per_cell[cellId] == 0) return;

    int i, j, k, cid;
    float x_F1 = 0.0f;
    float y_F1 = 0.0f;
    float z_F1 = 0.0f;

    float r = 0.0f;
    float V = 0.0f; // a temporary variable to store potentials

    float dist = 20.0f;
    float x_dist = 0.0f;
    float y_dist = 0.0f;
    float z_dist = 0.0f;
    float x_shift = 1.0f;
    float y_shift = 1.0f;

    for(k = 0; k < cell_no; k++)
    {
      // update cell-cell distance
      // cell-cell interaction enables when the distance is small enough
      x_dist = fabs(xc[cellId] - xc[k]);
      if(x_dist > dist && lx-x_dist > dist)
          continue;
      y_dist = fabs(yc[cellId] - yc[k]);
      if(y_dist > dist && ly-y_dist > dist)
          continue;

      x_shift = 0.0f;
      if(x_dist > (lx-x_dist))
      {
          x_dist = lx-x_dist;
          if(xc[k] > xc[cellId])
              x_shift = -1.0f*lx;
          else
              x_shift = lx;
      }
      y_shift = 0.0f;
      if(y_dist > (ly-y_dist))
      {
          y_dist = ly-y_dist;
          if(yc[k] > yc[cellId])
              y_shift = -1.0f*ly;
          else
              y_shift = ly;
      }
      z_dist = fabs(zc[cellId] - zc[k]);

      r = distance3_new(x_dist, y_dist, z_dist);
      if(r > dist)
          continue;

      for(j = 0; j < ele_per_cell[k]; j++)
      {
        i = k*max_ele + j;

        x_dist = x[gId] - (x[i]+x_shift);
        y_dist = y[gId] - (y[i]+y_shift);
        z_dist = z[gId] - z[i];
        r = distance3_new(x_dist, y_dist, z_dist);
        if(i == gId)
        {
            // environmental force
            // if(cell_type[cellId] == 1 && type[gId] == 2)
            // {
            //     external_potential(x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1, lx, ly, lz);
            // }
        }
        else
        {
           if(id[gId] == id[i])
           {
              V = intra_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
           }
           else
           {
              if(cell_type[cellId] == cell_type[k]) 
              {
                  V = inter_potential_same(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
              }
              else if((cell_type[cellId] + cell_type[k]) == 3)
              {
                  V = inter_potential_01(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
              }
              else
              {
                  V = inter_potential(r, x[gId], y[gId], z[gId], &x_F1, &y_F1, &z_F1);
              }
           }
        }
        x_F1 += V * (x[gId] - (x[i]+x_shift));
        y_F1 += V * (y[gId] - (y[i]+y_shift));
        z_F1 += V * (z[gId] - z[i]);
      }
    }

    float ratio = 0.5;
    x_F[gId] = ratio * dt * x_F1 + x[gId]; // save the calculated values
    y_F[gId] = ratio * dt * y_F1 + y[gId];
    z_F[gId] = ratio * dt * z_F1 + z[gId];

    if(z_F[gId] < -1.0)
        z_F[gId] = -1.0 + 0.05*eleId;

    x_dist = x[gId] - x_F[gId];
    y_dist = y[gId] - y_F[gId];
    z_dist = z[gId] - z_F[gId];
    r = distance3_new(x_dist, y_dist, z_dist);
    r = distance3(x[gId], y[gId], z[gId], x_F[gId], y_F[gId], z_F[gId], lx, ly, lz);
    if(r > 4.0) // restrict maximum displacement per step
    {
        x_F[gId] = (x_F[gId] - x[gId])/r*(3.5+eleId/20.0) + x[gId];
        y_F[gId] = (y_F[gId] - y[gId])/r*(3.5+eleId/20.0) + y[gId];
        z_F[gId] = (z_F[gId] - z[gId])/r*(3.5+eleId/20.0) + z[gId];
    }
}

