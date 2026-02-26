/*
Kernel to update cell center of cell[gId].
*/

__kernel void cellcenter( 
__global int * id, 
__global float * x, 
__global float * y, 
__global float * z, 
__global int * ele_per_cell, 
__global float * xc,
__global float * yc,
__global float * zc,
__const int max_ele,
__const float lx,
__const float ly,
__const float lz)
{
    int gId = get_global_id(0);

    if(ele_per_cell[gId] == 0)
    {
        xc[gId] = 0.0f;
        yc[gId] = 0.0f;
        zc[gId] = 0.0f;
        return;
    }
    
    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    int count_tmp = 0;
    int i, k, kk;

    for(i = 0; i < ele_per_cell[gId]; i++)
    {
        k = gId*max_ele + i; // element id of this cell
        if((id[k]-1) == gId)
        {
            cx += x[k];
            cy += y[k];
            cz += z[k];
            count_tmp++;
        }
    }
    if(count_tmp < 0 || count_tmp != ele_per_cell[gId])
    {
        ele_per_cell[gId] = -1;
        return;
    }

    if(count_tmp == ele_per_cell[gId])
    {
        cx = cx/count_tmp;
        cy = cy/count_tmp;
        cz = cz/count_tmp;
        xc[gId] = cx;
        yc[gId] = cy;
        zc[gId] = cz;
    }

    if(0) // enable (1) periodic boundary conditions
    {
        if(cx < -0.0f)
        {
          for(i = 0; i < ele_per_cell[gId]; i++)
          {
            kk = gId*max_ele + i;
            x[kk] = lx + x[kk];
          }
          cx = cx + lx;
          xc[gId] = cx;
        }
        else if(cx > (lx + 0.0f))
        {
          for(i = 0; i < ele_per_cell[gId]; i++)
          {
            kk = gId*max_ele + i;
            x[kk] = x[kk] - lx;
          }
          cx = cx - lx;
          xc[gId] = cx;
        }
        else
        {
          xc[gId] = cx;
        }

        if(cy < -0.0f)
        {
          for(i = 0; i < ele_per_cell[gId]; i++)
          {
            kk = gId*max_ele + i;
            y[kk] = ly + y[kk];
          }
          cy = cy + ly;
          yc[gId] = cy;
        }
        else if(cy > (ly + 0.0f))
        {
          for(i = 0; i < ele_per_cell[gId]; i++)
          {
            kk = gId*max_ele + i;
            y[kk] = y[kk] - ly;
          }
          cy = cy - ly;
          yc[gId] = cy;
        }
        else
        {
          yc[gId] = cy;
        }
    }
}
