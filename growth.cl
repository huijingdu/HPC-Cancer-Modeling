/*
Kernel for growth of cell[gId] 
Growth or volume increasement is modeled by adding one element to the cell
*/

__kernel void growth( 
__global int * id, 
__global float * x, 
__global float * y, 
__global float * z, 
__global int * ele_type, 
__const int ele_no,
__global int * flag, 
__global int * ele_per_cell,
__const float lx,
__const float ly,
__const float lz,
__const int max_ele,
__global int * cell_type,
__global int * flag_death)
{
    int gId = get_global_id(0);
    if(ele_per_cell[gId] >= max_ele || ele_per_cell[gId] <= 0) return;
    if(cell_type[gId] == 3 && flag_death[gId] == 1) return;

    float cx = 0.0f;
    float cy = 0.0f;
    float cz = 0.0f;
    float mz = -2.0f; 
    int n = 0; 
    int count_type = 0;
    int k;

    for(int i = 0; i < ele_per_cell[gId]; i++)
    {
        k = gId*max_ele + i;
        if(id[k] == (gId+1))
        {
            cx += x[k];
            cy += y[k];
            cz += z[k];
            if(z[k] > mz) mz = z[k];
            if(ele_type[k] == 2) count_type++;
            n++;
        }
    }

    if(n <= 0 || n != ele_per_cell[gId]) 
    {
        ele_per_cell[gId] = -1;
        return;
    }

    cx = cx/n;
    cy = cy/n;
    cz = cz/n;
    
    k = gId*max_ele + ele_per_cell[gId];
    x[k] = cx;
    y[k] = cy;
    //z[ele_no+gId] = mz + 0.05;
    z[k] = cz;
    id[k] = gId + 1;
    ele_per_cell[gId]++;
    if(count_type < n/2) 
    {
        ele_type[k] = 2;
        z[k] = z[k] - 0.05;
    }
    else
    {
        ele_type[k] = 1;
        z[k] = z[k] + 0.05;
    }
        
    flag[gId]++;
}
