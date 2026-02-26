/*
Kernel to update the intracellular gene network of cell[gId].
*/

__kernel void ovol_fun( 
__global int * id, 
__global float * x, 
__global float * y, 
__global float * z, 
__global int * ele_per_cell, 
__global float * chem1,
__global float * chem2,
__global float * o1,
__global float * o2,
__const int max_ele,
__global int * cell_type)
{
    int gId = get_global_id(0); 
    int cellId = gId/max_ele;
    if(ele_per_cell[gId] == 0)
    {
        return;
    }

    int NX = 40;
    int NY = NX;
    int NZ = 80;
    float LX = 40.0;
    float LY = LX;
    float LZ = 80.0;
    float DX = LX/NX;
    float DY = LY/NY;
    float DZ = LZ/NZ;
 
    int ni, nj, nk, i, k;
    int i0, i1, j0, j1, k0, k1;
    float to1 = 0.0;
    for(i = 0; i < ele_per_cell[gId]; i++)
    {
        k = gId*max_ele + i;
        ni = (int)(1.0*x[k]/(1.0*DX)); if(ni < 0) ni = NX-1; if(ni >= NX) ni = 0;
        nj = (int)(1.0*y[k]/(1.0*DY)); if(nj < 0) nj = NY-1; if(nj >= NY) nj = 0;
        nk = (int)(1.0*z[k]/(1.0*DZ)); if(nk < 0) nk = 0; if(nk >= NZ) nk = NZ-1;

        for(i0=-2; i0<=2; i0++)
        {
            i1 = ni + i0;
            if(i1 < 0) i1 = i1+NX; if(i1 >= NX) i1 = i1-NX;
            for(j0=-2; j0<=2; j0++)
            {
                j1 = nj + j0;
                if(j1 < 0) j1 = j1+NY; if(j1 >= NY) j1 = j1-NY;
                for(k0=-2; k0<=2; k0++)
                {
                    k1 = nk + k0;
                    if(k1 < 0) k1 = 0; if(k1 >= NZ) k1 = NZ-1;
                    to1 = to1 + chem1[i1 + j1*NX + k1*NX*NY];
                }
            }
        }
    }
    to1 = to1/125.0;

    if(cell_type[cellId] == 1)
    {
        o1[gId] = 0.0;
        o2[gId] = 0.0+to1;
    }
    if(cell_type[cellId] == 2 || cell_type[cellId] == 4)
    {
        o1[gId] = 0.0;
        o2[gId] = 0.0;
    }
    if(cell_type[cellId] == 3)
    {
        o1[gId] = 0.0+to1;
        o2[gId] = 0.0;
    }
}
