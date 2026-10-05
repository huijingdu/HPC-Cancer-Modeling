#include <stdlib.h>
#include <stdio.h>

#include "biology.h"
#include "sim_globals.h"

void  cell_in_block(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell, float * xo, float * yo, float * zo, float * vx, float * vy, float * vz)
{
    int i, j, k, index;
    int ni, nj, nk;

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
    {
        grid[i][j][k] = 0;
        grid1[i][j][k] = 0;
        grid2[i][j][k] = 0;
        grid3[i][j][k] = 0;
        grid4[i][j][k] = 0;
        index = i + j*g_nx + k*g_nx*g_ny;
        vx[index] = 0.0;
        vy[index] = 0.0;
        vz[index] = 0.0;
    }

    for(i = 0; i < cell_no; i++)
    {
        for(j = 0; j < ele_per_cell[i]; j++)
        {
            k = i*MAX_ELE + j;
            ni = (int)((x[k] - g_chem_ox)/(1.0*g_dx)); if(ni < 0) ni = 0; if(ni >= g_nx) ni = g_nx-1;
            nj = (int)((y[k] - g_chem_oy)/(1.0*g_dy)); if(nj < 0) nj = 0; if(nj >= g_ny) nj = g_ny-1;
            nk = (int)((z[k] - g_chem_oz)/(1.0*g_dz)); if(nk < 0) nk = 0; if(nk >= g_nz) nk = g_nz-1;

            index = ni + nj*g_nx + nk*g_nx*g_ny;
            vx[index] += x[k] - xo[k];
            vy[index] += y[k] - yo[k];
            vz[index] += z[k] - zo[k];

            grid[ni][nj][nk]++;
            if(cell_type[i] == 1)
            {
                grid1[ni][nj][nk]++;
            }
            else if(cell_type[i] == 2)
            {
                grid2[ni][nj][nk]++;
            }
            else if(cell_type[i] == 3)
            {
                grid3[ni][nj][nk]++;
            }
            else if(cell_type[i] == 4)
            {
                grid4[ni][nj][nk]++;
            }
            else
            {
                printf("ERROR::cell_in_block(), wrong cell type = %d\n", cell_type[i]);
                exit(0);
            }
            xo[k] = x[k];
            yo[k] = y[k];
            zo[k] = z[k];
        }
    } 

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
    {
        index = i + j*g_nx + k*g_nx*g_ny;
        if(grid[i][j][k] == 0)
        {
            vx[index] = 0.0;
            vy[index] = 0.0;
            vz[index] = 0.0;
        }
        else
        {
            vx[index] /= 1.0*grid[i][j][k];
            vy[index] /= 1.0*grid[i][j][k];
            vz[index] /= 1.0*grid[i][j][k];
        }
    }
}

void  initial_chem_paras(float * x, float * y, float * z, int * id, int * cell_type, int * ele_type, int cell_no, int * ele_per_cell)
{
    int i, j, k;
    int ni, nj, nk;
    int i0, i1, j0, j1, k0, k1;
    float cc = 0.05;

    for(i = 0; i < g_nx; i++)
        for(j = 0; j < g_ny; j++)
            for(k = 0; k < g_nz; k++)
    {
        grid[i][j][k] = 0;
        chem1[i][j][k] = 0.0;
        chem2[i][j][k] = 0.0;
    }

    return;
}

void  cell_lineage(int id, float x, float y, float z, int ctype0, int * ctype1, int * ctype2, float * chem1_gpu, float * chem2_gpu, float * OVOL1, float * OVOL2, int flag_gener)
{
    int i;
    int ni, nj, nk;
    float cc, pp1, pp2;

    float alpha = 10000.0;

    ni = (int)((x - g_chem_ox)/(1.0*g_dx)); if(ni < 0) ni = 0; if(ni >= g_nx) ni = g_nx-1;
    nj = (int)((y - g_chem_oy)/(1.0*g_dy)); if(nj < 0) nj = 0; if(nj >= g_ny) nj = g_ny-1;
    nk = (int)((z - g_chem_oz)/(1.0*g_dz)); if(nk < 0) nk = 0; if(nk >= g_nz) nk = g_nz-1;

    i = ni + nj*g_nx + nk*g_nx*g_ny;
    cc = chem1_gpu[i]/1.0;
    if(ctype0 == 1)
    {
        float q1 = 0.15, q2 = 0.1;
        float tmpp = RNG();
        if(tmpp <= q1)
        {
            *ctype1 = 1;
            *ctype2 = 1;
        }
        else if(tmpp <= (q1+q2))
        {
            *ctype1 = 2;
            *ctype2 = 2;
        }
        else
        {
            *ctype1 = 1;
            *ctype2 = 2;
        }

    }
    else if(ctype0 == 2)
    {
        float p1 = 0.1 + 0.2f/(1.0f + 1000.0f*OVOL1[id]);
        float p2 = 0.5 + 0.2f/(1.0f + 200.0f*OVOL1[id]);
        float tmpp = RNG();
        if(tmpp < p1)
        {
            *ctype1 = 2;
            *ctype2 = 2;
        }
        else if(tmpp < (p1+p2))
        {
            *ctype1 = *ctype2 = 3;
        }
        else
        {
            *ctype1 = 2;
            *ctype2 = 3;
        }

    }
}