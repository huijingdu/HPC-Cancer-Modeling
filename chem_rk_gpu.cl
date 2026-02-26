/*
Kernel to update chemical field at grid[gId]
Using chemical reaction-diffusion partial differential equation
Forward-Euler method with step size dt
*/

__kernel void chem_rk_gpu( 
__global float * chem1, 
__global int * grid, 
__const int nx0,
__const int ny0,
__const int nz0,
__const float dx0,
__const float dy0,
__const float dz0,
__const float dt,
__global float * chem11, 
__global int * grid1,
__global float * chem2, 
__global float * chem22,
__global int * grid2,
__global int * grid3,
__const int iter,
__global int * grid4,
__global float * vx,
__global float * vy,
__global float * vz)
{
     int gId = get_global_id(0); 
     int i, j, k;
     int i0, i1, j0, j1, k0, k1;
     int NX = nx0, NY = ny0, NZ = nz0;
     float DX = dx0, DY = dy0, DZ = dz0;
     float diff, reac, convect;
     float a_diff = 1.5f, a_reac = 0.01f, a_decay = 0.01f;
     float avg_grid, avg_grid1, avg_grid2, avg_grid3, avg_grid4;
     float diff_bottom;
     float alpha = 1.5f;
     float vx_d, vy_d, vz_d;

     i = gId%NX;
     k = gId/NX/NY;
     j = (gId - k*NX*NY)/NX;

     {
         i0 = i - 1;
         if(i == 0) i0 = NX - 1;
         i1 = i + 1;
         if(i == (NX - 1)) i1 = 0;

         j0 = j - 1;
         if(j == 0) j0 = NY - 1;
         j1 = j + 1;
         if(j == (NY - 1)) j1 = 0;

         k0 = k - 1;
         if(k == 0) k0 = 1;
         k1 = k + 1;
         if(k == (NZ - 1)) k1 = NZ - 1;

         avg_grid =  (grid[i0 + j*NX + k*NX*NY] + grid[i1 + j*NX + k*NX*NY] + grid[gId]
                    + grid[i + j0*NX + k*NX*NY] + grid[i + j1*NX + k*NX*NY] 
                    + grid[i + j*NX + k0*NX*NY] + grid[i + j*NX + k1*NX*NY]) / 7.0;
         avg_grid1 =  (grid1[i0 + j*NX + k*NX*NY] + grid1[i1 + j*NX + k*NX*NY] + grid1[gId]
                     + grid1[i + j0*NX + k*NX*NY] + grid1[i + j1*NX + k*NX*NY] 
                     + grid1[i + j*NX + k0*NX*NY] + grid1[i + j*NX + k1*NX*NY]) / 7.0;
         avg_grid2 =  (grid2[i0 + j*NX + k*NX*NY] + grid2[i1 + j*NX + k*NX*NY] + grid2[gId]
                     + grid2[i + j0*NX + k*NX*NY] + grid2[i + j1*NX + k*NX*NY] 
                     + grid2[i + j*NX + k0*NX*NY] + grid2[i + j*NX + k1*NX*NY]) / 7.0;
         avg_grid3 =  (grid3[i0 + j*NX + k*NX*NY] + grid3[i1 + j*NX + k*NX*NY] + grid3[gId]
                     + grid3[i + j0*NX + k*NX*NY] + grid3[i + j1*NX + k*NX*NY] 
                     + grid3[i + j*NX + k0*NX*NY] + grid3[i + j*NX + k1*NX*NY]) / 7.0;
         avg_grid4 =  (grid4[i0 + j*NX + k*NX*NY] + grid4[i1 + j*NX + k*NX*NY] + grid4[gId]
                     + grid4[i + j0*NX + k*NX*NY] + grid4[i + j1*NX + k*NX*NY] 
                     + grid4[i + j*NX + k0*NX*NY] + grid4[i + j*NX + k1*NX*NY]) / 7.0;

         a_diff = a_diff/(1.0f + 0.5f*avg_grid);
         a_reac = a_reac*(0.01f*avg_grid3 + 0.02f*avg_grid4);//0.0 0.075
         if(k == 0) 
         {
             diff_bottom = chem1[gId] - alpha*1.0*DZ*chem1[gId];
         }
         else
         {
             diff_bottom = chem1[i + j*NX + k0*NX*NY];
         }
         diff = (chem1[i0 + j*NX + k*NX*NY] + chem1[i1 + j*NX + k*NX*NY] - 2.0f*chem1[gId])/(1.0f*DX*DX)
              + (chem1[i + j0*NX + k*NX*NY] + chem1[i + j1*NX + k*NX*NY] - 2.0f*chem1[gId])/(1.0f*DY*DY) 
              + (diff_bottom + chem1[i + j*NX + k1*NX*NY] - 2.0f*chem1[gId])/(1.0f*DZ*DZ);
         diff = a_diff*diff;
         reac = a_reac;
         if(vx[gId] > 0.0)
         {
             vx_d = chem1[gId]*vx[gId] - chem1[i0 + j*NX + k*NX*NY]*vx[i0 + j*NX + k*NX*NY];
         }
         else if(vx[gId] < 0.0)
         {
             vx_d = chem1[i1 + j*NX + k*NX*NY]*vx[i1 + j*NX + k*NX*NY] - chem1[gId]*vx[gId];
         }
         else
         {
             vx_d = chem1[i1 + j*NX + k*NX*NY]*vx[i1 + j*NX + k*NX*NY] - chem1[i0 + j*NX + k*NX*NY]*vx[i0 + j*NX + k*NX*NY];
         }
         if(vy[gId] > 0.0)
         {
             vy_d = chem1[gId]*vy[gId] - chem1[i + j0*NX + k*NX*NY]*vy[i + j0*NX + k*NX*NY];
         }
         else if(vy[gId] < 0.0)
         {
             vy_d = chem1[i + j1*NX + k*NX*NY]*vy[i + j1*NX + k*NX*NY] - chem1[gId]*vy[gId];
         }
         else
         {
             vy_d = chem1[i + j1*NX + k*NX*NY]*vy[i + j1*NX + k*NX*NY] - chem1[i + j0*NX + k*NX*NY]*vy[i + j0*NX + k*NX*NY];
         }
         if(vz[gId] > 0.0)
         {
             vz_d = chem1[gId]*vz[gId] - chem1[i + j*NX + k0*NX*NY]*vz[i + j*NX + k0*NX*NY];
         }
         else if(vz[gId] < 0.0)
         {
             vz_d = chem1[i + j*NX + k1*NX*NY]*vz[i + j*NX + k1*NX*NY] - chem1[gId]*vz[gId];
         }
         else
         {
             vz_d = chem1[i + j*NX + k1*NX*NY]*vz[i + j*NX + k1*NX*NY] - chem1[i + j*NX + k0*NX*NY]*vz[i + j*NX + k0*NX*NY];
         }
         convect = vx_d/DX + vy_d/DY + vz_d/DZ;
         chem11[gId] = chem1[gId] + dt*(diff - convect + reac - a_decay*chem1[gId]);
         if(chem11[gId] < 0.0f) chem11[gId] = 0.0f;

         chem22[gId] = 0.0f;
     }
}
