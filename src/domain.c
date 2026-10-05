#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include "mpi_assert.h"
#include "sim_globals.h"
#include "domain.h"

int fit_bin_grid(float lo_x, float hi_x, float lo_y, float hi_y, float lo_z, float hi_z)
{
    g_bin_ox = lo_x - BIN_FIT_MARGIN;
    g_bin_oy = lo_y - BIN_FIT_MARGIN;
    g_bin_oz = lo_z - BIN_FIT_MARGIN;

    g_nbx = (int)ceilf((hi_x + BIN_FIT_MARGIN - g_bin_ox) / BIN_SIZE);
    g_nby = (int)ceilf((hi_y + BIN_FIT_MARGIN - g_bin_oy) / BIN_SIZE);
    g_nbz = (int)ceilf((hi_z + BIN_FIT_MARGIN - g_bin_oz) / BIN_SIZE);

    if (g_nbx < 1) g_nbx = 1;
    if (g_nby < 1) g_nby = 1;
    if (g_nbz < 1) g_nbz = 1;

    g_nbins = g_nbx * g_nby * g_nbz;
    return g_nbins;
}

void apply_x_cuts(const int * cut)
{
    for (int r = 0; r <= g_nprocs; r++) g_xcut[r] = cut[r];

    g_x_imin = cut[g_rank];
    g_x_imax = cut[g_rank + 1];
    g_x_lo = g_chem_ox + (float)cut[g_rank] * g_dx;
    g_x_hi = g_chem_ox + (float)cut[g_rank + 1] * g_dx;

    if (g_rank == 0) { g_x_lo = 0.0f; g_x_imin = 0; }
    if (g_rank == g_nprocs - 1) { g_x_hi = g_lx; g_x_imax = g_nx; }
}

static int balanced_cuts(const long long * pre, int nslice, int P, int min_w, int * cut)
{
    if (P == 1) { cut[0] = 0; cut[1] = nslice; return 1; }
    if ((long long)P * min_w > nslice) return 0;

    const long long INF = pre[nslice] + 1; // a band can never outweigh the whole domain

    long long (*dp)[nslice + 1] = malloc(sizeof(long long[P + 1][nslice + 1]));
    int (*back)[nslice + 1] = malloc(sizeof(int[P + 1][nslice + 1]));
    assert(dp && back);

    for (int k = 0; k <= nslice; k++)
    {
        dp[1][k] = (k >= min_w) ? pre[k] : INF;
        back[1][k] = 0;
    }
    for (int p = 2; p <= P; p++)
    {
        for (int k = 0; k <= nslice; k++)
        {
            dp[p][k] = INF;
            back[p][k] = -1;

            int jmin = (p - 1) * min_w;
            int jmax = k - min_w;
            for (int j = jmin; j <= jmax; j++)
            {
                if (dp[p - 1][j] >= INF) continue;
                long long last = pre[k] - pre[j];
                long long m = (dp[p - 1][j] > last) ? dp[p - 1][j] : last;
                if (m < dp[p][k]) { dp[p][k] = m; back[p][k] = j; }
            }
        }
    }

    int ok = (dp[P][nslice] < INF);
    if (ok)
    {
        int k = nslice;
        cut[P] = nslice;
        for (int p = P; p >= 1; p--) { cut[p - 1] = back[p][k]; k = cut[p - 1]; }
        cut[0] = 0;
    }
    free(dp);
    free(back);
    return ok;
}

static void equal_width_cuts(int nslice, int P, int * cut)
{
    for (int r = 0; r <= P; r++) cut[r] = (int)((long long)r * nslice / P);
}

void assign_x_cuts_balanced(const float * x, const float * y, const float * z,
                            const int * ele_per_cell, int cell_no)
{
    int P = g_nprocs;
    int nslice = g_nx;

    if (P > REBAL_MAX_PROCS)
    {
        if (g_rank == 0)
            fprintf(stderr, "[mpi] %d ranks exceeds REBAL_MAX_PROCS = %d\n", P, REBAL_MAX_PROCS);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    g_min_w = (int)ceilf(GHOST_DEPTH_F / g_dx);
    if (g_min_w < 1) g_min_w = 1;

    int * cut = (int *)malloc((size_t)(P + 1) * sizeof(int));
    long long * pre = (long long *)calloc((size_t)nslice + 1, sizeof(long long));
    long long * cpre = (long long *)calloc((size_t)nslice + 1, sizeof(long long));
    float * cx = (float *)malloc((size_t)cell_no * sizeof(float));
    float * cy = (float *)malloc((size_t)cell_no * sizeof(float));
    float * cz = (float *)malloc((size_t)cell_no * sizeof(float));
    int * ck = (int *)malloc((size_t)cell_no * sizeof(int));
    assert(cut && pre && cpre && cx && cy && cz && ck);

    float lo_x = 0.0f, hi_x = 0.0f, lo_y = 0.0f, hi_y = 0.0f, lo_z = 0.0f, hi_z = 0.0f;
    int seen = 0;
    for (int ci = 0; ci < cell_no; ci++)
    {
        ck[ci] = -1;
        int n = ele_per_cell[ci];
        if (n <= 0) continue;

        float sx = 0.0f, sy = 0.0f, sz = 0.0f;
        for (int j = 0; j < n; j++)
        {
            int e = ci * MAX_ELE + j;
            sx += x[e]; sy += y[e]; sz += z[e];
        }
        cx[ci] = sx / (float)n;
        cy[ci] = sy / (float)n;
        cz[ci] = sz / (float)n;

        if (!seen)
        {
            lo_x = hi_x = cx[ci]; lo_y = hi_y = cy[ci]; lo_z = hi_z = cz[ci];
            seen = 1;
        }
        else
        {
            if (cx[ci] < lo_x) lo_x = cx[ci];
            if (cx[ci] > hi_x) hi_x = cx[ci];
            if (cy[ci] < lo_y) lo_y = cy[ci];
            if (cy[ci] > hi_y) hi_y = cy[ci];
            if (cz[ci] < lo_z) lo_z = cz[ci];
            if (cz[ci] > hi_z) hi_z = cz[ci];
        }

        int k = (int)floorf((cx[ci] - g_chem_ox) / g_dx);
        if (k < 0) k = 0;
        if (k >= nslice) k = nslice - 1;
        ck[ci] = k;
    }

    int nbx = 1, nby = 1, nbz = 1;
    float ox = 0.0f, oy = 0.0f, oz = 0.0f;
    int * bin_ele = NULL;
    int * c_bx = (int *)malloc((size_t)cell_no * sizeof(int));
    int * c_by = (int *)malloc((size_t)cell_no * sizeof(int));
    int * c_bz = (int *)malloc((size_t)cell_no * sizeof(int));
    assert(c_bx && c_by && c_bz);

    if (seen)
    {
        ox = lo_x - BIN_SIZE;
        oy = lo_y - BIN_SIZE;
        oz = lo_z - BIN_SIZE;
        nbx = (int)ceilf((hi_x + BIN_SIZE - ox) / BIN_SIZE);
        nby = (int)ceilf((hi_y + BIN_SIZE - oy) / BIN_SIZE);
        nbz = (int)ceilf((hi_z + BIN_SIZE - oz) / BIN_SIZE);
        if (nbx < 1) nbx = 1;
        if (nby < 1) nby = 1;
        if (nbz < 1) nbz = 1;
    }
    bin_ele = (int *)calloc((size_t)nbx * nby * nbz, sizeof(int));
    assert(bin_ele);

    for (int ci = 0; ci < cell_no; ci++)
    {
        if (ck[ci] < 0) continue;
        int bx = (int)floorf((cx[ci] - ox) / BIN_SIZE);
        int by = (int)floorf((cy[ci] - oy) / BIN_SIZE);
        int bz = (int)floorf((cz[ci] - oz) / BIN_SIZE);
        if (bx < 0) bx = 0; else if (bx >= nbx) bx = nbx - 1;
        if (by < 0) by = 0; else if (by >= nby) by = nby - 1;
        if (bz < 0) bz = 0; else if (bz >= nbz) bz = nbz - 1;
        c_bx[ci] = bx; c_by[ci] = by; c_bz[ci] = bz;
        bin_ele[(size_t)bx + (size_t)by * nbx + (size_t)bz * nbx * nby] += ele_per_cell[ci];
    }

    for (int ci = 0; ci < cell_no; ci++)
    {
        if (ck[ci] < 0) continue;

        long long neigh = 0;
        for (int dz = -1; dz <= 1; dz++)
        {
            int bz = c_bz[ci] + dz;
            if (bz < 0 || bz >= nbz) continue;
            for (int dy = -1; dy <= 1; dy++)
            {
                int by = c_by[ci] + dy;
                if (by < 0 || by >= nby) continue;
                for (int dx = -1; dx <= 1; dx++)
                {
                    int bx = c_bx[ci] + dx;
                    if (bx < 0 || bx >= nbx) continue;
                    neigh += bin_ele[(size_t)bx + (size_t)by * nbx + (size_t)bz * nbx * nby];
                }
            }
        }

        long long w = (long long)ele_per_cell[ci] * neigh;
        if (w < 1) w = 1; // an isolated cell still costs its own work items
        pre[ck[ci] + 1] += w;
        cpre[ck[ci] + 1] += 1;
    }
    for (int k = 0; k < nslice; k++) { pre[k + 1] += pre[k]; cpre[k + 1] += cpre[k]; }

    free(cx); free(cy); free(cz); free(ck);
    free(c_bx); free(c_by); free(c_bz); free(bin_ele);

    int balanced = balanced_cuts(pre, nslice, P, g_min_w, cut);
    if (!balanced)
    {
        equal_width_cuts(nslice, P, cut);
        if (P > 1 && nslice / P < g_min_w)
        {
            if (g_rank == 0)
                fprintf(stderr, "[mpi] %d ranks over %d chem slices gives a band below the "
                                "%d slice halo floor. use at most %d ranks\n",
                        P, nslice, g_min_w, nslice / g_min_w);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    apply_x_cuts(cut);

    if (g_rank == 0)
    {
        long long wtot = pre[nslice], ctot = cpre[nslice];
        printf("[balance] work balanced X partition: %d slices of %.3f, min band %d%s\n",
               nslice, g_dx, g_min_w, balanced ? "" : "  [INFEASIBLE, equal width fallback]");

        double maxshare = 0.0;
        for (int r = 0; r < P; r++)
        {
            long long w = pre[cut[r + 1]] - pre[cut[r]];
            long long c = cpre[cut[r + 1]] - cpre[cut[r]];
            double ws = wtot ? 100.0 * (double)w / (double)wtot : 0.0;
            if (ws > maxshare) maxshare = ws;
            printf("[balance]   rank %d: slices [%d,%d) n=%d  cells=%lld (%.1f%%)  work=%.1f%%\n",
                   r, cut[r], cut[r + 1], cut[r + 1] - cut[r], c,
                   ctot ? 100.0 * (double)c / (double)ctot : 0.0, ws);
        }

        long long peak = 0;
        int peak_k = 0;
        for (int k = 0; k < nslice; k++)
        {
            long long sw = pre[k + 1] - pre[k];
            if (sw > peak) { peak = sw; peak_k = k; }
        }
        double ideal = 100.0 / (double)P;
        printf("[balance]   ideal %.1f%%, achieved max %.1f%%, headroom %d slices; "
               "peak slice %d = %.1f%%%s\n",
               ideal, maxshare, nslice - P * g_min_w, peak_k,
               wtot ? 100.0 * (double)peak / (double)wtot : 0.0,
               (maxshare > 1.5 * ideal) ? "  [1-D X cannot balance further]" : "");

#if REBALANCE_FR > 0
        if (REBALANCE_MAX_STEP >= g_min_w)
            fprintf(stderr, "[balance] WARNING: REBALANCE_MAX_STEP %d >= min band %d, "
                            "a rebalance could push a cell past its neighbour\n",
                    (int)REBALANCE_MAX_STEP, g_min_w);
#endif
        fflush(stdout);
    }

    free(pre); free(cpre); free(cut);
}
