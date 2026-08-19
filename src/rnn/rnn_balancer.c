// forward pass of the GRU trained by train_rnn.py

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "rnn_balancer.h"

// upper bounds on what a weights file may declare. they only size the hot-path
// scratch below, which is what keeps rebal_rnn_decide() allocation-free
#define NF_MAX  32
#define NH_MAX  256

// the two feature sets train_rnn.py can emit. "full" additionally needs the
// per-slice work histogram, which costs main.c an extra Allreduce per event
#define NF_BASE 13
#define NF_FULL 16

// must match main.c: only used to rebuild the step the residual corrects
#ifndef REBALANCE_TOL
#define REBALANCE_TOL   0.02
#endif
#ifndef REBALANCE_GAIN
#define REBALANCE_GAIN  0.5
#endif

// shape and weights, all scanned from the file at startup
static int    g_nf, g_nh, g_max_step, g_steps_per_growth;
static float *g_w_ih, *g_w_hh, *g_b_ih, *g_b_hh, *g_w_out;
static float  g_b_out;
static int    g_ready = 0;

struct RebalRNN {
    int    nprocs;
    int    ncuts;              // nprocs - 1 interior cuts
    float *h;                  // [ncuts][g_nh] hidden state, carried across events
    double *prev_imb;          // [ncuts] last local imbalance, feature 1
    double *prev_step;         // [ncuts] last applied step, feature 7
    double *prop;              // [ncuts] scratch: proposed steps this event
    int    *target;            // [nprocs+1] scratch: work-CDF target cuts
};

// weight file scanner

// next whitespace-delimited token, skipping '#' comments to end of line
static int next_tok(FILE *f, char *buf, int n)
{
    int c;
    for (;;) {
        do { c = fgetc(f); } while (c != EOF && isspace(c));
        if (c == EOF) return 0;
        if (c != '#') break;
        while (c != EOF && c != '\n') c = fgetc(f);
    }
    int i = 0;
    while (c != EOF && !isspace(c)) {
        if (i < n - 1) buf[i++] = (char)c;
        c = fgetc(f);
    }
    buf[i] = '\0';
    return i > 0;
}

static int scan_ints(FILE *f, int *out)
{
    char t[64];
    if (!next_tok(f, t, sizeof t)) return 0;
    char *end;
    long v = strtol(t, &end, 10);
    if (*end) return 0;
    *out = (int)v;
    return 1;
}

static int scan_floats(FILE *f, float *out, size_t n)
{
    char t[64];
    for (size_t i = 0; i < n; i++) {
        if (!next_tok(f, t, sizeof t)) return 0;
        char *end;
        double v = strtod(t, &end);
        if (*end) return 0;
        out[i] = (float)v;
    }
    return 1;
}

static void free_weights(void)
{
    free(g_w_ih); free(g_w_hh); free(g_b_ih); free(g_b_hh); free(g_w_out);
    g_w_ih = g_w_hh = g_b_ih = g_b_hh = g_w_out = NULL;
    g_ready = 0;
}

// keys may come in any order, but the four dimensions must precede the arrays
// they size, which is how train_rnn.py writes them
int rebal_rnn_load_weights(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    int nf = 0, nh = 0, ms = 0, spg = 0, rc = 0;
    float *w_ih = NULL, *w_hh = NULL, *b_ih = NULL, *b_hh = NULL, *w_out = NULL;
    float b_out = 0.0f;
    int got_b_out = 0;
    char key[64];

    while (next_tok(f, key, sizeof key)) {
        if      (!strcmp(key, "features"))         { if (!scan_ints(f, &nf))  { rc = -2; break; } }
        else if (!strcmp(key, "hidden"))           { if (!scan_ints(f, &nh))  { rc = -2; break; } }
        else if (!strcmp(key, "max_step"))         { if (!scan_ints(f, &ms))  { rc = -2; break; } }
        else if (!strcmp(key, "steps_per_growth")) { if (!scan_ints(f, &spg)) { rc = -2; break; } }
        else {
            // every remaining key is an array sized by the dimensions above
            if (nf <= 0 || nh <= 0) { rc = -3; break; }
            if (nf > NF_MAX || nh > NH_MAX) { rc = -4; break; }
            if (nf != NF_BASE && nf != NF_FULL) { rc = -4; break; }

            size_t n;
            float **dst;
            if      (!strcmp(key, "w_ih"))  { n = (size_t)3 * nh * nf; dst = &w_ih; }
            else if (!strcmp(key, "w_hh"))  { n = (size_t)3 * nh * nh; dst = &w_hh; }
            else if (!strcmp(key, "b_ih"))  { n = (size_t)3 * nh;      dst = &b_ih; }
            else if (!strcmp(key, "b_hh"))  { n = (size_t)3 * nh;      dst = &b_hh; }
            else if (!strcmp(key, "w_out")) { n = (size_t)nh;          dst = &w_out; }
            else if (!strcmp(key, "b_out")) {
                if (!scan_floats(f, &b_out, 1)) { rc = -2; break; }
                got_b_out = 1;
                continue;
            }
            else { rc = -5; break; }           // unknown key

            free(*dst);
            *dst = (float *)malloc(n * sizeof(float));
            if (!*dst) { rc = -6; break; }
            if (!scan_floats(f, *dst, n)) { rc = -2; break; }
        }
    }
    fclose(f);

    if (!rc && (!w_ih || !w_hh || !b_ih || !b_hh || !w_out || !got_b_out ||
                ms <= 0 || spg <= 0))
        rc = -7;                               // incomplete file

    if (rc) {
        free(w_ih); free(w_hh); free(b_ih); free(b_hh); free(w_out);
        return rc;                             // previous weights left alone
    }

    free_weights();
    g_nf = nf; g_nh = nh; g_max_step = ms; g_steps_per_growth = spg;
    g_w_ih = w_ih; g_w_hh = w_hh; g_b_ih = b_ih; g_b_hh = b_hh;
    g_w_out = w_out; g_b_out = b_out;
    g_ready = 1;
    return 0;
}

// GRU cell + residual head

static double sigmoid_(double x) {
    return 1.0 / (1.0 + exp(-x));
}

static double clampd(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// PyTorch nn.GRUCell layout, rows ordered [reset, update, new]. the reset gate
// multiplies the hidden term inside the tanh, after its bias, not outside
void rebal_rnn_cell(const float *x, const float *h_in,
                    float *h_out, float *resid_out)
{
    if (!g_ready) { *resid_out = 0.0f; return; }
    const int NF = g_nf, NH = g_nh;

    double gi[3 * NH_MAX], gh[3 * NH_MAX];
    for (int j = 0; j < 3 * NH; j++) {
        double a = g_b_ih[j];
        const float *w = g_w_ih + (size_t)j * NF;
        for (int f = 0; f < NF; f++) a += (double)w[f] * (double)x[f];
        gi[j] = a;

        double b = g_b_hh[j];
        const float *v = g_w_hh + (size_t)j * NH;
        for (int k = 0; k < NH; k++) b += (double)v[k] * (double)h_in[k];
        gh[j] = b;
    }

    for (int j = 0; j < NH; j++) {
        double r = sigmoid_(gi[j]           + gh[j]);
        double z = sigmoid_(gi[NH + j]      + gh[NH + j]);
        double n = tanh    (gi[2 * NH + j]  + r * gh[2 * NH + j]);
        h_out[j] = (float)((1.0 - z) * n + z * (double)h_in[j]);
    }

    double resid = g_b_out;
    for (int j = 0; j < NH; j++) resid += (double)g_w_out[j] * (double)h_out[j];
    *resid_out = (float)resid;
}

// additive in pre-tanh space, so a zero residual returns exactly heur_step and
// the tanh keeps |step| <= max_step structurally
float rebal_rnn_step_from_residual(double heur_step, double resid)
{
    double u = clampd(heur_step / (double)g_max_step, -0.995, 0.995);
    double base = 0.5 * log((1.0 + u) / (1.0 - u));            // atanh(u)
    return (float)(tanh(base + resid) * (double)g_max_step);
}

// the heuristic being corrected: main.c's rule, one cut

double rebal_rnn_heuristic_step(double t_lo, double t_hi, int w_lo, int w_hi)
{
    if (!(t_lo > 0.0) || !(t_hi > 0.0)) return 0.0;

    double mean = 0.5 * (t_lo + t_hi);
    if (mean < 1e-12) mean = 1e-12;
    double imb = (t_lo - t_hi) / mean;
    if (fabs(imb) < REBALANCE_TOL) return 0.0; // deadband

    double w_slow = (imb > 0.0) ? (double)w_lo : (double)w_hi;
    double slow   = (imb > 0.0) ? t_lo : t_hi;
    if (slow < 1e-12) slow = 1e-12;
    double raw = REBALANCE_GAIN * w_slow * fabs(t_lo - t_hi) / (2.0 * slow);

    double step = floor(raw + 0.5);
    if (step < 1.0) step = 1.0; // always move once
    if (step > (double)g_max_step) step = (double)g_max_step;
    return (imb < 0.0) ? -step : step;
}

// work-histogram quantities, "full" feature set only

// mean work per slice over [lo,hi), 0 if empty
static double range_mean(const long long *work, int lo, int hi)
{
    if (hi <= lo) return 0.0;
    double s = 0.0;
    for (int k = lo; k < hi; k++) s += (double)work[k];
    return s / (double)(hi - lo);
}

// work in the max_step slices inside a band's cut-side edge over the band
// mean, minus one. scale-free, so the histogram's units do not matter
static double marginal(const long long *work, int band_l, int band_r,
                       int edge_l, int edge_r)
{
    if (edge_l < band_l) edge_l = band_l;
    if (edge_r > band_r) edge_r = band_r;
    double band = range_mean(work, band_l, band_r);
    double edge = range_mean(work, edge_l, edge_r);
    if (band < 1e-12) return 0.0;
    return clampd(edge / band - 1.0, -1.0, 3.0);
}

void rebal_rnn_target_cuts(const long long *work, int n_slice, int nprocs,
                           int min_w, int *target)
{
    // cuts at the b/P quantiles of the work CDF
    double total = 0.0;
    for (int k = 0; k < n_slice; k++) total += (double)work[k];

    target[0] = 0;
    target[nprocs] = n_slice;

    if (total <= 0.0) {
        for (int b = 1; b < nprocs; b++)
            target[b] = (int)((long long)b * n_slice / nprocs);
    } else {
        for (int b = 1; b < nprocs; b++) {
            double want = (double)b / (double)nprocs;
            // leftmost i with cdf[i] >= want, matching torch.searchsorted
            double acc = 0.0;
            int idx = n_slice;
            for (int k = 0; k < n_slice; k++) {
                acc += (double)work[k];
                if (acc / total >= want) { idx = k; break; }
            }
            target[b] = idx;
        }
    }

    for (int b = 1; b < nprocs; b++) {
        int lo = target[b - 1] + min_w;
        int hi = n_slice - (nprocs - b) * min_w;
        if (target[b] < lo) target[b] = lo;
        if (target[b] > hi) target[b] = hi;
    }
}

// features, mirroring cut_features() in sim_env.py

void rebal_rnn_features(double t_lo, double t_hi, double global_imb,
                        int w_lo, int w_hi, int n_slice, int nprocs, int min_w,
                        double prev_imb, double prev_step,
                        double phase, double burst,
                        const long long *work, int cut_pos, int target_pos,
                        float *out)
{
    double mean = 0.5 * (t_lo + t_hi);
    if (mean < 1e-12) mean = 1e-12;
    double imb = (t_lo - t_hi) / mean; // >0 => band below is slower
    double wmean = (double)n_slice / (double)nprocs;
    double mw = (double)(min_w > 1 ? min_w : 1);
    const double ms = (double)g_max_step;

    out[0]  = (float)clampd(imb, -2.0, 2.0);
    out[1]  = (float)clampd(imb - prev_imb, -2.0, 2.0);
    out[2]  = (float)clampd(global_imb, 0.0, 2.0);
    out[3]  = (float)clampd((double)w_lo / wmean - 1.0, -1.0, 3.0);
    out[4]  = (float)clampd((double)w_hi / wmean - 1.0, -1.0, 3.0);
    out[5]  = (float)clampd(((double)w_lo - min_w) / wmean, 0.0, 3.0);
    out[6]  = (float)clampd(((double)w_hi - min_w) / wmean, 0.0, 3.0);
    out[7]  = (float)clampd(prev_step / ms, -1.0, 1.0);
    out[8]  = (float)clampd(((double)w_lo - min_w) / mw, 0.0, 4.0);
    out[9]  = (float)clampd(((double)w_hi - min_w) / mw, 0.0, 4.0);
    out[10] = (float)phase;
    out[11] = (float)burst;
    out[12] = (float)clampd(rebal_rnn_heuristic_step(t_lo, t_hi, w_lo, w_hi) / ms,
                            -1.0, 1.0);

    if (g_nf == NF_FULL) {
        int band_lo_l = cut_pos - w_lo, band_lo_r = cut_pos;
        int band_hi_l = cut_pos,        band_hi_r = cut_pos + w_hi;
        out[13] = (float)marginal(work, band_lo_l, band_lo_r,
                                  cut_pos - g_max_step, cut_pos);
        out[14] = (float)marginal(work, band_hi_l, band_hi_r,
                                  cut_pos, cut_pos + g_max_step);
        out[15] = (float)clampd((double)(target_pos - cut_pos) / ms, -1.0, 1.0);
    }
}

// lifecycle

RebalRNN *rebal_rnn_create(int nprocs, const char *weights_path)
{
    if (nprocs < 2) return NULL;
    if (rebal_rnn_load_weights(weights_path) != 0) return NULL;

    RebalRNN *r = (RebalRNN *)calloc(1, sizeof(RebalRNN));
    if (!r) return NULL;
    r->nprocs = nprocs;
    r->ncuts  = nprocs - 1;
    r->h         = (float  *)calloc((size_t)r->ncuts * g_nh, sizeof(float));
    r->prev_imb  = (double *)calloc((size_t)r->ncuts, sizeof(double));
    r->prev_step = (double *)calloc((size_t)r->ncuts, sizeof(double));
    r->prop      = (double *)calloc((size_t)r->ncuts, sizeof(double));
    r->target    = (int    *)calloc((size_t)nprocs + 1, sizeof(int));
    if (!r->h || !r->prev_imb || !r->prev_step || !r->prop || !r->target) {
        rebal_rnn_destroy(r);
        return NULL;
    }
    return r;
}

void rebal_rnn_destroy(RebalRNN *r)
{
    if (!r) return;
    free(r->h); free(r->prev_imb); free(r->prev_step);
    free(r->prop); free(r->target);
    free(r);
}

void rebal_rnn_reset(RebalRNN *r)
{
    if (!r) return;
    memset(r->h,         0, (size_t)r->ncuts * g_nh * sizeof(float));
    memset(r->prev_imb,  0, (size_t)r->ncuts * sizeof(double));
    memset(r->prev_step, 0, (size_t)r->ncuts * sizeof(double));
}

int rebal_rnn_max_step(void)   { return g_ready ? g_max_step : 0; }
int rebal_rnn_n_features(void) { return g_ready ? g_nf : 0; }

// decision

static int round_half_away(double x)
{
    // matches train_rnn.py's hard rollout. lround() would honour the current
    // FP rounding mode; this does not
    return (int)(x >= 0.0 ? floor(x + 0.5) : ceil(x - 0.5));
}

int rebal_rnn_decide(RebalRNN *r, const double *t_rank, const int *cut,
                     int n_slice, int min_w, int event,
                     const long long *work, int *newcut)
{
    if (!g_ready) return 0;
    const int P = r->nprocs;

    for (int b = 0; b <= P; b++) newcut[b] = cut[b];

    // the sweep's clamp bounds stay ordered only from a legal configuration.
    // below this the decomposition is already infeasible, so leave the cuts
    if (n_slice < P * min_w) return 0;

    // zeros here would be a different input distribution than was fitted
    if (g_nf == NF_FULL && !work) return 0;

    // no signal yet, or the clock went backwards
    double sum = 0.0, tmax = 0.0;
    for (int p = 0; p < P; p++) {
        if (!(t_rank[p] > 0.0)) return 0;
        sum += t_rank[p];
        if (t_rank[p] > tmax) tmax = t_rank[p];
    }
    double gmean = sum / (double)P;
    double global_imb = (gmean > 1e-12) ? (tmax - gmean) / gmean : 0.0;

    // the burst lands in the two intervals after a cycle boundary
    int ph = event % g_steps_per_growth;
    double phase = (double)ph / (double)g_steps_per_growth;
    double burst = (event > 0 && (ph == 0 || ph == 1)) ? 1.0 : 0.0;

    if (g_nf == NF_FULL)
        rebal_rnn_target_cuts(work, n_slice, P, min_w, r->target);

    // pass 1: evaluated on the pre-move decomposition so every cut sees a
    // consistent snapshot
    double *prop = r->prop;

    for (int b = 1; b < P; b++) {
        int ci = b - 1;
        int w_lo = cut[b] - cut[b - 1];
        int w_hi = cut[b + 1] - cut[b];
        float x[NF_MAX], hnew[NH_MAX], resid;

        rebal_rnn_features(t_rank[b - 1], t_rank[b], global_imb, w_lo, w_hi,
                           n_slice, P, min_w,
                           r->prev_imb[ci], r->prev_step[ci], phase, burst,
                           work, cut[b],
                           (g_nf == NF_FULL) ? r->target[b] : 0,
                           x);

        rebal_rnn_cell(x, r->h + (size_t)ci * g_nh, hnew, &resid);
        memcpy(r->h + (size_t)ci * g_nh, hnew, g_nh * sizeof(float));

        // recomputed rather than returned, to keep rebal_rnn_features() one-way
        double mean = 0.5 * (t_rank[b - 1] + t_rank[b]);
        r->prev_imb[ci] = (mean > 1e-12) ? (t_rank[b - 1] - t_rank[b]) / mean : 0.0;

        prop[ci] = (double)resid;
    }

    // pass 2: main.c's sweep verbatim
    int moved = 0;
    for (int b = 1; b < P; b++) {
        int ci = b - 1;
        int lo = newcut[b - 1] + min_w;
        int hi = cut[b + 1] - min_w;
        if (lo > hi) {                          // no legal position, leave it
            r->prev_step[ci] = 0.0;
            continue;
        }

        int w_lo = cut[b] - newcut[b - 1];      // updated left neighbour
        int w_hi = cut[b + 1] - cut[b];         // not yet updated
        double heur = rebal_rnn_heuristic_step(t_rank[b - 1], t_rank[b], w_lo, w_hi);
        float raw = rebal_rnn_step_from_residual(heur, prop[ci]);

        int nc = cut[b] - round_half_away((double)raw);   // +step moves down
        if (nc < lo) nc = lo;
        if (nc > hi) nc = hi;
        newcut[b] = nc;

        // feed back the applied step, not the proposed one
        r->prev_step[ci] = (double)(cut[b] - nc);
        moved += abs(cut[b] - nc);
    }
    newcut[0] = 0;
    newcut[P] = n_slice;
    return moved;
}
