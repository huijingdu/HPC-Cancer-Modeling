// recurrent X-cut load-balancing controller for main.c's rebalance block. the
// network emits a residual on the proportional rule, so zero reproduces it.
// weights are scanned from a text file at startup, not compiled in
#ifndef REBAL_RNN_BALANCER_H
#define REBAL_RNN_BALANCER_H


typedef struct RebalRNN RebalRNN;

// scans weights_path, then allocates one hidden state per interior cut. NULL if
// nprocs < 2, the file is unreadable or malformed, or the allocation fails --
// all of which the caller should treat as "use the heuristic", not as fatal
RebalRNN *rebal_rnn_create(int nprocs, const char *weights_path);

void rebal_rnn_destroy(RebalRNN *r);

// decides one rebalance event, returning the slices moved. event is
// iter/REBALANCE_FR; work is read only by the "full" set, otherwise NULL.
// newcut comes back sorted, pinned at 0/n_slice, every band >= min_w
int rebal_rnn_decide(RebalRNN *r, const double *t_rank, const int *cut,
                     int n_slice, int min_w, int event,
                     const long long *work, int *newcut);

// drop the recurrent state, e.g. after a restart from checkpoint
void rebal_rnn_reset(RebalRNN *r);

// slices one cut can move per event. single-hop migration needs this < min_w.
// only meaningful once a controller exists; 0 before that
int rebal_rnn_max_step(void);

// 13 = "base", 16 = "full", which also needs the work histogram. 0 if unloaded
int rebal_rnn_n_features(void);

// --- exposed for the parity test against PyTorch, not used by the solver ---

// one GRU cell + output head, emitting the bare residual
void rebal_rnn_cell(const float *x, const float *h_in,
                    float *h_out, float *resid_out);

// combines a residual with the heuristic step, additively in pre-tanh space
float rebal_rnn_step_from_residual(double heur_step, double resid);

// main.c reads w_lo off the partially updated cut array during its sweep, so
// the caller must pass the already-updated width
double rebal_rnn_heuristic_step(double t_lo, double t_hi, int w_lo, int w_hi);

// feature vector for one cut, matching cut_features() in sim_env.py
void rebal_rnn_features(double t_lo, double t_hi, double global_imb,
                        int w_lo, int w_hi, int n_slice, int nprocs, int min_w,
                        double prev_imb, double prev_step,
                        double phase, double burst,
                        const long long *work, int cut_pos, int target_pos,
                        float *out);

// cut positions that equalize the work CDF, the target reported by feature 15
void rebal_rnn_target_cuts(const long long *work, int n_slice, int nprocs,
                           int min_w, int *target);

// scan a weights file. returns 0 on success, and on any error leaves whatever
// was loaded before untouched. rebal_rnn_create() calls this for you
int rebal_rnn_load_weights(const char *path);

#endif
