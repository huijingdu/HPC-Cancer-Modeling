"""Differentiable surrogate of the X-cut load balancer in main.c"""

import torch


# cadence of main.c, one env step per rebalance interval
REBALANCE_FR = 100
GROWTH_FR = 1000
STEPS_PER_GROWTH = GROWTH_FR // REBALANCE_FR        # = 10

MAX_STEP = 8                # must equal REBALANCE_MAX_STEP in main.c


# scenario sampling

class Scenario:
    def __init__(self, batch, nprocs, nx, min_w, device, gen, dtype=torch.float32):
        self.B = batch
        self.P = nprocs
        self.nx = nx
        self.min_w = min_w
        self.device = device
        self.dtype = dtype
        self.t_step = 0                 # rebalance events elapsed, for the phase feature
        g = gen
        B = batch

        def U(lo, hi, *shape):
            return lo + (hi - lo) * torch.rand(*shape, generator=g,
                                               device=device, dtype=dtype)

        # initial cell profile, 1-3 gaussian blobs, tight core to diffuse spread
        k = torch.arange(nx, device=device, dtype=dtype).view(1, nx)
        nmodes = 3
        # widths picked so peak/mean slice work brackets the 1.79 seen on IC_37K.
        # the old 0.04-0.30 gave 3.57, about twice as peaked as anything real
        centre = U(0.15, 0.85, B, nmodes, 1) * nx
        width = U(0.08, 0.70, B, nmodes, 1) * nx
        amp = U(0.2, 1.0, B, nmodes, 1)
        keep = (torch.rand(B, nmodes, 1, generator=g, device=device) < 0.6).to(dtype)
        keep[:, 0, :] = 1.0
        prof = (amp * keep * torch.exp(-0.5 * ((k.view(1, 1, nx) - centre) / width) ** 2)).sum(1)
        prof = prof + U(0.02, 0.35, B, 1)          # background floor (see above)
        # covers IC_10K up to IC_435K
        total = U(1.0e4, 4.4e5, B, 1)
        self.c = prof / prof.sum(1, keepdim=True).clamp_min(1e-9) * total

        # mean elements per cell, capped by MAX_ELE in the real code
        self.e = U(6.0, 14.0, B, nx)

        # division propensity, uneven in x, which is what makes the best cuts
        # move. starts at zero because IC_37K turns out to be nearly static
        gc = U(0.1, 0.9, B, 1) * nx
        gw = U(0.05, 0.35, B, 1) * nx
        self.growth = U(0.0, 0.20, B, 1) * torch.exp(
            -0.5 * ((k - gc) / gw) ** 2) + U(0.0, 0.02, B, 1)

        # drift and diffusion, per 100-iteration interval
        self.drift = U(-0.025, 0.025, B, 1)        # slices per rebalance interval
        self.spread = U(0.0, 0.045, B, 1)          # diffusion per interval

        # cost coefficients, randomized so no one machine's balance gets baked in
        self.cA = U(0.5, 1.5, B, 1)
        self.cB = U(0.2, 1.0, B, 1)
        self.cC = U(0.1, 0.8, B, 1)
        self.cD = U(0.0, 0.06, B, 1)               # fixed per-rank overhead

        # growth and division burst, one interval in ten. main.c does not
        # subtract it, so the controller sees it
        self.burst = U(0.0, 3.0, B, 1)

        # per-rank speed spread and jitter. jitter is why the rule needs a deadband
        self.speed = 1.0 + U(-0.04, 0.04, B, nprocs)
        self.noise = U(0.005, 0.05, B, 1)

        # rho smoothing half-width, the 3x3x3 bin neighbourhood projected on x
        self.rho_hw = max(1, int(round(1.5 * 12.0)))

    # dynamics

    def advance(self, gen):
        """Move the population on by one rebalance interval. Growth and division fire
    once every STEPS_PER_GROWTH calls; movement happens every call."""
        c = self.c
        if self.growth_fires_next():
            c = c * (1.0 + self.growth)
            self.e = (self.e + 0.6).clamp(4.0, 16.0)
        c = self._diffuse(c, self.spread)
        c = self._shift(c, self.drift)
        self.c = c.clamp_min(0.0)
        self.t_step += 1

    def growth_fires_next(self):
        """True if the interval about to be simulated contains a growth event."""
        return (self.t_step + 1) % STEPS_PER_GROWTH == 0

    def growth_fired_last(self):
        """True if the interval just measured carried growth cost. Two intervals, not
    one, matching the window rebal_rnn_decide() uses."""
        return self.t_step > 0 and (self.t_step % STEPS_PER_GROWTH) in (0, 1)

    def phase(self):
        """Position in the growth cycle, in [0,1). Feature 10."""
        return float(self.t_step % STEPS_PER_GROWTH) / float(STEPS_PER_GROWTH)

    @staticmethod
    def _shift_up(c):
        """c shifted toward +k with edge replication. Out of place, for autograd."""
        return torch.cat([c[:, :1], c[:, :-1]], dim=1)

    @staticmethod
    def _shift_dn(c):
        return torch.cat([c[:, 1:], c[:, -1:]], dim=1)

    @classmethod
    def _diffuse(cls, c, amount):
        a = amount.clamp(0.0, 0.5)
        return (1 - 2 * a) * c + a * cls._shift_up(c) + a * cls._shift_dn(c)

    @classmethod
    def _shift(cls, c, d):
        """Fractional-slice advection by linear interpolation."""
        f = d.clamp(-1.0, 1.0)
        fa = f.abs()
        src = torch.where(f > 0, cls._shift_up(c), cls._shift_dn(c))
        return (1 - fa) * c + fa * src

    # cost model

    def _rho(self, ce):
        rho = self._box_smooth(ce, self.rho_hw)
        return rho / rho.mean(1, keepdim=True).clamp_min(1e-9)

    def slice_work(self):
        """True per-slice compute cost, (B, nx). Only the loss sees this."""
        ce = self.c * self.e
        w = self.cA * ce * self._rho(ce) + self.cB * ce + self.cC * self.c
        return w / w.sum(1, keepdim=True).clamp_min(1e-9) * float(self.nx)

    def observed_work(self):
        """The histogram main.c can actually measure. It is assign_x_cuts_balanced()'s
    A-term score, which under-prices diffuse cells."""
        ce = self.c * self.e
        w = ce * self._rho(ce)
        return w / w.sum(1, keepdim=True).clamp_min(1e-9) * float(self.nx)

    @staticmethod
    def _box_smooth(x, hw):
        if hw < 1:
            return x
        pad = torch.nn.functional.pad(x.unsqueeze(1), (hw, hw), mode='replicate')
        ker = torch.ones(1, 1, 2 * hw + 1, device=x.device, dtype=x.dtype) / (2 * hw + 1)
        return torch.nn.functional.conv1d(pad, ker).squeeze(1)

    def rank_times(self, cuts, tau, gen=None, hard=False, burst=False):
        """Per-rank compute time for the interval, (B, P+1) -> (B, P). Soft mode is a
    sigmoid window so gradients reach the cuts; hard mode is the indicator."""
        w = self.slice_work()
        if burst:
            cw = self.c / self.c.sum(1, keepdim=True).clamp_min(1e-9) * float(self.nx)
            w = w + self.burst * cw
        kc = torch.arange(self.nx, device=self.device,
                          dtype=self.dtype).view(1, 1, self.nx) + 0.5
        lo = cuts[:, :-1].unsqueeze(2)                     # (B, P, 1)
        hi = cuts[:, 1:].unsqueeze(2)
        if hard:
            m = ((kc >= lo) & (kc < hi)).to(self.dtype)
        else:
            m = torch.sigmoid((kc - lo) / tau) * torch.sigmoid((hi - kc) / tau)
        t = (m * w.unsqueeze(1)).sum(2) + self.cD          # (B, P)
        t = t * self.speed
        if gen is not None:
            jitter = 1.0 + self.noise * torch.randn(t.shape, generator=gen,
                                                    device=self.device,
                                                    dtype=self.dtype)
            t = t * jitter.clamp_min(0.2)
        return t.clamp_min(1e-6)

    # initial decomposition

    def initial_cuts(self):
        """Reproduce assign_x_cuts_balanced(), equalizing the A-term score only, so
    episodes start from the bias the real run starts with."""
        return self.legalize(self.quantile_cuts(self.observed_work()))

    def quantile_cuts(self, w):
        """Cuts at the b/P quantiles of the work CDF, (B, P+1). This is the target
    feature 14 reports; it tracks the min-max DP closely and costs far less."""
        pre = torch.cumsum(w, dim=1)
        pre = pre / pre[:, -1:].clamp_min(1e-9)
        cuts = torch.zeros(self.B, self.P + 1, device=self.device, dtype=self.dtype)
        cuts[:, self.P] = self.nx
        for b in range(1, self.P):
            target = torch.full((self.B, 1), b / self.P,
                                device=self.device, dtype=self.dtype)
            idx = torch.searchsorted(pre.contiguous(), target)
            cuts[:, b] = idx.squeeze(1).to(self.dtype)
        return cuts

    def legalize(self, cuts):
        """Enforce cut[0]=0, cut[P]=nx, monotone, every band >= min_w. One sweep left
    to right; the upper clamp reserves room for the bands still to come."""
        P, nx, mw = self.P, self.nx, self.min_w
        out = [torch.zeros(self.B, device=self.device, dtype=self.dtype)]
        for b in range(1, P):
            lo = out[b - 1] + mw
            hi = torch.full((self.B,), float(nx - (P - b) * mw),
                            device=self.device, dtype=self.dtype)
            out.append(torch.max(torch.min(cuts[:, b], hi), lo))
        out.append(torch.full((self.B,), float(nx), device=self.device, dtype=self.dtype))
        return torch.stack(out, dim=1)


# features, must match rnn_balancer.c. BASE (13) uses only what main.c already
# gathers, FULL (16) adds histogram features and one more Allreduce

N_FEATURES_BASE = 13
N_FEATURES_FULL = 16


def heuristic_raw_step(t_lo, t_hi, w_lo, w_hi, tol=0.02, gain=0.5,
                       max_step=MAX_STEP):
    """The step main.c's rule proposes for one cut. w_lo comes from the updated
    newcut[b-1] and w_hi from the pre-move cut[b+1]; that asymmetry is the rule."""
    mean = 0.5 * (t_lo + t_hi)
    imb = (t_lo - t_hi) / mean.clamp_min(1e-12)

    w_slow = torch.where(imb > 0, w_lo, w_hi)
    slow = torch.where(imb > 0, t_lo, t_hi)

    raw = gain * w_slow * (t_lo - t_hi).abs() / (2.0 * slow.clamp_min(1e-12))
    step = torch.floor(raw + 0.5).clamp(1.0, float(max_step))
    step = torch.where(imb < 0, -step, step)
    step = torch.where(imb.abs() < tol, torch.zeros_like(step), step)
    step = torch.where((t_lo <= 0) | (t_hi <= 0), torch.zeros_like(step), step)
    return step


def heuristic_proposal(t_rank, cuts, P, max_step=MAX_STEP):
    """Feature 12, the rule's proposal from the pre-move geometry, (B, P-1).
    Pre-move on purpose, so the GRU sees one snapshot per event."""
    return heuristic_raw_step(t_rank[:, :-1], t_rank[:, 1:],
                              cuts[:, 1:P] - cuts[:, 0:P - 1],
                              cuts[:, 2:P + 1] - cuts[:, 1:P],
                              max_step=max_step)


def cut_features(t_rank, cuts, prev_imb, prev_step, nx, P, min_w, max_step,
                 phase=0.0, burst_fired=0.0, work=None, target_cuts=None,
                 n_features=N_FEATURES_BASE):
    """Per-cut observation, ((B, P-1, n_features), imb). Every feature is
    dimensionless, so there is no normalization table to drift against C."""
    lo = t_rank[:, :-1]                                    # rank b-1, below cut b
    hi = t_rank[:, 1:]                                     # rank b,   above cut b
    mean = 0.5 * (lo + hi)
    imb = (lo - hi) / mean.clamp_min(1e-12)                # >0 => below is slower

    gmean = t_rank.mean(1, keepdim=True).clamp_min(1e-12)
    gimb = (t_rank.max(1, keepdim=True).values - gmean) / gmean   # (B,1)

    w_lo = cuts[:, 1:P] - cuts[:, 0:P - 1]
    w_hi = cuts[:, 2:P + 1] - cuts[:, 1:P]
    wmean = float(nx) / float(P)
    mw = float(max(min_w, 1))

    feats = [
        imb.clamp(-2.0, 2.0),                              # 0  local imbalance
        (imb - prev_imb).clamp(-2.0, 2.0),                 # 1  its trend
        gimb.expand_as(imb).clamp(0.0, 2.0),               # 2  global imbalance
        (w_lo / wmean - 1.0).clamp(-1.0, 3.0),             # 3  band width below
        (w_hi / wmean - 1.0).clamp(-1.0, 3.0),             # 4  band width above
        ((w_lo - min_w) / wmean).clamp(0.0, 3.0),          # 5  halo slack below
        ((w_hi - min_w) / wmean).clamp(0.0, 3.0),          # 6  halo slack above
        (prev_step / max_step).clamp(-1.0, 1.0),           # 7  last applied step
        # 8,9: slack in min_w units. 5/6 measure it against mean band width
        ((w_lo - min_w) / mw).clamp(0.0, 4.0),             # 8  slack below, in min_w
        ((w_hi - min_w) / mw).clamp(0.0, 4.0),             # 9  slack above, in min_w
        # 10,11: one interval in ten carries a spike main.c does not subtract
        torch.full_like(imb, float(phase)),                # 10 phase in [0,1)
        torch.full_like(imb, float(burst_fired)),          # 11 burst in this interval
        # 12: what the rule would do. the net is a correction to it
        (heuristic_proposal(t_rank, cuts, P, max_step=max_step)
         / max_step).clamp(-1.0, 1.0),                     # 12 heuristic proposal
    ]

    if n_features == N_FEATURES_FULL:
        # 13,14: work in the slices the cut can reach, not the band average
        feats.append(_marginal(work, cuts, P, max_step, side='lo'))
        feats.append(_marginal(work, cuts, P, max_step, side='hi'))
        # 15: where the work histogram says this cut belongs
        feats.append(((target_cuts[:, 1:P] - cuts[:, 1:P]) / max_step).clamp(-1.0, 1.0))

    return torch.stack(feats, dim=2), imb


def _marginal(work, cuts, P, max_step, side):
    """Work in the max_step slices at a band's cut-side edge, over the band mean,
    minus one. Soft window so it stays differentiable."""
    B, nx = work.shape
    k = torch.arange(nx, device=work.device, dtype=work.dtype).view(1, 1, nx) + 0.5
    lo = cuts[:, 0:P - 1].unsqueeze(2)          # band below: [cut_{b-1}, cut_b)
    mid = cuts[:, 1:P].unsqueeze(2)
    hi = cuts[:, 2:P + 1].unsqueeze(2)

    if side == 'lo':
        band_l, band_r = lo, mid
        edge_l, edge_r = mid - max_step, mid
    else:
        band_l, band_r = mid, hi
        edge_l, edge_r = mid, mid + max_step

    band = ((k >= band_l) & (k < band_r)).to(work.dtype)
    edge = ((k >= edge_l) & (k < edge_r) & (k >= band_l) & (k < band_r)).to(work.dtype)

    w = work.unsqueeze(1)
    band_w = (band * w).sum(2)
    band_n = band.sum(2).clamp_min(1.0)
    edge_w = (edge * w).sum(2)
    edge_n = edge.sum(2).clamp_min(1.0)

    ratio = (edge_w / edge_n) / (band_w / band_n).clamp_min(1e-12)
    return (ratio - 1.0).clamp(-1.0, 3.0)


# baseline: head's shipped controller

def apply_steps(cuts, resid, P, min_w, t_rank, max_step=MAX_STEP,
                hard=True, tol=0.02, gain=0.5):
    """main.c's decision sweep. Left to right and asymmetric: the step reads the
    updated newcut[b-1], the upper clamp the pre-move cut[b+1]."""
    new = cuts.clone()
    applied = torch.zeros(cuts.shape[0], P - 1, device=cuts.device, dtype=cuts.dtype)
    for b in range(1, P):
        w_lo = new[:, b] - new[:, b - 1]          # updated left neighbour
        w_hi = cuts[:, b + 1] - new[:, b]         # not yet updated
        heur = heuristic_raw_step(t_rank[:, b - 1], t_rank[:, b], w_lo, w_hi,
                                  tol=tol, gain=gain, max_step=max_step)
        if resid is None:
            step = heur
        else:
            base = torch.atanh((heur / max_step).clamp(-0.995, 0.995))
            step = torch.tanh(base + resid[:, b - 1]) * max_step
            if hard:
                # half away from zero, like floor(x+0.5) in rnn_balancer.c.
                # torch.round is half to even and would disagree at exactly .5
                step = torch.where(step >= 0, torch.floor(step + 0.5),
                                   torch.ceil(step - 0.5))

        lo_limit = new[:, b - 1] + min_w
        hi_limit = cuts[:, b + 1] - min_w
        nc = torch.max(torch.min(new[:, b] - step, hi_limit), lo_limit)
        nc = torch.where(lo_limit <= hi_limit, nc, new[:, b])
        applied[:, b - 1] = new[:, b] - nc
        new[:, b] = nc
    return new, applied


def heuristic_step(t_rank, cuts, P, min_w, tol=0.02, gain=0.5, max_step=MAX_STEP):
    """head's shipped controller: apply_steps() with no residual."""
    return apply_steps(cuts, None, P, min_w, t_rank, max_step=max_step,
                       tol=tol, gain=gain)
