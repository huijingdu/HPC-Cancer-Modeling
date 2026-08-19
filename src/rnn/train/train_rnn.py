#!/usr/bin/env python3
"""Train the X-cut load balancing controller and write the weights file.

Usage: python3 train/train_rnn.py --steps 6000 --features base
"""

import argparse
import os
import struct
import time

import torch
import torch.nn as nn

import sim_env
from sim_env import (Scenario, cut_features, heuristic_step, heuristic_proposal,
                     apply_steps,
                     N_FEATURES_BASE, N_FEATURES_FULL, MAX_STEP, STEPS_PER_GROWTH)


HIDDEN = 24


class CutController(nn.Module):
    """Shared-weight GRU controller, one hidden state per interior cut. The output
    is a residual on main.c's proportional step, in pre-tanh space."""

    def __init__(self, n_features=N_FEATURES_BASE, hidden=HIDDEN, max_step=MAX_STEP):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.max_step = max_step
        self.cell = nn.GRUCell(n_features, hidden)
        self.head = nn.Linear(hidden, 1)
        # zero the weight too, not just the bias, so the residual starts at zero
        # and the controller is the shipped rule until it learns something
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.head.weight)

    def residual(self, feats, h):
        """feats (B,C,F), h (B*C,H) -> resid (B,C), h_next (B*C,H). Runs on the
    pre-move decomposition; the step is assembled in apply_steps()."""
        B, C, F = feats.shape
        h = self.cell(feats.reshape(B * C, F), h)
        return self.head(h).view(B, C), h

    def forward(self, feats, h, heur):
        """Residual combined with a given heuristic step. Used by the parity export,
    which checks both halves together."""
        resid, h = self.residual(feats, h)
        # the coordinate where the residual is additive. 0.995 keeps atanh finite
        # when the heuristic step is saturated at +-MAX_STEP
        base = torch.atanh((heur / self.max_step).clamp(-0.995, 0.995))
        step = torch.tanh(base + resid) * self.max_step
        return step, h

    def init_hidden(self, B, C, device, dtype=torch.float32):
        return torch.zeros(B * C, self.hidden, device=device, dtype=dtype)


# rollout

def rollout(model, sc, T, tau, gen, hard=False, use_heuristic=False,
            no_control=False, collect=None, n_features=None):
    """Run one episode. Returns a dict of metrics."""
    B, P, nx, mw = sc.B, sc.P, sc.nx, sc.min_w
    C = P - 1
    dev, dt = sc.device, sc.dtype
    nf = n_features if n_features is not None else (
        model.n_features if model is not None else N_FEATURES_BASE)

    cuts = sc.initial_cuts()

    # difficulty scale per instance, detached so the policy cannot game it
    t0 = sc.rank_times(cuts, tau, gen=None, hard=hard)
    m0 = t0.mean(1)
    scale = (((t0.max(1).values - m0) / m0.clamp_min(1e-12))
             .detach().clamp_min(0.05))

    h = model.init_hidden(B, C, dev, dt) if model is not None else None
    prev_imb = torch.zeros(B, C, device=dev, dtype=dt)
    prev_step = torch.zeros(B, C, device=dev, dtype=dt)

    imb_sum = torch.zeros(B, device=dev, dtype=dt)
    spread_sum = torch.zeros(B, device=dev, dtype=dt)
    moved_sum = torch.zeros(B, device=dev, dtype=dt)

    for t in range(T):
        burst = sc.growth_fired_last()
        # what the ranks actually measured, jitter included
        t_obs = sc.rank_times(cuts, tau, gen=gen, hard=hard, burst=burst)
        # what it really costs. this is what the loss sees
        t_true = sc.rank_times(cuts, tau, gen=None, hard=hard, burst=burst)

        mean = t_true.mean(1)
        imb_sum = imb_sum + (t_true.max(1).values - mean) / mean.clamp_min(1e-12)
        spread_sum = spread_sum + t_true.std(1) / mean.clamp_min(1e-12)

        if no_control:
            new_cuts = cuts
        elif use_heuristic:
            new_cuts, _ = heuristic_step(t_obs, cuts, P, mw, max_step=MAX_STEP)
        else:
            work = sc.observed_work() if nf == N_FEATURES_FULL else None
            target = sc.legalize(sc.quantile_cuts(work)) if nf == N_FEATURES_FULL else None
            feats, imb = cut_features(t_obs, cuts, prev_imb, prev_step,
                                      nx, P, mw, MAX_STEP,
                                      phase=sc.phase(),
                                      burst_fired=1.0 if burst else 0.0,
                                      work=work, target_cuts=target,
                                      n_features=nf)
            if collect is not None:
                collect.append((feats.detach().clone(), h.detach().clone(),
                                heuristic_proposal(t_obs, cuts, P,
                                                   MAX_STEP).detach().clone()))
            resid, h = model.residual(feats, h)
            # cut b moves down when the band below is slower, as in main.c:2392.
            # the step is built inside the sweep, where the partial widths exist
            new_cuts, applied = apply_steps(cuts, resid, P, mw, t_obs,
                                            max_step=MAX_STEP, hard=hard)
            prev_imb = imb.detach() if hard else imb
            prev_step = applied

        moved_sum = moved_sum + (cuts[:, 1:P] - new_cuts[:, 1:P]).abs().sum(1)
        cuts = new_cuts
        sc.advance(gen)

    return {
        'imb': (imb_sum / T),                    # raw, for reporting
        'spread': (spread_sum / T),
        'moved': moved_sum,
        'imb_n': (imb_sum / T) / scale,          # normalized, for the loss
        'spread_n': (spread_sum / T) / scale,
        'scale': scale,
    }


# weight export

def export_weights(model, out_dir, meta):
    """Write rnn_weights.txt, which the solver reads at startup. nn.GRUCell
    layout: *_IH/*_HH rows ordered [reset, update, new]."""
    sd = model.state_dict()
    w_ih = sd['cell.weight_ih'].detach().cpu().float()
    w_hh = sd['cell.weight_hh'].detach().cpu().float()
    b_ih = sd['cell.bias_ih'].detach().cpu().float()
    b_hh = sd['cell.bias_hh'].detach().cpu().float()
    w_o = sd['head.weight'].detach().cpu().float().view(-1)
    b_o = sd['head.bias'].detach().cpu().float().view(-1)

    H, F = model.hidden, w_ih.shape[1]
    assert w_ih.shape == (3 * H, F) and w_hh.shape == (3 * H, H)

    def block(name, t):
        v = t.contiguous().view(-1).tolist()
        lines = [name]
        for i in range(0, len(v), 6):
            lines.append('  ' + ' '.join(f'{x:.9e}' for x in v[i:i + 6]))
        return '\n'.join(lines)

    txt = f"""# rebal RNN controller weights, read by rebal_rnn_load_weights()
# generated by train_rnn.py, do not edit by hand
# trained {meta['when']}, {meta['steps']} steps, feature set "{meta['features']}"
# {meta['horizon']}-event validation, imbalance: static {meta['val_static']:.2f}% -> \
heuristic {meta['val_heur']:.2f}% -> RNN {meta['val_rnn']:.2f}%
# slices migrated: heuristic {meta['mv_heur']:.1f} -> RNN {meta['mv_rnn']:.1f}

features {F}
hidden {H}
max_step {model.max_step}
steps_per_growth {STEPS_PER_GROWTH}

{block('w_ih', w_ih)}

{block('w_hh', w_hh)}

{block('b_ih', b_ih)}

{block('b_hh', b_hh)}

{block('w_out', w_o)}

b_out
  {b_o.item():.9e}
"""
    path = os.path.join(out_dir, 'rnn_weights.txt')
    with open(path, 'w') as f:
        f.write(txt)
    return path


def export_parity_vectors(model, out_dir, samples):
    """Dump (x, h_in, heur) -> (h_out, step_raw) so the C can be checked. Samples
    come from real rollouts, so the test sees the inputs the controller will."""
    xs = torch.cat([s[0].reshape(-1, s[0].shape[-1]) for s in samples], 0)
    hs = torch.cat([s[1] for s in samples], 0)
    us = torch.cat([s[2].reshape(-1) for s in samples], 0)
    n = min(512, xs.shape[0])
    idx = torch.randperm(xs.shape[0])[:n]
    x, h, u = xs[idx].float(), hs[idx].float(), us[idx].float()
    with torch.no_grad():
        h_out = model.cell(x, h)
        resid = model.head(h_out).view(-1)
        base = torch.atanh((u / model.max_step).clamp(-0.995, 0.995))
        step = torch.tanh(base + resid) * model.max_step
    path = os.path.join(out_dir, 'parity_vectors.bin')
    with open(path, 'wb') as f:
        f.write(b'XBPAR\x00\x00\x01')
        f.write(struct.pack('<iii', n, x.shape[1], model.hidden))
        for t in (x, h, u, h_out, step):
            f.write(t.contiguous().numpy().astype('<f4').tobytes())
    return path, n


# main

def sample_shape(gen):
    """Randomized problem geometry. Feasibility (nx >= P*min_w) is checked here,
    not at runtime, because main.c:490 aborts below it."""
    P = [2, 4, 8, 16][torch.randint(0, 4, (1,), generator=gen).item()]
    min_w = int(torch.randint(4, 33, (1,), generator=gen).item())
    lo = P * min_w + 8
    hi = max(lo + 32, 513)
    nx = int(torch.randint(max(64, lo), hi, (1,), generator=gen).item())
    return P, nx, min_w


def evaluate(model, nf, horizon, trials=24, batch=24, seed=99991, collect=None):
    """Hard/integer rollout of all three policies on identical scenarios."""
    dev = torch.device('cpu')
    egen = torch.Generator(device=dev); egen.manual_seed(seed)
    res = {'static': [], 'heur': [], 'rnn': []}
    mv = {'heur': [], 'rnn': []}
    with torch.no_grad():
        for trial in range(trials):
            P, nx, min_w = sample_shape(egen)
            base = egen.get_state()
            for tag, kw in (('static', dict(no_control=True)),
                            ('heur', dict(use_heuristic=True)),
                            ('rnn', dict())):
                egen.set_state(base)
                sc = Scenario(batch, P, nx, min_w, dev, egen)
                g2 = torch.Generator(device=dev); g2.manual_seed(5000 + trial)
                m = rollout(model, sc, horizon, 0.4, g2, hard=True, n_features=nf,
                            collect=(collect if tag == 'rnn' else None), **kw)
                res[tag].append(m['imb'].mean().item())
                if tag in mv:
                    mv[tag].append(m['moved'].mean().item())
    val = {k: 100.0 * sum(v) / len(v) for k, v in res.items()}
    mvv = {k: sum(v) / len(v) for k, v in mv.items()}
    return val, mvv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=32)
    # 50 events is a 5000-iteration run. eval uses 100, so we measure how it
    # holds up past the horizon it trained on
    ap.add_argument('--horizon', type=int, default=50)
    ap.add_argument('--eval-horizon', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-3)
    # charge much more than this and "never move" wins. chasing noise is already
    # punished by the imbalance term, which is measured on the clean times
    ap.add_argument('--mig-w', type=float, default=0.0005)
    ap.add_argument('--spread-w', type=float, default=0.3)
    ap.add_argument('--features', choices=['base', 'full'], default='base',
                    help='base = no new MPI collective; full = + work-histogram features')
    ap.add_argument('--seed', type=int, default=1234)
    # the solver reads rnn/rnn_weights.txt, and this file sits in rnn/train/
    ap.add_argument('--out', default=os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument('--tag', default=None, help='suffix for controller checkpoint')
    ap.add_argument('--no-export', action='store_true',
                    help='train and evaluate only; do not overwrite rnn_weights.txt')
    ap.add_argument('--from-ckpt', default=None,
                    help='evaluate and export this checkpoint instead of training')
    args = ap.parse_args()

    nf = N_FEATURES_BASE if args.features == 'base' else N_FEATURES_FULL

    dev = torch.device('cpu')
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev); gen.manual_seed(args.seed)

    model = CutController(n_features=nf).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'controller: {n_params} parameters (GRU {nf}->{HIDDEN}, head {HIDDEN}->1, '
          f'max_step {MAX_STEP}, feature set "{args.features}")')
    print(f'cadence: REBALANCE_FR={sim_env.REBALANCE_FR}, '
          f'growth every {STEPS_PER_GROWTH} events')

    if args.from_ckpt:
        ck = torch.load(args.from_ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(ck['model'])
        trained_steps = ck.get('meta', {}).get('steps', 0)
        n_train = 0
        print(f'loaded {args.from_ckpt}: re-evaluating, no training\n')
    else:
        trained_steps = n_train = args.steps
        print(f'training: {args.steps} steps, batch {args.batch}, '
              f'horizon {args.horizon}\n')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, n_train), eta_min=args.lr * 0.05)

    t0 = time.time()
    for it in range(n_train):
        # anneal toward the hard integer assignment the deployed code uses, so
        # training finishes on the real objective
        frac = it / max(1, n_train - 1)
        tau = 1.5 * (1 - frac) + 0.4 * frac

        P, nx, min_w = sample_shape(gen)
        sc = Scenario(args.batch, P, nx, min_w, dev, gen)
        m = rollout(model, sc, args.horizon, tau, gen, n_features=nf)

        # balance is normalized, migration is not, so a slice costs the same everywhere
        loss = (m['imb_n'] + args.spread_w * m['spread_n']
                + args.mig_w * m['moved']).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if it % 250 == 0 or it == n_train - 1:
            print(f'  step {it:5d}  loss {loss.item():7.4f}  '
                  f'kept {m["imb_n"].mean().item():5.2f}x  '
                  f'imb {m["imb"].mean().item()*100:6.2f}%  '
                  f'moved {m["moved"].mean().item():6.1f}  '
                  f'(P={P:2d} nx={nx:3d} mw={min_w:2d} tau={tau:.2f})  '
                  f'{time.time()-t0:5.0f}s')

    print(f'\nevaluating: hard integer cuts, unseen scenarios, '
          f'{args.eval_horizon} events (= a {args.eval_horizon*sim_env.REBALANCE_FR}'
          f'-iteration run)...')
    collect = []
    model.eval()
    val, mvv = evaluate(model, nf, args.eval_horizon, collect=collect)
    print(f'  mean imbalance (max-mean)/mean:')
    print(f'    no rebalancing : {val["static"]:6.2f}%')
    print(f'    heuristic      : {val["heur"]:6.2f}%   ({mvv["heur"]:.1f} slices moved)')
    print(f'    RNN            : {val["rnn"]:6.2f}%   ({mvv["rnn"]:.1f} slices moved)')
    rel = (val['heur'] - val['rnn']) / val['heur'] * 100.0 if val['heur'] > 0 else 0.0
    print(f'    RNN vs heuristic: {rel:+.1f}% relative')

    # long-horizon check: something with memory can drift or oscillate well past
    # the horizon it saw in training. this is IC_37K's real geometry
    with torch.no_grad():
        sgen = torch.Generator(device=dev); sgen.manual_seed(31337)
        sc = Scenario(16, 4, 256, 21, dev, sgen)
        long_m = rollout(model, sc, 400, 0.4, sgen, hard=True, n_features=nf)
    print(f'  400-event stability (P=4, nx=256, min_w=21, IC_37K geometry): '
          f'imbalance {long_m["imb"].mean().item()*100:.2f}%, '
          f'{long_m["moved"].mean().item():.0f} slices moved')

    tag = args.tag or args.features
    torch.save({'model': model.state_dict(),
                'meta': dict(steps=trained_steps, features=args.features,
                             n_features=nf, hidden=HIDDEN, max_step=MAX_STEP)},
               os.path.join(args.out, f'controller_{tag}.pt'))

    if args.no_export:
        print(f'\n--no-export: checkpoint saved as controller_{tag}.pt, '
              f'rnn_weights.txt untouched')
        return

    meta = dict(when=time.strftime('%Y-%m-%d %H:%M'), steps=trained_steps,
                features=args.features, horizon=args.eval_horizon,
                val_static=val['static'], val_heur=val['heur'], val_rnn=val['rnn'],
                mv_heur=mvv['heur'], mv_rnn=mvv['rnn'])
    wpath = export_weights(model, args.out, meta)
    pv, npv = export_parity_vectors(model, args.out, collect)
    print(f'\nexported:\n  {wpath}\n  {pv} ({npv} vectors)')


if __name__ == '__main__':
    main()
