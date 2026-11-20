"""Weight-free comparison of the selected-expert shaders and two MLX controls."""
import argparse
import csv
from importlib.metadata import version
import itertools
from statistics import median
from time import perf_counter

import mlx.core as mx
import mlx.nn as nn

from kernels import D, E, F, pack, selected_decode, top_two


def projection_inputs(seed):
    mx.random.seed(seed)
    return tuple(tuple(mx.random.normal(s) * (s[-1] ** -0.5) for _ in range(E))
                 for s in ((F, D), (F, D), (D, F)))


def controls(x, weights, logits, mode):
    ix, alpha = top_two(logits)
    a, b, c = (mx.swapaxes(w, -1, -2) for w in weights)
    v = x.reshape(1, 1, D)
    if mode == 'gather':
        def pick(m):
            return mx.gather_mm(v, m, rhs_indices=ix)
        h = nn.silu(pick(a)) * pick(b)
        o = mx.gather_mm(h, c, lhs_indices=mx.arange(2, dtype=mx.uint32), rhs_indices=ix)
        return mx.sum(o * alpha.reshape(2, 1, 1), axis=0).reshape(D)
    gates = mx.sum((ix[:, None] == mx.arange(E)[None, :]) * alpha[:, None], axis=0)
    h = nn.silu(v @ a) * (v @ b)
    return mx.sum((h @ c) * gates.reshape(E, 1, 1), axis=0).reshape(D)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rounds', type=int, default=24)
    parser.add_argument('--seed', type=int, default=800)
    parser.add_argument('--csv', help='optional output path, created exclusively')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    if not mx.metal.is_available():
        raise SystemExit('Native Apple Silicon Metal GPU required')
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 << 20)
    groups = projection_inputs(args.seed)
    mx.eval(groups)
    t = perf_counter()
    packed = pack(groups)
    mx.eval(packed)
    print(f'MLX {version("mlx")}; seed {args.seed}; pack {(perf_counter()-t)*1000:.3f} ms; '
          f'{sum(w.nbytes for w in packed)/2**20:.0f} MiB for one layer')
    x = mx.random.normal((D,))
    # All ordered top-two choices, unequal mixing, and nontrivial signed activations.
    routes = [mx.array([3.0 if i == a else 1.5 if i == b else -0.5 for i in range(E)])
              for a, b in itertools.permutations(range(E), 2)]
    mx.eval(x, routes)
    modes = ('stack_dense', 'packed_dense', 'gather', 'metal')
    def run(mode, scores):
        if mode == 'metal':
            return selected_decode(x, packed, scores)
        w = pack(groups) if mode == 'stack_dense' else packed
        return controls(x, w, scores, mode)
    # Reuse only inputs and weights, never a computed output/lazy graph.
    worst = 0.0
    for scores in routes:
        ref = run('stack_dense', scores)
        for mode in modes[1:]:
            err = mx.max(mx.abs(run(mode, scores) - ref)).item()
            if not err < 1e-4:  # also rejects NaN
                raise AssertionError((mode, err))
            worst = max(worst, err)
    print(f'12 ordered pairs checked: maximum absolute output error {worst:.8g}')
    rows = []
    order = ((0,1,3,2), (1,2,0,3), (2,3,1,0), (3,0,2,1))
    for r in range(args.rounds):
        scores = routes[r % len(routes)]
        for j in order[r % 4]:
            name = modes[j]
            t = perf_counter()
            out = run(name, scores)
            mx.eval(out)
            rows.append((r, r % len(routes), name, (perf_counter()-t)*1000))
    if args.csv:
        with open(args.csv, 'x', newline='') as f:
            w = csv.writer(f, lineterminator='\n')
            w.writerow(('round', 'route_case', 'mode', 'latency_ms'))
            w.writerows(rows)
    for mode in modes:
        times = [t for _, _, m, t in rows if m == mode]
        print(f'{mode:13s} {median(times):.4f} ms median (n={len(times)})')


if __name__ == '__main__':
    main()
