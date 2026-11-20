"""Verify saved scalar receipts and recompute medians; Python stdlib only."""
import csv
import math
from pathlib import Path
from statistics import median as med

DATA = Path(__file__).resolve().parent / 'data'


def read(name):
    with (DATA / (name + '.csv')).open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    old = read('prefill_only_pairs')
    assert len(old) == 9 and all(r['same_tokens'] == '1' and r['reference_rot_calls'] == '0'
        and r['fused_rot_calls'] == '5' and r['decode_rot_calls'] == '0' for r in old)
    print('Earlier prefill-only paired speedup (n=9):',
          round(med(float(r['reference_prefill_s']) / float(r['fused_prefill_s']) for r in old), 3))
    rows = read('integration_2x2')
    modes = ('reference', 'prefill_only', 'decode_only', 'both')
    durations = {}
    expected_shapes = {44:8, 128:8, 160:8, 192:8, 512:4, 960:4, 4032:4}
    for r in rows:
        n, k, j = (int(r[col]) for col in ('prompt_tokens','cached_steps','round'))
        mode = r['mode']
        assert n in expected_shapes and 0 <= j < expected_shapes[n] and mode in modes
        assert k == (16 if n <= 192 else 63) and r['same_tokens'] == '1'
        fused = n >= 128 and n % 32 == 0 and mode in ('prefill_only','both')
        metal = mode in ('decode_only','both')
        assert [int(r[k]) for k in ('prefill_rot','prefill_gate','prefill_down',
                'decode_rot','decode_gate','decode_down')] == [5*int(fused),0,0,0,5*k*metal,5*k*metal]
        pre, first, dec, total = (float(r[k]) for k in ('prefill_s','ttft_s','decode_s','total_s'))
        assert 0 < pre <= first < total and dec > 0 and math.isclose(first+dec,total)
        key = (n,j,mode)
        assert key not in durations
        durations[key] = total
    assert len(durations) == 4*sum(expected_shapes.values())
    print('2x2: prompt | reference / prefill only / decode only / both (median ms) | paired speedup [min,max]')
    for n, rounds in expected_shapes.items():
        latency = [med(durations[n,j,m] for j in range(rounds))*1000 for m in modes]
        ratio = [durations[n,j,'reference']/durations[n,j,'both'] for j in range(rounds)]
        print(f'{n:4d}  ' + ' / '.join(f'{t:.1f}' for t in latency)
              + f'   {med(ratio):.3f}x [{min(ratio):.3f},{max(ratio):.3f}], n={rounds}')
    parity = read('integration_parity')
    assert {(int(r['prompt_tokens']),r['mode']) for r in parity} == {
        (n,m) for n in expected_shapes for m in modes[1:]} and len(parity) == 21
    assert all(r['same_tokens']=='1' and all(0 <= float(r[c]) < .001 for c in
               ('max_abs_logit','max_abs_prefill_cache','max_abs_final_cache')) for r in parity)
    controls = read('decode_controls')
    for kind, mm, shapes, rounds in [('fixed', ('baseline','packed','gather','metal'), (48,128,192),8),
            ('application', ('baseline','gather','metal'), (44,48,56),6)]:
        rr = [r for r in controls if r['measurement'] == kind]
        assert len(rr) == len(mm)*len(shapes)*rounds
        rates = {(r['mode'],int(r['prompt_tokens']),int(r['round'])):
                 int(r['counted_tokens'])/float(r['seconds']) for r in rr}
        assert all(r['same_tokens']=='1' for r in rr)
        for m in mm:
            vals = [rates[m,n,j] for n in shapes for j in range(rounds)]
            print(f'{kind:11s} {m:10s} {med(vals):7.2f} tok/s (n={len(vals)})')
    kernels = read('dispatch')
    assert len(kernels) == 50
    print('Standalone dispatch medians (ms):',
          *(round(med(float(r[c]) for r in kernels),4) for c in ('gate_up_swiglu_ms','down_mix_ms')))
    synthetic = read('synthetic_m3max')
    for m in ('stack_dense','packed_dense','gather','metal'):
        sr = [r for r in synthetic if r['mode'] == m]
        assert len(sr) == 48 and {int(r['route_case']) for r in sr} == set(range(12))
        print(f'Synthetic {m:13s} {med(float(r["latency_ms"]) for r in sr):.4f} ms (n=48)')


if __name__ == '__main__':
    main()
