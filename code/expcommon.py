"""expcommon.py - shared, manuscript-consistent metric/state definitions for the Stage-3 experiments.

The five-state model and M1/M2/M3 are taken VERBATIM (same formulas) from the confirmatory
analysis script analyze_confirmatory_corrected.py (protocol s6/s7 + 26 Oct decisions), so every
Stage-3 experiment reports metrics on exactly the same eligible-constraint basis as the manuscript:
  - eligible set  : register-eligible conditions (44 CIC / 73 UNSW) from register_extract.json
  - M1 (coverage) : (s3+s4+s5)/N  = converted (benign-side, candidate exists)
  - M2 (primary)  : s4/(s3+s4+s5) = eligible class-2/3 violation rate among converted
  - M3 (joint)    : s5/N          = converted AND no eligible violation
States: s1 no candidate, s2 candidate but not converted, s3 conv + class-1 eligible viol only,
        s4 conv + class-2/3 eligible viol, s5 conv + no eligible viol.
Nothing here is tuned; it only re-expresses the saved-array audit in the frozen vocabulary.
"""
import os, json, numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REG = json.load(open(os.path.join(_HERE, 'register_extract.json'), encoding='utf-8'))


def elig_map(ds):
    """code -> class-digit ('1'/'2'/'3') for register-eligible conditions of this dataset."""
    sheet = 'CIC-IDS2017' if ds == 'cic' else 'UNSW-NB15'
    return {r['code']: r['cls'][:1] for r in _REG[sheet]
            if r['elig_e'] and r['elig_e'].startswith('Eligible')}


def elig_sets(ds):
    el = elig_map(ds)
    cls2 = set(c for c, k in el.items() if k in ('2', '3'))
    return el, cls2


def states_for(z, elig, cls2_cols, elig_cols, is_C):
    """Five states + M1/M2/M3 from one saved npz-like mapping z (must expose p_benign, returned,
    viol, and, for arm C, train_row). Identical arithmetic to the confirmatory analysis."""
    pb = z['p_benign']; ret = z['returned']; viol = z['viol']; N = len(pb)
    tr = z['train_row'] if ('train_row' in getattr(z, 'files', []) or 'train_row' in z) else np.zeros(0)
    exists = ret if is_C else np.ones(N, bool)
    conv = (pb > 0.5) & exists
    v23 = viol[:, cls2_cols].any(1) if len(cls2_cols) else np.zeros(N, bool)
    vel = viol[:, elig_cols].any(1)
    v1only = vel & ~v23
    s1 = int((~exists).sum()); s2 = int((exists & ~conv).sum())
    s4 = int((conv & v23).sum()); s3 = int((conv & v1only).sum()); s5 = int((conv & ~vel).sum())
    denom = s3 + s4 + s5
    return dict(N=N, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, conv=conv, n_conv=int(conv.sum()),
                M1=denom / N, M2=(s4 / denom if denom else None), M3=s5 / N)


def cols_for(z, ds):
    """Map eligible / class-2|3 constraint codes to column indices of this array's viol matrix."""
    el, cls2 = elig_sets(ds)
    ids = list(z['constraint_ids']); col = {c: i for i, c in enumerate(ids)}
    elig_cols = [col[c] for c in el if c in col]
    cls2_cols = [col[c] for c in cls2 if c in col]
    return el, cls2, elig_cols, cls2_cols


def metrics_from_array(z, ds, is_C):
    el, cls2, elig_cols, cls2_cols = cols_for(z, ds)
    return states_for(z, el, cls2_cols, elig_cols, is_C)


def wilson(k, n, zc=1.96):
    if n == 0:
        return (None, None)
    p = k / n; d = 1 + zc * zc / n; c = p + zc * zc / (2 * n)
    h = zc * np.sqrt(p * (1 - p) / n + zc * zc / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
