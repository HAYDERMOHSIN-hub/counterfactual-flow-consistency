"""expbase.py - shared setup for Stage-3 experiments: model loading, Space construction (with the
26-Oct UNSW immutability mask that includes 'service'), confirmatory-case selection, and a uniform
npz save. All experiments import from here so the detector, training donor pool, seed, feature order
and immutable set are IDENTICAL to the confirmatory run (fair comparison on the rebuilt models)."""
import os, sys, time, pickle, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cfg
from common import load, SEED
from generator import Space, genetic_search, nearest_real

EXP = os.path.join(cfg.ROOT, 'experiments')
os.makedirs(EXP, exist_ok=True)

# 26-Oct decision: UNSW immutability mask (D5) = proto, is_sm_ips_ports, service.
UNSW_IMMUTABLE = ('proto', 'is_sm_ips_ports', 'service')


def build_space(ds, extra_immutable=()):
    """Return (sp, feats, meta). extra_immutable adds feature names to the immutable mask
    (used by the TTL-fixed experiment: extra_immutable=('sttl','dttl'))."""
    Xtr, ytr, ctr, feats, meta = load(ds, 'train')
    sp = Space(ds, Xtr, ytr, feats, meta, SEED)
    del Xtr, ytr
    imm = list(UNSW_IMMUTABLE) if ds == 'unsw' else []
    imm += [f for f in extra_immutable if f in feats]
    if imm:
        sp.immutable = np.array([f in imm for f in feats])
    return sp, feats, meta


def load_model(ds, det):
    model = pickle.load(open(os.path.join(cfg.DATA, f'{ds}_{det}.pkl'), 'rb'))
    try:
        model.set_params(n_jobs=1)
    except Exception:
        pass
    return model


def conf_cases(ds, feats):
    """Confirmatory test cases (seed-20261008 draw) at the saved indices; returns X0, c0, idx.

    Requires data/{ds}_conf_idx_20261008.npy (the CONFIRMATORY draw). This is distinct from
    data/{ds}_orig_idx.npy, which the training script writes for the EXPLORATORY/pilot draw and must
    NOT be substituted here. If the confirmatory index is absent it can be reconstructed from the
    deposited arrays with code/extract_conf_idx.py; we fail loudly rather than fall back to the pilot
    draw or proceed silently."""
    conf_path = os.path.join(cfg.DATA, f'{ds}_conf_idx_20261008.npy')
    if not os.path.exists(conf_path):
        raise SystemExit(
            f'ERROR: confirmatory index not found: {conf_path}\n'
            f'  Do NOT substitute data/{ds}_orig_idx.npy (that is the exploratory/pilot draw).\n'
            f'  Reconstruct it from the deposited arrays:\n'
            f'      python code/extract_conf_idx.py\n')
    idx = np.load(conf_path)
    Xte, yte, cte, _, _ = load(ds, 'test')
    if idx.size == 0 or idx.min() < 0 or idx.max() >= len(Xte):
        raise SystemExit(
            f'ERROR: confirmatory index out of bounds for {ds} test split '
            f'(n_test={len(Xte)}, idx range [{int(idx.min())}, {int(idx.max())}]).')
    X0, c0 = Xte[idx], cte[idx]
    del Xte
    return X0, c0, idx


def immutable_check(sp, cf, X0, feats):
    """Per-candidate immutability audit: immutable features must equal the original. Returns
    (n_mismatch, per_feat_dict)."""
    imm_idx = np.flatnonzero(sp.immutable)
    mism = np.zeros(len(cf), dtype=bool); per = {}
    for j in imm_idx:
        mm = cf[:, j] != X0[:, j]
        mism |= mm; per[feats[j]] = int(mm.sum())
    return int(mism.sum()), per


def save_arm(path, sp, feats, arm, seed, X0, c0, idx, r, extra=None):
    cf = r['cf']; pb = r['p_benign']; flipped = r['returned'] & (pb > 0.5)
    viol, app = sp.audit(cf)
    n_mis, per = immutable_check(sp, cf, X0, feats)
    d = dict(cf=cf, x0=X0, cls=c0, test_idx=idx, returned=r['returned'], flipped=flipped,
             p_benign=pb, viol=viol, app=app, oor=sp.out_of_range(cf),
             d_pct=sp.dist_pct(sp.pct(X0), sp.pct(cf)), d_gower=sp.dist_gower(X0, cf), l0=sp.l0(X0, cf),
             constraint_ids=np.array([c.id for c in sp.constraints]),
             constraint_cls=np.array([c.cls for c in sp.constraints]),
             train_row=r.get('train_row', np.array([])), pool_size=r.get('pool_size', 0),
             queries=r.get('queries', 0), arm=arm, seed=seed,
             immutable_feats=np.array([f for f, b in zip(feats, sp.immutable) if b]),
             immutable_mismatch_count=n_mis,
             immutable_mismatch_per_feat=np.array(json.dumps(per)),
             elig_mask=sp.elig_mask)
    if extra:
        d.update(extra)
    np.savez_compressed(path, **d)
    return dict(flip=float(flipped.mean()), n_mis=n_mis, queries=int(r.get('queries', 0)))


def run_arm(sp, model, X0, arm, seed):
    return nearest_real(sp, model, X0) if arm == 'C' else genetic_search(sp, model, X0, arm, seed)
