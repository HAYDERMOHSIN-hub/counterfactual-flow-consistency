"""generator.py - Genetic counterfactual search (arms A, A-typed, B), nearest-real-flow baseline (arm C),
percentile-shift distance, and the repair operators written from the FROZEN constraint lists.

Arm B = arm A + `repair` + feasibility-first ranking. Nothing else differs (same seeds, same budget).

CORRECTED 2026-11-02 (supervisor report 31 Oct 2026, sections 3 & 6; protocol v0.4 Section 6.2 / D12):
arm B now enforces ONLY the proposed-eligible frozen constraints (44 CIC / 73 UNSW: T1 AND numeric
screen), not the full retained numeric-screen list (187 CIC / 109 UNSW). Two coupled changes, both
confined to arm B; arms A, A_typed and C are byte-for-byte unchanged (they never call repair or the
mode-'B' audit branch):
  (1) repair_cic / repair_unsw project onto the eligible relations/identities only
      (CIC-135/136/137 PktLen ordering, CIC-161/165 avgseg=mean; UNSW-090 tcprtt, UNSW-091 smeansz,
      UNSW-092 dmeansz), on top of the shared type-awareness (clip + integer round + non-negativity,
      identical to arm A_typed). Every ineligible projection (T2 integer/flag/subflow/ratio/rate/
      conditional-zero, T3 header-length/n-1-identity/ct_*-bounds/sloss/tcprtt<=dur, T4 mean-from-count
      and Sload/Dload) is removed.
  (2) feasibility-first ranking checks violations over the eligible subset only (Space.elig_mask).
On candidate return (arm B): genetic_search always STORES the best-ranked candidate in `cf` (even an
infeasible one). The `returned` flag is set only when that best candidate is valid (p_benign>0.5) and
feasible (passes the eligible in-loop audit and is in range): `ok = valid[ar,best] & feas[ar,best]`.
An instance whose best candidate is infeasible is a generation failure (returned=False) and is excluded
from the M2 denominator; the candidate itself is still stored and flagged, so the audit does not withhold
it from storage. Thus M2(B)=0 holds by construction of the feasibility-first (Deb, 2000) ranking whenever
a feasible candidate exists for every instance. In every confirmatory run here all 1,000 arm-B candidates
were feasible, so `returned` and the analysis's `converted` sets coincide (1,000 each) and nothing was
withheld; this is declared as an open point, not presented as an empirical finding."""
import json, numpy as np
import constraints as K

POP, GENS, ELITE = 60, 40, 30
P_DONOR, P_GAUSS, P_RESET = 0.4, 0.4, 0.2
SPARSITY_EPS = 1e-3


class Space:
    """Training-derived quantities shared by all arms (base ranges, ECDF reference, donors)."""

    def __init__(self, ds, Xtr, ytr, feats, meta, seed):
        rng = np.random.default_rng(seed)
        self.ds, self.feats, self.meta = ds, feats, meta
        self.p = len(feats)
        self.ix = {f: j for j, f in enumerate(feats)}
        self.cat = np.array([f in meta['categorical'] for f in feats])
        imm = ['proto', 'is_sm_ips_ports'] if ds == 'unsw' else []
        self.immutable = np.array([f in imm for f in feats])
        self.lo, self.hi = Xtr.min(0), Xtr.max(0)
        ref = Xtr[rng.choice(len(Xtr), 200000, replace=False)]
        self.ref_sorted = np.sort(ref, axis=0)
        q05, q95 = np.percentile(Xtr[::7], [5, 95], axis=0)
        sd = Xtr[::7].std(0)
        self.sigma = 0.1 * np.where(q95 - q05 > 0, q95 - q05, sd)
        self.range = np.where(self.hi - self.lo > 0, self.hi - self.lo, 1.0)
        ben = np.flatnonzero(ytr == 0)
        self.ben_pool_idx = np.sort(rng.choice(ben, 200000, replace=False))   # donors and arm-C pool
        self.donors = Xtr[self.ben_pool_idx]
        ints = K.CIC_INT if ds == 'cic' else K.UNSW_INT
        self.int_mask = np.array([f in ints for f in feats])
        frozen = set(json.load(open(f'protocol/{ds}_frozen_constraints.json'))['frozen'])
        self.constraints = [c for c in K.candidates(ds, feats) if c.id in frozen]
        self.nonneg_mask = np.array([any(c.id.endswith(f'-D-nonneg[{f}]') for c in self.constraints) for f in feats])
        # CORRECTED 2026-11-02: proposed-eligible subset (T1 AND numeric screen) that arm B enforces.
        self.elig_ids = set(json.load(open(f'protocol/{ds}_eligible_constraints.json'))['eligible'])
        self.elig_mask = np.array([c.id in self.elig_ids for c in self.constraints])

    # ---- distance -------------------------------------------------------------------------
    def pct(self, X):
        out = np.empty(X.shape, dtype=np.float64)
        n = self.ref_sorted.shape[0]
        for j in range(self.p):
            if self.cat[j]:
                out[..., j] = X[..., j]
            else:
                r = self.ref_sorted[:, j]
                out[..., j] = (np.searchsorted(r, X[..., j], 'left') + np.searchsorted(r, X[..., j], 'right')) / (2.0 * n)
        return out

    def dist_pct(self, P0, P1):
        """Primary distance between percentile-space arrays (categorical columns hold raw codes)."""
        d = np.abs(P1 - P0)
        d[..., self.cat] = (d[..., self.cat] > 0)
        return d.mean(-1)

    def dist_gower(self, X0, X1):
        d = np.abs(X1 - X0) / self.range
        d[..., self.cat] = (X1[..., self.cat] != X0[..., self.cat])
        return d.mean(-1)

    def l0(self, X0, X1):
        return (np.abs(X1 - X0) > 1e-12).sum(-1)

    # ---- audit ----------------------------------------------------------------------------
    def audit(self, X):
        viol, app = K.evaluate(self.constraints, X, self.feats, self.meta)
        return viol, app

    def out_of_range(self, X):
        return ((X < self.lo - 1e-9) | (X > self.hi + 1e-9)).any(-1)

    # ---- repairs --------------------------------------------------------------------------
    def type_repair(self, X):
        X[..., self.int_mask] = np.round(X[..., self.int_mask])
        nn = self.nonneg_mask
        X[..., nn] = np.maximum(X[..., nn], 0.0)
        return X

    def repair(self, X, X0):
        return repair_cic(self, X) if self.ds == 'cic' else repair_unsw(self, X, X0)


def _cols(sp, X):
    class V:
        def __getitem__(s, f): return X[..., sp.ix[f]]
        def __setitem__(s, f, v): X[..., sp.ix[f]] = v
    return V()


def repair_cic(sp, X):
    """CORRECTED 2026-11-02: projection onto the ELIGIBLE CIC-IDS2017 constraints only
    (protocol v0.4 Section 6.2 / D12; supervisor report 31 Oct 2026, section 3).
    Shared type-awareness (clip + integer round + non-negativity, identical to arm A_typed),
    then the eligible relations/identities ONLY: CIC-135 (Min PktLen <= PktLen Mean),
    CIC-136 (PktLen Mean <= Max PktLen), CIC-137 (PktLen Std <= Max-Min PktLen),
    CIC-161 (Avg Fwd Segment Size = Fwd Packet Length Mean),
    CIC-165 (Avg Bwd Segment Size = Bwd Packet Length Mean).
    Every ineligible projection present in the earlier arm B (per-direction FwdLen/BwdLen and
    IAT/Active/Idle orderings, Max/Min PktLen = f(Fwd,Bwd) identities, mean-from-total and
    avg-segment-from-total T4 formulas, conditional zeros, header-length bounds, subflow/flag
    bounds, Down/Up ratio, packets/s rates, signed-OK clipping) is REMOVED."""
    np.clip(X, sp.lo, sp.hi, out=X)
    sp.type_repair(X)
    V = _cols(sp, X)
    # CIC-135 and CIC-136: Min Packet Length <= Packet Length Mean <= Max Packet Length
    V['Packet Length Mean'] = np.clip(V['Packet Length Mean'], V['Min Packet Length'], V['Max Packet Length'])
    # CIC-137: Packet Length Std <= (Max Packet Length - Min Packet Length)
    V['Packet Length Std'] = np.minimum(V['Packet Length Std'], V['Max Packet Length'] - V['Min Packet Length'])
    # CIC-161 / CIC-165: Avg {Fwd,Bwd} Segment Size identical to {Fwd,Bwd} Packet Length Mean (feature as given)
    V['Avg Fwd Segment Size'] = V['Fwd Packet Length Mean']
    V['Avg Bwd Segment Size'] = V['Bwd Packet Length Mean']
    return X


def repair_unsw(sp, X, X0):
    """CORRECTED 2026-11-02: projection onto the ELIGIBLE UNSW-NB15 constraints only
    (protocol v0.4 Section 6.2 / D12; supervisor report 31 Oct 2026, section 3).
    Shared type-awareness (clip + integer round + non-negativity, identical to arm A_typed),
    then the eligible identities ONLY: UNSW-090 (tcprtt = synack + ackdat),
    UNSW-091 (smeansz = sbytes / Spkts, tol 1), UNSW-092 (dmeansz = dbytes / Dpkts, tol 1).
    Every ineligible projection present in the earlier arm B (ttl<=255 range, is_ftp_login/ct_*
    range bounds, sbytes>=Spkts / dbytes>=Dpkts, sloss/dloss bounds, ct_* subset relations,
    conditional destination/TCP/ICMP/FTP zeros, tcprtt<=dur rescaling, state resets) is REMOVED.
    X0 is retained in the signature for call compatibility and is no longer read."""
    np.clip(X, sp.lo, sp.hi, out=X)
    sp.type_repair(X)
    V = _cols(sp, X)
    # UNSW-090: tcprtt = synack + ackdat  (NO tcprtt<=dur rescaling; that condition is T3, ineligible)
    V['tcprtt'] = V['synack'] + V['ackdat']
    # UNSW-091: |smeansz - sbytes/Spkts| <= 1
    qs = V['sbytes'] / np.maximum(V['Spkts'], 1.0)
    V['smeansz'] = np.where(np.abs(V['smeansz'] - qs) <= 1, V['smeansz'], np.round(qs))
    # UNSW-092: |dmeansz - dbytes/Dpkts| <= 1  (applies when Dpkts > 0)
    qd = V['dbytes'] / np.maximum(V['Dpkts'], 1.0)
    V['dmeansz'] = np.where((V['Dpkts'] > 0) & (np.abs(V['dmeansz'] - qd) > 1), np.round(qd), V['dmeansz'])
    return X


# --------------------------------------------------------------------------------------------
def p_benign(model, X2d):
    return model.predict_proba(X2d)[:, 0]


def genetic_search(sp, model, X0, mode, seed):
    """mode in {'A', 'A_typed', 'B'}. Returns dict with the single selected candidate per instance."""
    rng = np.random.default_rng(seed)
    N, p = X0.shape
    mut = ~sp.immutable
    P0 = sp.pct(X0)
    X0b = X0[:, None, :]

    def fix(X):
        if mode == 'B': return sp.repair(X, X0b)
        np.clip(X, sp.lo, sp.hi, out=X)
        if mode == 'A_typed': sp.type_repair(X)
        return X

    def score(X):
        n, k, _ = X.shape
        pb = p_benign(model, X.reshape(-1, p)).reshape(n, k)
        d = sp.dist_pct(P0[:, None, :], sp.pct(X)) + SPARSITY_EPS * sp.l0(X0b, X) / p
        valid = pb > 0.5
        s = np.where(valid, d, 10.0 + (1.0 - pb))
        feas = np.ones((n, k), dtype=bool)
        if mode == 'B':
            viol, _ = sp.audit(X)
            # CORRECTED 2026-11-02: feasibility is defined over the ELIGIBLE subset only
            # (protocol v0.4 Section 6.2 / D12). The full retained audit is still saved to the
            # npz by the driver; only arm B's in-loop feasibility ranking is restricted here.
            nv = viol[..., sp.elig_mask].sum(-1) + sp.out_of_range(X)
            feas = nv == 0
            s = np.where(feas, s, 1000.0 + 20.0 * nv + s)
        return s, valid, feas, pb

    # initial population: original with 1-5 features copied from one benign donor
    X = np.repeat(X0b, POP, axis=1)
    don = sp.donors[rng.integers(len(sp.donors), size=(N, POP))]
    k = rng.integers(1, 6, size=(N, POP, 1))
    r = rng.random((N, POP, p)); r[..., ~mut] = 2.0
    thr = np.take_along_axis(np.sort(r, axis=-1), k - 1, axis=-1)
    m0 = r <= thr
    X = np.where(m0, don, X); X = fix(X)
    s, valid, feas, pb = score(X)
    nq = POP
    for g in range(GENS):
        order = np.argsort(s, axis=1)[:, :ELITE]
        E = np.take_along_axis(X, order[..., None], axis=1); sE = np.take_along_axis(s, order, axis=1)
        vE = np.take_along_axis(valid, order, axis=1); fE = np.take_along_axis(feas, order, axis=1); pE = np.take_along_axis(pb, order, axis=1)
        nc = POP - ELITE
        pa = rng.integers(ELITE, size=(N, nc)); pb_ = rng.integers(ELITE, size=(N, nc))
        A_ = np.take_along_axis(E, pa[..., None], axis=1); B_ = np.take_along_axis(E, pb_[..., None], axis=1)
        C = np.where(rng.random((N, nc, p)) < 0.5, A_, B_)
        mm = (rng.random((N, nc, p)) < 3.0 / p) & mut
        kind = rng.random((N, nc, p))
        don = sp.donors[rng.integers(len(sp.donors), size=(N, nc))]
        gauss = C + rng.normal(size=(N, nc, p)) * sp.sigma
        use_d = mm & ((kind < P_DONOR) | (sp.cat & (kind < P_DONOR + P_GAUSS)))
        use_g = mm & (kind >= P_DONOR) & (kind < P_DONOR + P_GAUSS) & ~sp.cat
        use_r = mm & (kind >= P_DONOR + P_GAUSS)
        C = np.where(use_d, don, C); C = np.where(use_g, gauss, C); C = np.where(use_r, np.broadcast_to(X0b, C.shape), C)
        C = fix(C)
        sC, vC, fC, pC = score(C); nq += nc
        X = np.concatenate([E, C], axis=1); s = np.concatenate([sE, sC], axis=1)
        valid = np.concatenate([vE, vC], axis=1); feas = np.concatenate([fE, fC], axis=1); pb = np.concatenate([pE, pC], axis=1)
    best = np.argmin(s, axis=1); ar = np.arange(N)
    cf = X[ar, best]
    ok = valid[ar, best] & feas[ar, best]
    return dict(cf=cf, returned=ok, p_benign=pb[ar, best], queries=nq)


def nearest_real(sp, model, X0):
    """Arm C: nearest training benign record that the detector classifies benign and shares the immutable features."""
    pool = sp.donors
    keep = p_benign(model, pool) > 0.5
    pool = pool[keep]
    Pp = sp.pct(pool).astype(np.float32); P0 = sp.pct(X0).astype(np.float32)
    cat = sp.cat; N = len(X0)
    cf = X0.copy(); found = np.zeros(N, dtype=bool); pool_row = -np.ones(N, dtype=np.int64)
    orig_rows = sp.ben_pool_idx[keep]
    for i in range(N):
        el = np.ones(len(pool), dtype=bool)
        for j in np.flatnonzero(sp.immutable): el &= pool[:, j] == X0[i, j]
        if not el.any(): continue
        idx = np.flatnonzero(el)
        d = np.abs(Pp[idx] - P0[i]); d[:, cat] = d[:, cat] > 0
        b = idx[np.argmin(d.sum(1))]
        cf[i] = pool[b]; found[i] = True; pool_row[i] = orig_rows[b]
    return dict(cf=cf, returned=found, p_benign=p_benign(model, cf), pool_size=int(len(pool)), train_row=pool_row)
