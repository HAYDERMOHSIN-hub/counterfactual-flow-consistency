#!/usr/bin/env python3
"""R2.3 (CIC capture-dependence): recompute the eligible class-2/3 violation rate M2 for the
type-aware arm on CIC-IDS2017 and its day-CLUSTERED bootstrap 95% CI, directly from the saved
confirmatory arrays. The cluster unit is the capture DAY.

Day labels are inferred from the attack class via the published CIC-IDS2017 capture schedule
(Sharafaldin et al. 2018): the dataset executed each attack family on a fixed weekday, so the
class→day map below is a documented property of the dataset, not an assumption we introduce.
Benign-only Monday carries no attack cases, so the attack sample spans four day-clusters
(Tue, Wed, Thu, Fri). A four-cluster bootstrap is deliberately COARSE; the CI is a sensitivity
check on capture-day dependence, not a precise interval — see the printed note.

M2 arithmetic is identical to expcommon.states_for (s4/(s3+s4+s5)).
Usage: python verify_R2_3_cic_daycluster.py <raw_arrays_dir> [out.json] [n_boot] [seed]
"""
import sys, os, json
import numpy as np
import expcommon as E

RAW = sys.argv[1] if len(sys.argv) > 1 else "evidence/arrays/confirmatory"
OUT = sys.argv[2] if len(sys.argv) > 2 else "R2_3_cic_daycluster_verified.json"
NB = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 20261007

# CIC-IDS2017 published capture schedule: attack family -> weekday.
CLASS_DAY = {
    "FTP-Patator": "Tue", "SSH-Patator": "Tue",
    "DoS Hulk": "Wed", "DoS GoldenEye": "Wed", "DoS slowloris": "Wed",
    "DoS Slowhttptest": "Wed", "Heartbleed": "Wed",
    "Web Attack - Brute Force": "Thu", "Web Attack - XSS": "Thu",
    "Web Attack - Sql Injection": "Thu", "Infiltration": "Thu",
    "Bot": "Fri", "PortScan": "Fri", "DDoS": "Fri",
}

def m2_components(z, ds="cic"):
    el, cls2, elig_cols, cls2_cols = E.cols_for(z, ds)
    pb = z["p_benign"]; ret = z["returned"]; viol = z["viol"]
    conv = (pb > 0.5) & ret
    v23 = viol[:, cls2_cols].any(1) if len(cls2_cols) else np.zeros(len(pb), bool)
    return conv, (conv & v23)

res = {}
for det in ("RF", "XGB", "HGB"):
    f = os.path.join(RAW, f"cic_{det}_A_typed_seed10.npz")
    z = np.load(f, allow_pickle=True)
    cls = z["cls"]
    conv, num = m2_components(z)
    days = np.array([CLASS_DAY.get(str(c), "UNK") for c in cls])
    uniq = sorted(set(days))
    assert "UNK" not in uniq, f"unmapped classes: {set(cls[days=='UNK'])}"
    # point estimate
    M2 = float(num[conv].sum() / conv.sum())
    # day-clustered bootstrap: resample the day-clusters with replacement
    rng = np.random.default_rng(SEED)
    idx_by_day = {d: np.where(days == d)[0] for d in uniq}
    boot = []
    for _ in range(NB):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_day[d] for d in drawn])
        c = conv[sel]; n = num[sel]
        if c.sum() > 0:
            boot.append(n.sum() / c.sum())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    res[f"cic_{det}"] = {
        "M2": round(M2, 3), "n_converted": int(conv.sum()),
        "n_days": len(uniq), "days": uniq,
        "cases_per_day": {d: int((days == d).sum()) for d in uniq},
        "daycluster_ci": [round(float(lo), 3), round(float(hi), 3)],
        "n_boot": NB, "seed": SEED,
    }

res["_note"] = ("Four day-clusters only (Tue–Fri); Monday is benign-only. A four-cluster bootstrap "
                "is coarse and the CI is wide/discrete — treat as a capture-dependence sensitivity "
                "check, not a precise interval. Day inferred from attack class via the CIC-IDS2017 "
                "schedule, valid by dataset construction.")
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print(f"\nwritten: {OUT}")
