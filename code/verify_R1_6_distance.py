#!/usr/bin/env python3
"""R1.6 verification: recompute per-arm percentile distances (M4 = d_pct), the
B-vs-C paired medians/ratios, the A_typed-vs-B direction split, the matched-pairs
rank-biserial effect size, and the count of exactly-zero arm-B distances, directly
from the saved per-instance arrays. Nothing is copied from a prior summary.

Definitions (explicit, so the numbers are unambiguous):
- An arm "converts" instance i when flipped[i] is True (p_benign>0.5 on the stored candidate).
- distance = d_pct (percentile/ECDF distance in [0,1]) stored per instance.
- B-vs-C pairing: instances converted by BOTH B and C (arm C exists only where a nearest
  real benign flow was found: 1000/1000 on CIC-IDS2017, 722/1000 on UNSW-NB15).
- A_typed-vs-B pairing: instances converted by BOTH A_typed and B (1000/1000 in every cell,
  since both generated arms convert all instances).
- rank-biserial: matched-pairs effect from the Wilcoxon signed-rank test on the paired
  differences, convention diff = d_pct(B) - d_pct(A_typed); r = (R+ - R-)/(R+ + R-) over
  non-zero differences (zeros dropped, 'wilcox' zero method). r>0 => B has the larger
  (farther) distance. p is the two-sided Wilcoxon p-value on the same pairs.
Usage: python verify_R1_6_distance.py <raw_arrays_dir> [out.json]
"""
import sys, os, json, glob
import numpy as np
from scipy import stats


def holm(pvals):
    """Holm step-down adjusted p-values (monotone), m = number of cells in the family.
    Matches the deposited confirmatory analysis routine."""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        a = (m - rank) * pvals[idx]
        run = max(run, a)
        adj[idx] = min(run, 1.0)
    return adj

RAW = sys.argv[1] if len(sys.argv) > 1 else "evidence/arrays/confirmatory"
# Arm B enforcement was re-run on the eligible list (manuscript Section 3.5, deviation 1); the
# deposited analysis reads arm B from the corrected set. A_typed and C come from the confirmatory
# raw set. Point RAW_B at the corrected arm-B arrays so the distances match the deposited record.
RAW_B = sys.argv[3] if len(sys.argv) > 3 else "evidence/arrays/B_corrected"
OUT = sys.argv[2] if len(sys.argv) > 2 else "R1_6_distance_verified.json"
CELLS = [f"{ds}_{det}" for ds in ("cic", "unsw") for det in ("RF", "XGB", "HGB")]

def load(cell, arm):
    # arm in {'A_typed','B','C'}; seed 10 (primary). Arm B from the corrected (eligible-list) set.
    base = RAW_B if arm == "B" else RAW
    f = os.path.join(base, f"{cell}_{arm}_seed10.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f, allow_pickle=True)
    # "converted" = flipped AND returned, identical to the deposited analysis (conv = (pb>0.5)&ret)
    conv = (z["p_benign"] > 0.5) & z["returned"]
    return dict(flipped=conv, d=z["d_pct"].astype(float), returned=z["returned"])

def med_ratio(b, c, m):
    rb = b[m]; rc = c[m]
    nz = rb > 0
    # per-case C/B ratio (how many times farther C is than B), median over cases with B>0
    cob = float(np.median((rc[nz] / rb[nz]))) if nz.any() else None
    return float(np.median(rb)), float(np.median(rc)), (float(np.median(rc)) / float(np.median(rb)) if np.median(rb) > 0 else None), cob

res = {}
for cell in CELLS:
    A = load(cell, "A_typed"); B = load(cell, "B"); C = load(cell, "C")
    if A is None or B is None or C is None:
        res[cell] = {"error": "missing array"}; continue
    bc = B["flipped"] & C["flipped"]           # B-vs-C paired (C-limited)
    ab = A["flipped"] & B["flipped"]           # A-vs-B paired (all)
    medB, medC, ratio_med, med_ratios = med_ratio(B["d"], C["d"], bc)
    # A-vs-B direction split on d_pct
    dB = B["d"][ab]; dA = A["d"][ab]
    diff = dB - dA
    closerB = int((dB < dA).sum()); fartherB = int((dB > dA).sum()); equal = int((dB == dA).sum())
    nz = diff != 0
    if nz.sum() >= 1 and (dB != dA).any():
        w = stats.wilcoxon(dB[nz], dA[nz], zero_method="wilcox", alternative="two-sided")
        ranks = stats.rankdata(np.abs(diff[nz]))
        Rpos = float(ranks[diff[nz] > 0].sum()); Rneg = float(ranks[diff[nz] < 0].sum())
        rb = (Rpos - Rneg) / (Rpos + Rneg)
        pval = float(w.pvalue)
    else:
        rb = 0.0; pval = 1.0
    # B-vs-C direction split + rank-biserial on the B-vs-C paired set (diff = d_pct(B) - d_pct(C))
    dBc = B["d"][bc]; dCc = C["d"][bc]
    diffc = dBc - dCc
    bc_closerB = int((dBc < dCc).sum()); bc_fartherB = int((dBc > dCc).sum()); bc_equal = int((dBc == dCc).sum())
    nzc = diffc != 0
    if nzc.sum() >= 1:
        wc = stats.wilcoxon(dBc[nzc], dCc[nzc], zero_method="wilcox", alternative="two-sided")
        rc_ = stats.rankdata(np.abs(diffc[nzc]))
        rbc = (rc_[diffc[nzc] > 0].sum() - rc_[diffc[nzc] < 0].sum()) / rc_.sum()
        pbc = float(wc.pvalue)
    else:
        rbc = 0.0; pbc = 1.0
    # arm-B exactly-zero distance among B-converted (direct count, not inferred from a median)
    b_conv = B["flipped"]
    b_zero = int((B["d"][b_conv] == 0).sum())
    res[cell] = {
        "n_BC_paired": int(bc.sum()), "medB_vs_C": round(medB, 5), "medC": round(medC, 5),
        "ratio_of_medians": None if ratio_med is None else round(ratio_med, 1),
        "median_of_per_case_C_over_B": None if med_ratios is None else round(med_ratios, 1),
        "n_AB_paired": int(ab.sum()),
        "AvsB_B_closer": closerB, "AvsB_B_farther": fartherB, "AvsB_equal": equal,
        "AvsB_rank_biserial_B_minus_A": round(rb, 3),
        "AvsB_wilcoxon_p": pval,
        "BvsC_B_closer": bc_closerB, "BvsC_B_farther": bc_fartherB, "BvsC_equal": bc_equal,
        "BvsC_rank_biserial_B_minus_C": round(rbc, 3), "BvsC_wilcoxon_p": pbc,
        "B_converted_n": int(b_conv.sum()), "B_zero_distance_count": b_zero,
    }

# ---- Holm correction within the paired families (m = 6 cells), matching the manuscript ---------
# F2 family: M4 A_typed vs B (AvsB Wilcoxon). F3 family: M4 B vs C (BvsC Wilcoxon).
# A cell with < 100 usable pairs would be treated as non-rejected (p=1) before Holm; here every
# cell has 1000 (A-vs-B) or >=722 (B-vs-C) pairs, so none is inconclusive. The raw p-values are the
# pre-Holm values the manuscript reports (e.g. UNSW-NB15/HGB rank-biserial +0.177, p = 0.00112...).
ok_cells = [c for c in CELLS if "error" not in res[c]]
for fam_key, fam_desc in [("AvsB_wilcoxon_p", "F2: M4 A_typed vs B"),
                          ("BvsC_wilcoxon_p", "F3: M4 B vs C")]:
    praw = [res[c][fam_key] for c in ok_cells]
    padj = holm(praw)
    for c, pa in zip(ok_cells, padj):
        res[c][fam_key + "_holm"] = float(pa)
        res[c][fam_key + "_sig_holm_0.05"] = bool(pa < 0.05)

json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
print("\n=== Holm-corrected Wilcoxon p within each family (m=6) ===")
for c in ok_cells:
    print("  %-10s  A-vs-B: rb=%+0.3f p=%.6g p_holm=%.4g | B-vs-C: rb=%+0.3f p=%.3g p_holm=%.3g" % (
        c, res[c]["AvsB_rank_biserial_B_minus_A"], res[c]["AvsB_wilcoxon_p"],
        res[c]["AvsB_wilcoxon_p_holm"], res[c]["BvsC_rank_biserial_B_minus_C"],
        res[c]["BvsC_wilcoxon_p"], res[c]["BvsC_wilcoxon_p_holm"]))
print(f"\nwritten: {OUT}")
