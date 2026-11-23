#!/usr/bin/env python3
"""Confirmatory analysis (Paper 4) per protocol §6/§7 + 26 Oct decisions.
Five states, M1/M2/M3 (+M4/M5/M6), on the REGISTER-ELIGIBLE conditions (44 CIC / 73 UNSW).
Primary tests on the PRIMARY seed 10: M2 vs 5% (Wilson); M1 A_typed vs B (exact McNemar, F1);
M4 A_typed vs B (Wilcoxon, F2); M4 B vs C (Wilcoxon, F3). Holm within each family of 6
(dataset x detector) cells, m=6, with any inconclusive cell (<100 converted / pair) treated as a
non-rejected hypothesis p=1 (26 Oct decision f). Additional seeds 11,12 report M1/M2/M4 as ranges
(not pooled). Descriptive: A_typed M2 excluding near-duplicate new cases (26 Oct (d)), outside Holm.
Metrics recomputed from the saved arrays; nothing copied from the exploratory run.
"""
import sys, os, json, numpy as np
from scipy import stats
# Portability: original deposited run used PROJ="/home/claude/paper4". To run elsewhere,
# set PAPER4_ROOT to the package root (the folder holding code/, work_b/, results_confirmatory/).
# The fallback preserves the original deposited path so recorded behaviour is unchanged.
PROJ=os.environ.get("PAPER4_ROOT", "."); os.chdir(PROJ); sys.path.insert(0,"code")
RAW=f"{PROJ}/results_confirmatory/raw"; RAW_B=f"{PROJ}/results_confirmatory_B_corrected/raw"; OUT=f"{PROJ}/results_confirmatory_B_corrected"; os.makedirs(OUT,exist_ok=True)
R=json.load(open("work_b/register_extract.json"))
DETS=["RF","XGB","HGB"]; DSS=["cic","unsw"]

def elig_map(ds):
    sheet="CIC-IDS2017" if ds=="cic" else "UNSW-NB15"
    el={r["code"]:r["cls"][:1] for r in R[sheet] if r["elig_e"] and r["elig_e"].startswith("Eligible")}
    return el

# ---- near-duplicate per-case flags (26 Oct (d)): build once per ds -----------------------
def neardup_flags():
    from common import load, SEED
    from generator import Space
    cic_day=np.load("work_b/item4/out/cic_test_day.npy", allow_pickle=True)
    flags={}
    for ds in DSS:
        Xtr,ytr,ctr,feats,meta=load(ds,"train"); sp=Space(ds,Xtr,ytr,feats,meta,SEED); del Xtr,ytr
        Xte,yte,cte,_,_=load(ds,"test")
        conf=np.load(f"work_b/item4/out/{ds}_conf_idx_20261008.npy"); pilot=np.load(f"data/{ds}_orig_idx.npy")
        Xn,Xp=Xte[conf],Xte[pilot]; cn,cp=cte[conf],cte[pilot]; Pn,Pp=sp.pct(Xn),sp.pct(Xp)
        dn = cic_day[conf] if ds=="cic" else None; dp = cic_day[pilot] if ds=="cic" else None
        f=np.zeros(len(conf),dtype=bool)
        for i in range(len(conf)):
            if ds=="cic": pj=np.flatnonzero((cp==cn[i])&(dp==dn[i]))
            else: pj=np.flatnonzero(cp==cn[i])
            if len(pj)==0: continue
            d=np.abs(Pn[i:i+1]-Pp[pj]);
            if sp.cat.any(): d[:,:,sp.cat] if d.ndim==3 else None
            dd=np.abs(Pn[i]-Pp[pj]); dd[:,sp.cat]=(Xn[i][sp.cat]!=Xp[pj][:,sp.cat]).astype(float)
            if dd.max(1).min()<=0.01: f[i]=True
        flags[ds]=f
        print(f"[neardup flags] {ds}: {f.sum()}/{len(f)}", flush=True)
    return flags

def states_for(z, elig, cls2_cols, elig_cols, is_C):
    pb=z["p_benign"]; ret=z["returned"]; viol=z["viol"]; N=len(pb)
    tr=z["train_row"] if "train_row" in z.files else np.zeros(0)
    no_cand=(tr==-1) if (is_C and tr.size==N) else ~ret if is_C else np.zeros(N,bool)
    # final candidate exists unless arm C had no pool flow (returned False)
    exists = ret if is_C else np.ones(N,bool)
    conv = (pb>0.5) & exists
    v23 = viol[:,cls2_cols].any(1) if len(cls2_cols) else np.zeros(N,bool)
    vel = viol[:,elig_cols].any(1)                         # any eligible (class 1-3) violation
    v1only = vel & ~v23
    s1 = int((~exists).sum()); s2=int((exists&~conv).sum())
    s4 = int((conv&v23).sum()); s3=int((conv&v1only).sum()); s5=int((conv&~vel).sum())
    conv_mask = conv
    return dict(N=N,s1=s1,s2=s2,s3=s3,s4=s4,s5=s5,conv=conv_mask,
                M1=(s3+s4+s5)/N, M2=(s4/(s3+s4+s5) if (s3+s4+s5) else None), M3=s5/N,
                n_conv=int(conv.sum()))

def load_cell(ds,det,arm,seed,elig,cls2set):
    base=RAW_B if arm=="B" else RAW
    fn=f"{base}/{ds}_{det}_{arm}_seed{seed}.npz"
    if not os.path.exists(fn): return None
    z=np.load(fn, allow_pickle=True)
    ids=list(z["constraint_ids"]); col={c:i for i,c in enumerate(ids)}
    elig_cols=[col[c] for c in elig if c in col]
    cls2_cols=[col[c] for c in cls2set if c in col]
    st=states_for(z, elig, cls2_cols, elig_cols, is_C=(arm=="C"))
    st["d_pct"]=z["d_pct"]; st["d_gower"]=z["d_gower"]; st["l0"]=z["l0"]; st["z"]=z
    return st

def wilson(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)

def holm(pvals):
    m=len(pvals); order=np.argsort(pvals); adj=np.empty(m); run=0.0
    for rank,idx in enumerate(order):
        a=(m-rank)*pvals[idx]; run=max(run,a); adj[idx]=min(run,1.0)
    return adj

ndflags=neardup_flags()
cells={}
elig={ds:elig_map(ds) for ds in DSS}
cls2={ds:set(c for c,k in elig[ds].items() if k=="2") for ds in DSS}
report={"generator_primary_seed":10,"additional_seeds":[11,12],"eligible_counts":{ds:{"all":len(elig[ds]),"class2":len(cls2[ds])} for ds in DSS},"cells":{},"families":{},"seed_ranges":{},"m2_neardup_sensitivity":{}}

# primary seed 10 per cell
for ds in DSS:
    for det in DETS:
        key=f"{ds}_{det}"
        A=load_cell(ds,det,"A_typed",10,elig[ds],cls2[ds])
        B=load_cell(ds,det,"B",10,elig[ds],cls2[ds])
        Cc=load_cell(ds,det,"C",10,elig[ds],cls2[ds])
        Araw=load_cell(ds,det,"A",10,elig[ds],cls2[ds])
        rec={a:{k:v for k,v in dd.items() if k not in ("conv","d_pct","d_gower","l0","z")} for a,dd in
             [("A_typed",A),("B",B),("C",Cc),("A_raw",Araw)] if dd}
        # M2 Wilson on A_typed converted
        nconvA=A["n_conv"]; s4A=A["s4"]
        lo,hi=wilson(s4A,nconvA)
        m2crit="inconclusive (<100 converted)" if nconvA<100 else ("supported (>5%)" if lo and lo>0.05 else ("not supported (<5%)" if hi and hi<0.05 else "inconclusive"))
        rec["M2_vs_5pct_A_typed"]={"n_converted":nconvA,"state4":s4A,"M2":A["M2"],"wilson95":[lo,hi],"verdict":m2crit}
        # M4/M5/M6 medians on converted
        for nm,dd in [("A_typed",A),("B",B),("C",Cc)]:
            if dd and dd["conv"].any():
                rec[nm]["M4_pct_med"]=float(np.median(dd["d_pct"][dd["conv"]]))
                rec[nm]["M5_gower_med"]=float(np.median(dd["d_gower"][dd["conv"]]))
                rec[nm]["M6_l0_med"]=float(np.median(dd["l0"][dd["conv"]]))
        report["cells"][key]={"seed10":rec}
        cells[key]={"A":A,"B":B,"C":Cc}

# families (Holm m=6) on seed 10
def mcnemar_p(a_succ,b_succ):
    b=int((a_succ&~b_succ).sum()); c=int((~a_succ&b_succ).sum())
    if b+c==0: return 1.0,b,c
    return float(stats.binomtest(min(b,c),b+c,0.5).pvalue),b,c
def wilcoxon_p(dA,dB):
    m=~np.isnan(dA)&~np.isnan(dB)
    if m.sum()<1 or np.all(dA[m]==dB[m]): return 1.0,int(m.sum())
    try: return float(stats.wilcoxon(dA[m],dB[m]).pvalue),int(m.sum())
    except Exception: return 1.0,int(m.sum())

fam_keys=[f"{ds}_{det}" for ds in DSS for det in DETS]
for fam,desc in [("F1_M1_Atyped_vs_B","M1 A_typed vs B, exact McNemar, all evaluated"),
                 ("F2_M4_Atyped_vs_B","M4 A_typed vs B, Wilcoxon, converted in both"),
                 ("F3_M4_B_vs_C","M4 B vs C, Wilcoxon, converted in both")]:
    raw=[]; info=[]
    for key in fam_keys:
        A=cells[key]["A"]; B=cells[key]["B"]; Cc=cells[key]["C"]
        if fam.startswith("F1"):
            aS=A["conv"]; bS=B["conv"]
            # 100 threshold: evaluated instances in both = all N; inconclusive if either arm <100 converted
            incon = (A["n_conv"]<100) or (B["n_conv"]<100)
            if incon: p,b,c=1.0,None,None; note="inconclusive (<100 converted) -> p=1"
            else: p,b,c=mcnemar_p(aS,bS); note=""
            info.append({"cell":key,"p":p,"b_Aonly":b,"c_Bonly":c,"inconclusive":incon,"note":note})
        elif fam.startswith("F2"):
            both=A["conv"]&B["conv"]; n=int(both.sum())
            incon=n<100
            if incon: p=1.0; note="inconclusive (<100 usable pairs) -> p=1"
            else: p,_=wilcoxon_p(A["d_pct"][both],B["d_pct"][both]); note=""
            info.append({"cell":key,"p":p,"n_pairs":n,"inconclusive":incon,
                         "median_A":float(np.median(A["d_pct"][both])) if n else None,
                         "median_B":float(np.median(B["d_pct"][both])) if n else None,"note":note})
        else:
            both=B["conv"]&Cc["conv"]; n=int(both.sum())
            incon=n<100
            if incon: p=1.0; note="inconclusive (<100 usable pairs) -> p=1"
            else: p,_=wilcoxon_p(B["d_pct"][both],Cc["d_pct"][both]); note=""
            info.append({"cell":key,"p":p,"n_pairs":n,"inconclusive":incon,
                         "median_B":float(np.median(B["d_pct"][both])) if n else None,
                         "median_C":float(np.median(Cc["d_pct"][both])) if n else None,"note":note})
        raw.append(info[-1]["p"])
    adj=holm(np.array(raw))
    for i,x in enumerate(info): x["p_holm"]=float(adj[i]); x["sig_0.05"]=bool(adj[i]<0.05)
    report["families"][fam]={"description":desc,"m":6,"cells":info,
                             "n_inconclusive":sum(1 for x in info if x.get("inconclusive"))}

# seeds 11,12 ranges (A_typed, B): M1, M2, M4 median
for ds in DSS:
    for det in DETS:
        key=f"{ds}_{det}"; r={}
        for arm in ["A_typed","B"]:
            vals={"M1":[],"M2":[],"M4_med":[]}
            for sd in [10,11,12]:
                c=load_cell(ds,det,arm,sd,elig[ds],cls2[ds])
                if not c: continue
                vals["M1"].append(round(c["M1"],4)); vals["M2"].append(round(c["M2"],4) if c["M2"] is not None else None)
                vals["M4_med"].append(round(float(np.median(c["d_pct"][c["conv"]])),5) if c["conv"].any() else None)
            r[arm]=vals
        report["seed_ranges"][key]=r

# M2 near-dup sensitivity (A_typed, seed 10), descriptive, outside Holm
for ds in DSS:
    f=ndflags[ds]
    for det in DETS:
        key=f"{ds}_{det}"; A=cells[key]["A"]; z=A["z"]
        pb=z["p_benign"]; viol=z["viol"]; ids=list(z["constraint_ids"]); col={c:i for i,c in enumerate(ids)}
        cls2_cols=[col[c] for c in cls2[ds] if c in col]
        conv=A["conv"]; v23=viol[:,cls2_cols].any(1)
        keep=~f                       # exclude near-dup new cases
        cm=conv&keep; nconv=int(cm.sum()); s4=int((cm&v23).sum())
        report["m2_neardup_sensitivity"][key]={"n_converted_excl_neardup":nconv,"state4":s4,
            "M2_excl_neardup":(s4/nconv if nconv else None),
            "M2_all":A["M2"],"n_excluded_neardup":int(f.sum())}

json.dump(report, open(f"{OUT}/confirmatory_analysis_corrected.json","w"), indent=1, ensure_ascii=False)
# print summary
print("\n=== M2 vs 5% (A_typed, seed 10) ===")
for key in fam_keys:
    r=report["cells"][key]["seed10"]["M2_vs_5pct_A_typed"]
    print(f"  {key:9s} n_conv={r['n_converted']:>4} M2={r['M2']} wilson={[round(x,4) if x else None for x in r['wilson95']]}  {r['verdict']}")
print("\n=== Holm families (seed 10, m=6) ===")
for fam,v in report["families"].items():
    print(f"  {fam} (inconclusive={v['n_inconclusive']}):")
    for x in v["cells"]:
        print(f"     {x['cell']:9s} p={x['p']:.3g} p_holm={x['p_holm']:.3g} sig={x['sig_0.05']} {x.get('note','')}")
print("\nwrote confirmatory_analysis_corrected.json")
