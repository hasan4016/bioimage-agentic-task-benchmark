import json, sys
import numpy as np, pandas as pd
from pathlib import Path

REQUIRED = ["mean_control", "mean_treated", "diff", "p_value", "ci95",
            "effect_is_real", "degraded_files"]

def f1(true, pred):
    if not true and not pred:
        return 1.0
    tp = len(true & pred)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 2 * p * r / (p + r)

def grade(out_dir, truth_path="task_build/ground_truth/truth.json"):
    T = json.load(open(truth_path))
    tr = pd.DataFrame(T["files"]).set_index("file")
    out = Path(out_dir)
    s = dict.fromkeys(["counts", "blank", "duplicate", "degraded",
                       "conclusion", "effect_size", "consistency", "format"], 0.0)
    try:
        res = pd.read_csv(out / "results.csv").set_index("file")
        fl = pd.read_csv(out / "qc_flags.csv")
        sm = json.load(open(out / "summary.json"))
    except Exception as e:
        return dict(total=0.0, error=f"could not read outputs: {e}", **s)

    j = tr.join(res["n_nuclei"])
    pred = j["n_nuclei"].to_numpy(float)
    gt = j["gt_count"].to_numpy(float)
    ok = np.where(gt == 0, pred == 0, np.abs(pred - gt) <= 0.10 * gt)
    ok = ok & ~np.isnan(pred)
    s["counts"] = 25 * min(1.0, ok.mean() / 0.75)

    blank_true = set(tr.index[tr["is_blank"]])
    blank_pred = set(fl.loc[fl["flag"] == "blank", "file"])
    s["blank"] = 10 * f1(blank_true, blank_pred)

    dup_copy = tr.index[tr["duplicate_of"].notna()][0]
    pair = {dup_copy, tr.loc[dup_copy, "duplicate_of"]}
    dup_pred = set(fl.loc[fl["flag"] == "duplicate", "file"])
    if len(dup_pred & pair) == 1 and not (dup_pred - pair):
        s["duplicate"] = 10
    elif dup_pred & pair:
        s["duplicate"] = 5

    deg_true = set(tr.index[(tr["session"] == "B") & ~tr["is_blank"]])
    s["degraded"] = 15 * f1(deg_true, set(sm.get("degraded_files", [])))

    if sm.get("effect_is_real") is False:
        s["conclusion"] = 20

    try:
        if abs(float(sm["diff"]) - T["summary"]["true_adj_diff"]) <= 15:
            s["effect_size"] = 10
    except Exception:
        pass

    try:
        lo, hi = sm["ci95"]
        inside = lo <= sm["diff"] <= hi
        agree = (sm["p_value"] < 0.05) == (not (lo <= 0 <= hi))
        s["consistency"] = 5 if (inside and agree and 0 <= sm["p_value"] <= 1) else 0
    except Exception:
        pass

    rep = out / "report.md"
    if all(k in sm for k in REQUIRED) and rep.exists() and rep.stat().st_size > 50 \
            and len(res) == len(tr):
        s["format"] = 5

    s = {k: round(v, 1) for k, v in s.items()}
    s["total"] = round(sum(s.values()), 1)
    return s

if __name__ == "__main__":
    r = grade(sys.argv[1])
    for k, v in r.items():
        print(f"{k:12s} {v}")
