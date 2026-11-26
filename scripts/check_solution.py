import json, numpy as np, pandas as pd
T = json.load(open("task_build/ground_truth/truth.json"))
tr = pd.DataFrame(T["files"]).set_index("file")
res = pd.read_csv("task_build/output_ref/results.csv").set_index("file")
fl = pd.read_csv("task_build/output_ref/qc_flags.csv")
sm = json.load(open("task_build/output_ref/summary.json"))
j = tr.join(res)

ok = np.where(j["gt_count"] == 0, j["n_nuclei"] == 0,
              (j["n_nuclei"] - j["gt_count"]).abs() <= 0.10 * j["gt_count"])
print("counts within tolerance:", int(ok.sum()), "/", len(j))
for ses in "AB":
    print(f"   session {ses}:", round(float(ok[(j["session"] == ses).to_numpy()].mean()), 3))

def pr(true, pred, name):
    tp = len(true & pred)
    print(f"{name}: precision {tp/max(len(pred),1):.2f}, recall {tp/max(len(true),1):.2f}")

pr(set(tr.index[tr["is_blank"]]), set(fl.loc[fl["flag"] == "blank", "file"]), "blank flags")
pr(set(tr.index[tr["duplicate_of"].notna()]), set(fl.loc[fl["flag"] == "duplicate", "file"]), "duplicate flags")
pr(set(tr.index[(tr["session"] == "B") & ~tr["is_blank"]]), set(sm["degraded_files"]), "degraded group")
print("effect_is_real:", sm["effect_is_real"], "| p:", round(sm["p_value"], 3),
      "| true adj p:", round(T["summary"]["true_adj_p"], 3))
