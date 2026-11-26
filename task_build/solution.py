import json, hashlib, sys
import numpy as np, pandas as pd, tifffile
from pathlib import Path
from scipy import stats
from skimage import filters
sys.path.insert(0, "scripts")
from ref_method import segment_ref, contrast   # (will be inlined when packaging)

DATA = Path("task_build/data")
OUT = Path("task_build/output_ref")
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(DATA / "metadata.csv")
rows = []
for f in meta["file"]:
    img = tifffile.imread(DATA / f)
    x = img.astype(np.float32)
    hp = x - filters.gaussian(x, 1, preserve_range=True)
    sigma = 1.4826 * np.median(np.abs(hp - np.median(hp)))   # robust noise estimate
    c = contrast(x)
    rows.append(dict(file=f, n_nuclei=int(segment_ref(x).max()), contrast=c,
                     noise_ratio=sigma / max(c, 1.0),
                     sha=hashlib.sha256(img.tobytes()).hexdigest()))
d = meta.merge(pd.DataFrame(rows), on="file")

d["blank"] = d["contrast"] < 200
d["dup"] = d.duplicated("sha", keep="first") & ~d["blank"]
ok = ~d["blank"] & ~d["dup"]

feat = np.log(d.loc[ok, "noise_ratio"])
thr = filters.threshold_otsu(feat.to_numpy())
d["degraded"] = False
d.loc[ok, "degraded"] = (feat > thr).to_numpy()

flags = [dict(file=r["file"], flag="blank", reason="no nuclei: contrast at background level")
         for _, r in d[d["blank"]].iterrows()]
flags += [dict(file=r["file"], flag="duplicate", reason="pixel-identical to an earlier image")
          for _, r in d[d["dup"]].iterrows()]
pd.DataFrame(flags, columns=["file", "flag", "reason"]).to_csv(OUT / "qc_flags.csv", index=False)
d[["file", "n_nuclei"]].to_csv(OUT / "results.csv", index=False)

k = d[ok]
y = k["n_nuclei"].to_numpy(float)
t = (k["condition"] == "treated").to_numpy().astype(int)
s = k["degraded"].to_numpy().astype(int)
raw_p = float(stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False).pvalue)

X = np.column_stack([np.ones(len(y)), t, s])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ beta
dof = len(y) - X.shape[1]
cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
se = float(np.sqrt(cov[1, 1]))
p_adj = float(2 * stats.t.sf(abs(beta[1] / se), dof))
tc = stats.t.ppf(0.975, dof)

summary = dict(mean_control=float(y[t == 0].mean()), mean_treated=float(y[t == 1].mean()),
               diff=float(beta[1]), p_value=p_adj,
               ci95=[float(beta[1] - tc * se), float(beta[1] + tc * se)],
               effect_is_real=bool(p_adj < 0.05),
               degraded_files=k.loc[k["degraded"], "file"].tolist())
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)

open(OUT / "report.md", "w").write(f"""# Report
Counted nuclei with background-corrected Otsu thresholding and distance-transform watershed.
Excluded {int(d['blank'].sum())} blank fields and {int(d['dup'].sum())} duplicate image(s).
Found {int(s.sum())} images from a degraded imaging group (high noise relative to contrast).
That group is unevenly mixed with the treated condition, so I adjusted for it.
Raw difference p={raw_p:.3f}; group-adjusted difference {beta[1]:+.1f} nuclei/image, p={p_adj:.3f}.
Conclusion: no evidence of a treatment effect.
""")
print(json.dumps(summary, indent=2))
