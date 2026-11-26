import json
import numpy as np, pandas as pd, tifffile
from pathlib import Path
from scipy import stats
from ref_method import segment_naive

DATA = Path("task_build/data")
OUT = Path("task_build/output_naive")
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(DATA / "metadata.csv")
meta["n_nuclei"] = [int(segment_naive(tifffile.imread(DATA / f).astype(np.float32)).max())
                    for f in meta["file"]]
meta[["file", "n_nuclei"]].to_csv(OUT / "results.csv", index=False)
pd.DataFrame(columns=["file", "flag", "reason"]).to_csv(OUT / "qc_flags.csv", index=False)

a = meta.loc[meta["condition"] == "treated", "n_nuclei"].to_numpy(float)
b = meta.loc[meta["condition"] == "control", "n_nuclei"].to_numpy(float)
r = stats.ttest_ind(a, b, equal_var=False)
ci = r.confidence_interval(0.95)
sm = dict(mean_control=float(b.mean()), mean_treated=float(a.mean()),
          diff=float(a.mean() - b.mean()), p_value=float(r.pvalue),
          ci95=[float(ci.low), float(ci.high)], effect_is_real=bool(r.pvalue < 0.05),
          degraded_files=[])
json.dump(sm, open(OUT / "summary.json", "w"), indent=2)
open(OUT / "report.md", "w").write(
    "# Report\nCounted nuclei with global Otsu thresholding and compared treated and "
    "control with a Welch t-test. No exclusions.\n")
print(json.dumps(sm, indent=2))
