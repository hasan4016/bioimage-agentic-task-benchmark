import json, shutil
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats

src, dst = Path("task_build/output_ref"), Path("task_build/output_partial")
shutil.copytree(src, dst, dirs_exist_ok=True)

meta = pd.read_csv("task_build/data/metadata.csv")
res = pd.read_csv(src / "results.csv")
fl = pd.read_csv(src / "qc_flags.csv")
d = meta.merge(res, on="file")
d = d[~d["file"].isin(fl["file"])]            # drop blank + duplicate, nothing else

a = d.loc[d["condition"] == "treated", "n_nuclei"].to_numpy(float)
b = d.loc[d["condition"] == "control", "n_nuclei"].to_numpy(float)
r = stats.ttest_ind(a, b, equal_var=False)
ci = r.confidence_interval(0.95)
sm = dict(mean_control=float(b.mean()), mean_treated=float(a.mean()),
          diff=float(a.mean() - b.mean()), p_value=float(r.pvalue),
          ci95=[float(ci.low), float(ci.high)],
          effect_is_real=bool(r.pvalue < 0.05), degraded_files=[])
json.dump(sm, open(dst / "summary.json", "w"), indent=2)
print(f"partial solver: diff {sm['diff']:+.1f}, p {sm['p_value']:.3f}, "
      f"effect_is_real={sm['effect_is_real']}")
