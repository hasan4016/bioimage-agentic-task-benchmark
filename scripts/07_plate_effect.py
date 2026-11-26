import numpy as np, pandas as pd
from scipy import stats
from common import MSK, load_labels

df = pd.read_csv("data/raw/plates_clean.csv")
df["count"] = [int(load_labels(MSK / m).max()) for m in df["mask_file"]]
real = df[df["count"] > 0]

print(real.groupby("plate")["count"].agg(["size", "mean", "std"]).round(1).to_string())
groups = [g["count"].values for _, g in real.groupby("plate") if len(g) > 1]
F, p = stats.f_oneway(*groups)
grand = real["count"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = ((real["count"] - grand) ** 2).sum()
print(f"\nANOVA across plates: F={F:.2f}, p={p:.4g}")
print(f"share of count variance explained by plate: {ss_between / ss_total:.2f}")
