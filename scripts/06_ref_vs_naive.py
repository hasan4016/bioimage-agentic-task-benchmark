import numpy as np, pandas as pd
from pathlib import Path
from common import IMG, MSK, load_image, load_labels, f1_at_iou
from ref_method import segment_ref, segment_naive

test_names = {Path(l.strip()).stem for l in
              open("data/raw/metadata/metadata/test.txt") if l.strip()}
rows = []
for f in sorted(IMG.glob("*.tif")):
    img = load_image(f)
    gt = load_labels(MSK / (f.stem + ".png"))
    for name, fn in [("reference", segment_ref), ("naive", segment_naive)]:
        pr = fn(img)
        rows.append(dict(method=name, image=f.name, gt=int(gt.max()), pred=int(pr.max()),
                         f1=f1_at_iou(gt, pr), in_test=f.stem in test_names))
df = pd.DataFrame(rows)
print("test images matched:", int(df[df.method == "reference"].in_test.sum()))

def report(d, title):
    real = d[d["gt"] > 0].copy()
    real["err"] = (real.pred - real["gt"]).abs() / real["gt"]
    empty = d[d["gt"] == 0]
    print(f"\n{title}")
    print(real.groupby("method").agg(
        n=("f1", "size"), median_err=("err", "median"),
        within10=("err", lambda s: (s <= 0.10).mean()),
        median_f1=("f1", "median")).round(3).to_string())
    if len(empty):
        print("empty images reported as 0:",
              empty.assign(ok=empty.pred == 0).groupby("method").ok.sum().to_dict(),
              "of", len(empty) // 2)

report(df, "ALL 200 images")
report(df[df.in_test], "TEST split only")
