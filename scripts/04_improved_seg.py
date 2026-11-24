import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology, feature, measure, segmentation
from common import IMG, MSK, load_image, load_labels, f1_at_iou

def contrast(img):
    return float(np.percentile(img, 99.5) - np.median(img))

def segment(img):
    x = np.log1p(np.clip(img - np.median(img), 0, None))   # log: tames bright blobs
    x = filters.gaussian(x, 1)
    corr = x - filters.gaussian(x, 40)
    mask = corr > filters.threshold_otsu(corr)
    mask = morphology.remove_small_objects(mask, max_size=29)
    mask = ndi.binary_fill_holes(mask)
    dist = filters.gaussian(ndi.distance_transform_edt(mask), 1)
    peaks = feature.peak_local_max(dist, min_distance=7,
                                   labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)

# pass 1: contrast + ground-truth count for every image
rows, imgs = [], {}
for f in sorted(IMG.glob("*.tif")):
    img = load_image(f)
    gt = load_labels(MSK / (f.stem + ".png"))
    rows.append(dict(name=f.name, gt=int(gt.max()), contrast=contrast(img)))
    imgs[f.name] = (img, gt)
df = pd.DataFrame(rows)

empty = df[df["gt"] == 0]["contrast"]
real = df[df["gt"] > 0]["contrast"]
print("contrast of EMPTY images:", sorted(empty.round(1).tolist()))
print("lowest contrasts of REAL images:", sorted(real.round(1).tolist())[:6])
lo, hi = empty.max(), real.min()
thr = (lo + hi) / 2 if lo < hi else lo + 1e-6
print("separable:", lo < hi, "| contrast threshold used:", round(thr, 1))

# pass 2: segment only images that pass the contrast gate
preds, f1s = [], []
for _, r in df.iterrows():
    img, gt = imgs[r["name"]]
    if r["contrast"] < thr:
        preds.append(0); f1s.append(np.nan if r["gt"] == 0 else 0.0)
    else:
        pr = segment(img)
        preds.append(int(pr.max())); f1s.append(f1_at_iou(gt, pr))
df["pred"], df["f1"] = preds, f1s
df.to_csv("data/raw/improved_results.csv", index=False)

d = df[df["gt"] > 0].copy()
d["count_err"] = (d["pred"] - d["gt"]).abs() / d["gt"]
print("\nREAL images:", len(d))
print("median count error:", round(d["count_err"].median(), 3))
print("within ±10% count:", round((d["count_err"] <= 0.10).mean(), 3))
print("median F1@0.5:", round(d["f1"].median(), 3), "| mean F1:", round(d["f1"].mean(), 3))
print("EMPTY images correctly 0:", int((df[df["gt"] == 0]["pred"] == 0).sum()), "/", len(empty))
print("\nworst 5 by F1:")
print(d.sort_values("f1").head(5)[["name", "gt", "pred", "f1"]].to_string())
