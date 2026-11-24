import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology, feature, measure, segmentation
from common import IMG, MSK, load_image, load_labels, f1_at_iou

GATE = 200  # blank-field rule: (p99.5 - median) < 200  -> report 0 nuclei

def contrast(img):
    return float(np.percentile(img, 99.5) - np.median(img))

def prep(img, norm):
    x = np.clip(img - np.median(img), 0, None)
    if norm == "linear":
        return x / np.percentile(x, 99.8)
    if norm == "clip":                      # cap bright blobs at the 99th percentile
        p = np.percentile(x, 99)
        return np.clip(x, 0, p) / p
    if norm == "log":
        return np.log1p(x)

def segment(img, norm, thr_method):
    x = filters.gaussian(prep(img, norm), 1)
    corr = x - filters.gaussian(x, 40)
    t = filters.threshold_otsu(corr) if thr_method == "otsu" else filters.threshold_triangle(corr)
    mask = corr > t
    mask = morphology.remove_small_objects(mask, max_size=29)
    mask = ndi.binary_fill_holes(mask)
    dist = filters.gaussian(ndi.distance_transform_edt(mask), 1)
    peaks = feature.peak_local_max(dist, min_distance=7,
                                   labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)

variants = {"linear_otsu": ("linear", "otsu"), "clip_otsu": ("clip", "otsu"),
            "log_otsu": ("log", "otsu"), "clip_triangle": ("clip", "triangle")}

rows = []
for f in sorted(IMG.glob("*.tif")):
    img = load_image(f)
    gt = load_labels(MSK / (f.stem + ".png"))
    if gt.max() == 0 or contrast(img) < GATE:
        continue                      # blank fields handled by the gate; evaluate real images
    for name, (norm, th) in variants.items():
        pr = segment(img, norm, th)
        rows.append(dict(variant=name, image=f.name, gt=int(gt.max()),
                         pred=int(pr.max()), f1=f1_at_iou(gt, pr)))

df = pd.DataFrame(rows)
df["count_err"] = (df["pred"] - df["gt"]).abs() / df["gt"]
df.to_csv("data/raw/variant_results.csv", index=False)

summary = df.groupby("variant").agg(
    n=("f1", "size"),
    median_count_err=("count_err", "median"),
    within10=("count_err", lambda s: (s <= 0.10).mean()),
    median_f1=("f1", "median"),
    mean_f1=("f1", "mean"),
    bad_f1_lt_0_5=("f1", lambda s: (s < 0.5).sum()),
).round(3)
print(summary.to_string())
