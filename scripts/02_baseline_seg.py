import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology, feature, measure, segmentation
from common import IMG, MSK, load_image, load_labels, f1_at_iou

def segment(img):
    img = img / np.percentile(img, 99.8)            # per-image normalization
    img = filters.gaussian(img, 1)
    corr = img - filters.gaussian(img, 40)          # background correction
    mask = corr > filters.threshold_otsu(corr)
    mask = morphology.remove_small_objects(mask, max_size=29)
    mask = ndi.binary_fill_holes(mask)
    dist = filters.gaussian(ndi.distance_transform_edt(mask), 1)
    peaks = feature.peak_local_max(dist, min_distance=7,
                                   labels=measure.label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)

rows = []
for f in sorted(IMG.glob("*.tif")):
    gt = load_labels(MSK / (f.stem + ".png"))
    pr = segment(load_image(f))
    rows.append(dict(name=f.name, gt=int(gt.max()), pred=int(pr.max()),
                     f1=f1_at_iou(gt, pr)))
df = pd.DataFrame(rows)
df["count_err"] = (df["pred"] - df["gt"]).abs() / df["gt"]
df.to_csv("data/raw/baseline_results.csv", index=False)
print(df[["gt", "pred", "count_err", "f1"]].describe().round(3))
print("within ±10% count:", (df.count_err <= 0.10).mean().round(3))
print("worst 5 by F1:")
print(df.sort_values("f1").head(5)[["name", "gt", "pred", "f1"]].to_string())
