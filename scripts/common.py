import numpy as np, tifffile
from pathlib import Path
from skimage import io, measure

IMG = Path("data/raw/images/images")
MSK = Path("data/raw/masks/masks")

def load_image(path):
    return tifffile.imread(path).astype(np.float32)

def load_labels(mask_path):
    """Label value is in the red channel (0 = background); split each value into blobs."""
    flat = io.imread(mask_path)[..., 0].astype(np.int32)
    labels = np.zeros(flat.shape, np.int32)
    n = 0
    for v in np.unique(flat):
        if v == 0:
            continue
        cc = measure.label(flat == v, connectivity=1)
        labels[cc > 0] = cc[cc > 0] + n
        n += cc.max()
    return labels

def f1_at_iou(gt, pr, thr=0.5):
    ngt, npr = int(gt.max()), int(pr.max())
    if ngt == 0 or npr == 0:
        return 0.0
    idx = gt.ravel().astype(np.int64) * (npr + 1) + pr.ravel()
    inter = np.bincount(idx, minlength=(ngt + 1) * (npr + 1)).reshape(ngt + 1, npr + 1)
    agt, apr = inter.sum(1), inter.sum(0)
    iou = inter / (agt[:, None] + apr[None, :] - inter + 1e-9)
    tp = int((iou[1:, 1:] > thr).sum())
    prec, rec = tp / npr, tp / ngt
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)
