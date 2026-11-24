import numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io, measure

IMG = Path("data/raw/images/images")
MSK = Path("data/raw/masks/masks")

def load_labels(mask_path):
    """Decode a BBBC039 PNG mask into an integer label image (0 = background)."""
    m = io.imread(mask_path)
    if m.ndim == 3:
        m = m[..., :3].astype(np.uint32)
        flat = (m[..., 0] << 16) | (m[..., 1] << 8) | m[..., 2]
    else:
        flat = m.astype(np.uint32)
    labels = np.zeros(flat.shape, np.int32)
    n = 0
    for v in np.unique(flat):
        if v == 0:
            continue
        cc = measure.label(flat == v, connectivity=1)
        for k in range(1, cc.max() + 1):
            n += 1
            labels[cc == k] = n
    return labels, m.shape, m.dtype, len(np.unique(flat))

files = sorted(IMG.glob("*.tif"))[:3]
fig, axes = plt.subplots(len(files), 2, figsize=(10, 4 * len(files)))

for i, f in enumerate(files):
    img = tifffile.imread(f)
    labels, shp, dt, ncol = load_labels(MSK / (f.stem + ".png"))
    areas = np.bincount(labels.ravel())[1:]
    print(f"{f.name[:30]}...  img {img.shape} {img.dtype} max={img.max()}")
    print(f"   mask raw {shp} {dt}, unique values={ncol}")
    print(f"   nuclei={labels.max()}  median area={np.median(areas):.0f}px  "
          f"min={areas.min()}  max={areas.max()}")
    axes[i, 0].imshow(img, cmap="gray", vmax=np.percentile(img, 99.5))
    axes[i, 0].set_title("image")
    axes[i, 1].imshow(labels, cmap="nipy_spectral", interpolation="nearest")
    axes[i, 1].set_title(f"expert mask: {labels.max()} nuclei")
    for a in axes[i]:
        a.axis("off")

plt.tight_layout()
plt.savefig("data/raw/preview.png", dpi=100)
print("saved data/raw/preview.png")
