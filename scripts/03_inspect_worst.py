import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage import io
from common import IMG, MSK, load_image, load_labels

names = ["IXMtest_F13_s7", "IXMtest_L01_s2", "IXMtest_L10_s6",
         "IXMtest_F22_s6", "IXMtest_A12_s7"]
fig, axes = plt.subplots(len(names), 2, figsize=(10, 4 * len(names)))
for i, n in enumerate(names):
    f = next(IMG.glob(n + "*.tif"))
    img = load_image(f)
    m = io.imread(MSK / (f.stem + ".png"))
    print(n, "| img max", int(img.max()), "p50", int(np.percentile(img, 50)),
          "p99", int(np.percentile(img, 99)), "| mask shape", m.shape)
    for c in range(m.shape[-1]):
        print("   mask channel", c, "values:", np.unique(m[..., c])[:8])
    axes[i, 0].imshow(img, cmap="gray", vmax=np.percentile(img, 99.5))
    axes[i, 0].set_title(n)
    axes[i, 1].imshow(load_labels(MSK / (f.stem + ".png")), cmap="nipy_spectral")
    axes[i, 1].set_title("expert mask")
    for a in axes[i]:
        a.axis("off")
plt.tight_layout()
plt.savefig("data/raw/worst.png", dpi=80)
