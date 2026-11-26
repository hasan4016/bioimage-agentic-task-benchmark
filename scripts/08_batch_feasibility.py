import numpy as np, pandas as pd
from scipy import stats
from skimage import filters
from common import IMG, MSK, load_image, load_labels
from ref_method import segment_ref, segment_naive

def degrade(img, rng):
    """Session-B artifact: illumination ramp, blur, raised background, noise."""
    h, w = img.shape
    p = np.percentile(img, 99.5)
    ramp = np.linspace(0.7, 1.1, w)[None, :]
    if rng.random() < 0.5:
        ramp = ramp[:, ::-1]
    x = filters.gaussian(img * ramp, 1.5, preserve_range=True)
    x = x + 0.10 * p + rng.normal(0, 0.04 * p, img.shape)
    return np.clip(x, 0, 4095).astype(np.float32)

def count(fn, img):
    return int(fn(img).max())

rows = []
for i, f in enumerate(sorted(IMG.glob("*.tif"))):
    gt = int(load_labels(MSK / (f.stem + ".png")).max())
    if gt == 0:
        continue                      # skip the 3 blank fields for this test
    img = load_image(f)
    d = degrade(img, np.random.default_rng(i))
    rows.append(dict(name=f.name, gt=gt,
                     naive_A=count(segment_naive, img), ref_A=count(segment_ref, img),
                     naive_B=count(segment_naive, d), ref_B=count(segment_ref, d)))
df = pd.DataFrame(rows)
df.to_csv("data/raw/batch_counts.csv", index=False)

print("mean relative count error (pred-gt)/gt:")
for c in ["naive_A", "ref_A", "naive_B", "ref_B"]:
    print(f"  {c:8s} {((df[c] - df['gt']) / df['gt']).mean():+.3f}")

def ols_treat(y, treated, sess):
    X = np.column_stack([np.ones(len(y)), treated, sess])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    t = beta[1] / np.sqrt(cov[1, 1])
    return beta[1], 2 * stats.t.sf(abs(t), dof)

n = len(df)
out = []
for seed in range(200):
    rng = np.random.default_rng(1000 + seed)
    sess = (rng.permutation(n) < n // 2).astype(int)           # half the images in B
    treated = (rng.random(n) < np.where(sess == 1, 0.8, 0.2)).astype(int)
    obs = {"true": df["gt"].to_numpy(float),
           "naive": np.where(sess == 1, df["naive_B"], df["naive_A"]).astype(float),
           "ref": np.where(sess == 1, df["ref_B"], df["ref_A"]).astype(float)}
    r = {"seed": seed}
    for k, y in obs.items():
        r[k + "_diff"] = y[treated == 1].mean() - y[treated == 0].mean()
        r[k + "_raw_p"] = stats.ttest_ind(y[treated == 1], y[treated == 0],
                                          equal_var=False).pvalue
        r[k + "_adj_p"] = ols_treat(y, treated, sess)[1]
    out.append(r)
r = pd.DataFrame(out)

print("\nover", len(r), "random label assignments (treated minus control, nuclei/image):")
for k in ["true", "naive", "ref"]:
    print(f"  {k:6s} median diff {r[k + '_diff'].median():+6.1f} | "
          f"raw p<0.05: {(r[k + '_raw_p'] < 0.05).mean():.2f} | "
          f"session-adjusted p<0.05: {(r[k + '_adj_p'] < 0.05).mean():.2f}")
good = (r["naive_raw_p"] < 0.01) & (r["true_raw_p"] > 0.2) & (r["true_adj_p"] > 0.2)
print("assignments with a FALSE naive effect (p<0.01) but no true effect (p>0.2):",
      int(good.sum()), "of", len(r))
