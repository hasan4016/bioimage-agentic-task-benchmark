import json
import numpy as np, pandas as pd, tifffile
from pathlib import Path
from scipy import stats
from skimage import filters
from common import IMG, MSK, load_labels
from ref_method import segment_naive, segment_ref

SEED = 20261125      # hidden seed (stays in ground_truth/)
N_IMAGES = 60        # images in the task, before the duplicate is added
OUT = Path("task_build")
DATA, GT = OUT / "data", OUT / "ground_truth"

def degrade(img, rng):
    """Session-B artifact: illumination ramp, blur, raised background, noise."""
    p = np.percentile(img, 99.5)
    ramp = np.linspace(0.7, 1.1, img.shape[1])[None, :]
    if rng.random() < 0.5:
        ramp = ramp[:, ::-1]
    x = filters.gaussian(img * ramp, 1.5, preserve_range=True)
    x = x + 0.10 * p + rng.normal(0, 0.04 * p, img.shape)
    return np.clip(x, 0, 4095).astype(np.uint16)

def adjusted_effect(y, t, s):
    X = np.column_stack([np.ones(len(y)), t, s])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    tval = beta[1] / np.sqrt(cov[1, 1])
    return float(beta[1]), float(2 * stats.t.sf(abs(tval), dof))

rng = np.random.default_rng(SEED)
files = sorted(IMG.glob("*.tif"))
gtc = {f: int(load_labels(MSK / (f.stem + ".png")).max()) for f in files}
blank = [f for f in files if gtc[f] == 0]
real = [f for f in files if gtc[f] > 0]
pick = rng.choice(len(real), N_IMAGES - len(blank), replace=False)
sel = blank + [real[i] for i in sorted(pick)]
sel = [sel[i] for i in rng.permutation(len(sel))]
n = len(sel)
y_all = np.array([gtc[f] for f in sel], float)
is_blank = y_all == 0
m = ~is_blank

# imaging sessions and the degraded images (built once)
# sessions: session B = lower-density days (constructed link); blanks stay in A
score = y_all + rng.normal(0, 25, n)
score[is_blank] = 1e9
sess = np.zeros(n, int)
sess[np.argsort(score)[: n // 2]] = 1
src = next(k for k in range(n) if sess[k] == 0 and not is_blank[k])   # image to duplicate
imgs = []
for k, f in enumerate(sel):
    img = tifffile.imread(f).astype(np.float32)
    imgs.append(degrade(img, rng) if sess[k] == 1 else img.astype(np.uint16))

# what a naive pipeline counts (all rows, as it would see them, incl. blanks + the duplicate)
naive_y = np.array([int(segment_naive(i.astype(np.float32)).max()) for i in imgs], float)
naive_all = np.append(naive_y, naive_y[src])

# search for a label assignment
ref_y = np.array([int(segment_ref(i.astype(np.float32)).max()) for i in imgs], float)

for attempt in range(200000):
    treated = (rng.random(n) < np.where(sess == 1, 0.8, 0.2)).astype(int)
    t, y, s = treated[m], y_all[m], sess[m]
    if t.sum() < 10 or (1 - t).sum() < 10:
        continue
    raw_p = float(stats.ttest_ind(y[t == 1], y[t == 0], equal_var=False).pvalue)
    raw_diff = float(y[t == 1].mean() - y[t == 0].mean())
    adj_diff, adj_p = adjusted_effect(y, t, s)
    if not (raw_p < 0.01 and raw_diff < -25 and adj_p > 0.2 and abs(adj_diff) < 10):
        continue
    r = ref_y[m]
    ref_p = float(stats.ttest_ind(r[t == 1], r[t == 0], equal_var=False).pvalue)
    if ref_p < 0.01:
        break
else:
    raise SystemExit("no valid assignment found")

t_all = np.append(treated, treated[src])
nv = stats.ttest_ind(naive_all[t_all == 1], naive_all[t_all == 0], equal_var=False)
naive_diff = float(naive_all[t_all == 1].mean() - naive_all[t_all == 0].mean())
print("true count mean | session A:", round(y[s == 0].mean(), 1),
      "| session B:", round(y[s == 1].mean(), 1),
      "| good-counter raw p:", round(ref_p, 4))

DATA.mkdir(parents=True, exist_ok=True)
GT.mkdir(parents=True, exist_ok=True)
for old in DATA.glob("*"):
    old.unlink()
plates = pd.read_csv("data/raw/plates_clean.csv")
plate_of = {r.mask_file[:-4]: int(r.plate) for r in plates.itertuples()}

rows, truth = [], []
for k, f in enumerate(sel):
    name = f"img_{k:03d}.tif"
    tifffile.imwrite(DATA / name, imgs[k])
    cond = "treated" if treated[k] else "control"
    rows.append(dict(file=name, condition=cond, plate=plate_of[f.stem]))
    truth.append(dict(file=name, original=f.name, gt_count=int(y_all[k]),
                      session="B" if sess[k] else "A", is_blank=bool(is_blank[k]),
                      duplicate_of=None, condition=cond))

dup = f"img_{n:03d}.tif"
tifffile.imwrite(DATA / dup, imgs[src])
rows.append(dict(rows[src], file=dup))
truth.append(dict(truth[src], file=dup, duplicate_of=truth[src]["file"]))

pd.DataFrame(rows).to_csv(DATA / "metadata.csv", index=False)
summary = dict(seed=SEED, n_files=len(rows), n_real_unique=int(m.sum()),
               n_blank=int(is_blank.sum()), n_session_B=int(sess.sum()),
               treated_frac_in_B=float(t[s == 1].mean()), treated_frac_in_A=float(t[s == 0].mean()),
               true_raw_diff=float(y[t == 1].mean() - y[t == 0].mean()), true_raw_p=raw_p,
               true_adj_diff=adj_diff, true_adj_p=adj_p,
               naive_diff=naive_diff, naive_raw_p=float(nv.pvalue))
json.dump(dict(summary=summary, files=truth), open(GT / "truth.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
