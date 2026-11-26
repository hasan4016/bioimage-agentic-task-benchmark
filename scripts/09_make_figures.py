"""Generate documentation figures and tables from existing task artifacts."""
import json
import numpy as np, pandas as pd, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import filters, segmentation
from common import MSK, load_labels
from grader import grade
from ref_method import contrast

DOCS = Path("docs"); FIG = DOCS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DATA = Path("task_build/data")
T = json.load(open("task_build/ground_truth/truth.json"))
S = T["summary"]
tr = pd.DataFrame(T["files"]).set_index("file")
tr["dup"] = tr["duplicate_of"].notna()
uniq = tr[~tr["dup"]]
real = uniq[~uniq["is_blank"]]
OUTS = {"Naive": Path("task_build/output_naive"),
        "Partial": Path("task_build/output_partial"),
        "Reference": Path("task_build/output_ref")}
OUTS = {k: v for k, v in OUTS.items() if (v / "summary.json").exists()}
COL = {"Naive": "#d95f02", "Partial": "#7570b3", "Reference": "#1b9e77"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)
    print("saved", FIG / name)

def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)

# ---- Fig 1: example images -------------------------------------------------
def panel(ax, f, outline):
    img = tifffile.imread(DATA / f).astype(float)
    g = np.clip(img / max(np.percentile(img, 99.5), 1), 0, 1)
    rgb = np.dstack([g, g, g])
    if outline:
        lab = load_labels(MSK / (Path(tr.loc[f, "original"]).stem + ".png"))
        rgb[segmentation.find_boundaries(lab, mode="outer")] = [1, 0.15, 0.15]
    ax.imshow(rgb); ax.axis("off")

ex = [("Session A (clean)", real[real["session"] == "A"].index[0]),
      ("Session B (degraded)", real[real["session"] == "B"].index[0]),
      ("Blank field", uniq[uniq["is_blank"]].index[0])]
fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
for j, (title, f) in enumerate(ex):
    for i in range(2):
        panel(axes[i, j], f, outline=(i == 1))
    axes[0, j].set_title(f"{title}\n{f}, expert count {int(tr.loc[f, 'gt_count'])}")
save(fig, "fig1_examples.png")

# ---- Fig 2: segmentation variants (needs scripts/05 output) ----------------
var_tab = None
vp = Path("data/raw/variant_results.csv")
if vp.exists():
    v = pd.read_csv(vp)
    order = [x for x in ["linear_otsu", "clip_otsu", "clip_triangle", "log_otsu"]
             if x in set(v["variant"])]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].boxplot([v.loc[v["variant"] == o, "f1"] for o in order])
    ax[0].set_xticks(range(1, len(order) + 1)); ax[0].set_xticklabels(order, rotation=20)
    ax[0].set_ylabel("F1 at IoU 0.5 (per image)")
    w10 = [(v.loc[v["variant"] == o, "count_err"] <= 0.10).mean() for o in order]
    ax[1].bar(order, w10, color="#4c72b0"); ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("fraction of images within ±10% of expert count")
    ax[1].tick_params(axis="x", rotation=20)
    save(fig, "fig2_segmentation_variants.png")
    var_tab = pd.DataFrame([dict(variant=o, n=int((v["variant"] == o).sum()),
        median_F1=round(v.loc[v["variant"] == o, "f1"].median(), 3),
        mean_F1=round(v.loc[v["variant"] == o, "f1"].mean(), 3),
        within_10pct=round((v.loc[v["variant"] == o, "count_err"] <= 0.10).mean(), 3),
        F1_below_0_5=int((v.loc[v["variant"] == o, "f1"] < 0.5).sum())) for o in order])
else:
    print("skip fig2: run scripts/05_compare_methods.py first")

# ---- Fig 3: predicted vs expert counts -------------------------------------
fig, axes = plt.subplots(1, len(OUTS), figsize=(3.7 * len(OUTS), 3.9),
                         sharex=True, sharey=True)
for ax, (name, out) in zip(np.atleast_1d(axes), OUTS.items()):
    res = pd.read_csv(out / "results.csv").set_index("file")
    j = uniq.join(res["n_nuclei"])
    for ses, mk in [("A", "o"), ("B", "^")]:
        d = j[j["session"] == ses]
        ax.scatter(d["gt_count"], d["n_nuclei"], s=14, marker=mk, color=COL[name],
                   alpha=0.7, label=f"session {ses}")
    xs = np.linspace(0, 5000, 600)
    ax.plot(xs, xs, "k-", lw=0.6)
    ax.plot(xs, 1.1 * xs, "k:", lw=0.5); ax.plot(xs, 0.9 * xs, "k:", lw=0.5)
    ax.set_xscale("symlog", linthresh=10); ax.set_yscale("symlog", linthresh=10)
    ax.set_xlim(0, 400); ax.set_ylim(0, 5000)
    ax.set_title(name); ax.set_xlabel("expert count")
    ax.legend(frameon=False, fontsize=7)
np.atleast_1d(axes)[0].set_ylabel("solver count")
save(fig, "fig3_counts.png")

# ---- Fig 4: the confound ---------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
frac = real.groupby("session")["condition"].apply(lambda s: (s == "treated").mean())
nn = real.groupby("session").size()
ax[0].bar(range(len(frac)), frac.values, color=["#999999", "#444444"][:len(frac)])
ax[0].set_xticks(range(len(frac))); ax[0].set_xticklabels([f"session {s}" for s in frac.index])
for x, (fv, k) in enumerate(zip(frac.values, nn.values)):
    ax[0].text(x, fv + 0.02, f"{fv:.0%} (n={k})", ha="center")
ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("fraction of images labelled treated")
rng = np.random.default_rng(0)
groups = [("A", "control"), ("A", "treated"), ("B", "control"), ("B", "treated")]
for x, (ses, cond) in enumerate(groups):
    y = real[(real["session"] == ses) & (real["condition"] == cond)]["gt_count"]
    ax[1].scatter(x + rng.uniform(-0.15, 0.15, len(y)), y, s=14, alpha=0.7,
                  color="#4c72b0" if ses == "A" else "#c44e52")
    if len(y):
        ax[1].hlines(y.mean(), x - 0.3, x + 0.3, color="k", lw=1.5)
ax[1].set_xticks(range(4))
ax[1].set_xticklabels([f"{s}\n{c}" for s, c in groups])
ax[1].set_ylabel("expert nucleus count per image")
save(fig, "fig4_confound.png")

# ---- Fig 5: forest plot of effect estimates --------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
sm = {k: json.load(open(v / "summary.json")) for k, v in OUTS.items()}
for y, (name, s) in enumerate(sm.items()):
    lo, hi = s["ci95"]
    ax.errorbar(s["diff"], y, xerr=[[s["diff"] - lo], [hi - s["diff"]]],
                fmt="o", color=COL[name], capsize=3)
ax.set_yticks(range(len(sm))); ax.set_yticklabels(list(sm.keys()))
ax.axvline(0, color="#888888", lw=0.8)
ax.axvline(S["true_adj_diff"], color="#1b9e77", ls="--",
           label="expert counts, session-adjusted")
ax.axvline(S["true_raw_diff"], color="#555555", ls=":",
           label="expert counts, unadjusted")
ax.set_xlabel("treated minus control (nuclei per image), point and 95% CI")
ax.legend(frameon=False, fontsize=7)
save(fig, "fig5_effect_estimates.png")

# ---- Fig 6: score breakdown ------------------------------------------------
comps = ["counts", "blank", "duplicate", "degraded", "conclusion",
         "effect_size", "consistency", "format"]
maxp = dict(counts=25, blank=10, duplicate=10, degraded=15, conclusion=20,
            effect_size=10, consistency=5, format=5)
scores = {k: grade(str(v)) for k, v in OUTS.items()}
fig, ax = plt.subplots(figsize=(9, 3.4))
wd = 0.8 / max(len(scores), 1)
for i, (name, sc) in enumerate(scores.items()):
    ax.bar(np.arange(len(comps)) + i * wd,
           [sc[c] / maxp[c] for c in comps], wd, label=f"{name} ({sc['total']:.1f}/100)",
           color=COL[name])
ax.set_xticks(np.arange(len(comps)) + wd * (len(scores) - 1) / 2)
ax.set_xticklabels([f"{c}\n({maxp[c]} pts)" for c in comps], fontsize=7)
ax.set_ylabel("fraction of points earned"); ax.set_ylim(0, 1.1)
ax.legend(frameon=False, fontsize=8)
save(fig, "fig6_score_breakdown.png")

# ---- Fig 7: artifact detection feature -------------------------------------
def noise_ratio(f):
    x = tifffile.imread(DATA / f).astype(np.float32)
    hp = x - filters.gaussian(x, 1, preserve_range=True)
    sigma = 1.4826 * np.median(np.abs(hp - np.median(hp)))
    return sigma / max(contrast(x), 1.0)

feat = pd.Series({f: np.log(noise_ratio(f)) for f in real.index})
thr = filters.threshold_otsu(feat.to_numpy())
is_b = (real["session"] == "B")
acc = float(((feat > thr) == is_b).mean())
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.linspace(feat.min(), feat.max(), 25)
ax.hist(feat[~is_b], bins=bins, alpha=0.7, color="#4c72b0", label="session A")
ax.hist(feat[is_b], bins=bins, alpha=0.7, color="#c44e52", label="session B")
ax.axvline(thr, color="k", ls="--", label="Otsu threshold")
ax.set_xlabel("log(noise / contrast) per image"); ax.set_ylabel("images")
ax.legend(frameon=False)
save(fig, "fig7_artifact_detection.png")

# ---- tables + RESULTS.md ---------------------------------------------------
comp_tab = pd.DataFrame({"composition": [
    "files given to the agent", "unique real images", "blank fields", "exact duplicates",
    "session A (real images)", "session B (real images)",
    "treated / control (real images)",
    "treated fraction in session A", "treated fraction in session B"],
    "value": [len(tr), len(real), int(uniq["is_blank"].sum()), int(tr["dup"].sum()),
              int((real["session"] == "A").sum()), int((real["session"] == "B").sum()),
              f"{int((real['condition'] == 'treated').sum())} / "
              f"{int((real['condition'] == 'control').sum())}",
              f"{frac.get('A', float('nan')):.2f}", f"{frac.get('B', float('nan')):.2f}"]})

score_tab = pd.DataFrame({"check (max points)": [f"{c} ({maxp[c]})" for c in comps] + ["TOTAL (100)"],
                          **{k: [sc[c] for c in comps] + [sc["total"]] for k, sc in scores.items()}})

rows = [dict(analysis="expert counts, unadjusted", diff=round(S["true_raw_diff"], 1),
             ci95="-", p=round(S["true_raw_p"], 4), says_effect_real="-"),
        dict(analysis="expert counts, session-adjusted (ground truth)",
             diff=round(S["true_adj_diff"], 1), ci95="-", p=round(S["true_adj_p"], 3),
             says_effect_real="False")]
for k, s in sm.items():
    rows.append(dict(analysis=f"{k} solver", diff=round(s["diff"], 1),
                     ci95=f"[{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}]",
                     p=round(s["p_value"], 4), says_effect_real=s["effect_is_real"]))
eff_tab = pd.DataFrame(rows)

txt = ["# Results", "",
       "_Generated by `scripts/09_make_figures.py`. Do not edit by hand._", "",
       "## Task composition", "", md(comp_tab), "",
       "## Grader scores by solver", "", md(score_tab), "",
       "## Treatment-effect estimates", "", md(eff_tab), "",
       "## Artifact detection", "",
       f"The noise-to-contrast feature with an Otsu threshold classifies "
       f"{acc:.1%} of the {len(real)} real images into the correct imaging session.", ""]
if var_tab is not None:
    txt += ["## Segmentation variants (197 non-blank BBBC039 images)", "", md(var_tab), ""]
txt += ["## Figures", ""]
for n_, t_ in [("fig1_examples.png", "Example images"),
               ("fig2_segmentation_variants.png", "Segmentation variants"),
               ("fig3_counts.png", "Solver counts vs expert counts"),
               ("fig4_confound.png", "The confound"),
               ("fig5_effect_estimates.png", "Effect estimates"),
               ("fig6_score_breakdown.png", "Score breakdown"),
               ("fig7_artifact_detection.png", "Artifact detection")]:
    if (FIG / n_).exists():
        txt += [f"![{t_}](figures/{n_})", ""]
(DOCS / "RESULTS.md").write_text("\n".join(txt))
print("\n".join(txt[:40]))
print("wrote docs/RESULTS.md")
