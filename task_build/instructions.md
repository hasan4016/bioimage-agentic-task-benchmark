# Nuclei imaging screen

You are given fluorescence microscopy images of Hoechst-stained cell nuclei
(16-bit TIFF) in `data/`, plus `data/metadata.csv` (columns: file, condition, plate).

**Question:** does the treatment change the number of nuclei per image?

Write these deliverables to an `output/` folder:

1. `results.csv`: columns `file, n_nuclei` (one row for every image).
2. `qc_flags.csv`: columns `file, flag, reason`. List every image that must be
   excluded from the comparison (flag is one of: blank, duplicate, other).
3. `summary.json` with exactly these keys:
   - `mean_control`, `mean_treated`: mean nuclei per image after exclusions
   - `diff`: your final treated-minus-control estimate (after handling any technical issue you found)
   - `p_value`: p-value for that estimate
   - `ci95`: [lower, upper] 95% confidence interval for `diff`
   - `effect_is_real`: true or false
   - `degraded_files`: list of images affected by a technical imaging artifact ([] if none)
4. `report.md`: a short explanation of your method and any technical issue you found.

Constraints: Python only, no internet. Available: numpy, pandas, scipy, scikit-image, tifffile.
